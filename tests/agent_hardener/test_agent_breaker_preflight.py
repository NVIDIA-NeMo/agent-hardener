# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener import model_check
from agent_hardener.agents.attackers.agent_breaker import preflight, run_agent_breaker
from agent_hardener.agents.attackers.agent_breaker.config import build_agent_breaker_config
from agent_hardener.errors import ModelUnavailableError
from agent_hardener.models import ARTIFACT_DIR_KEY, AgentConfig, AgentRunInput, TargetInput
from agent_hardener.runtime.stages.attack import raise_on_attacker_failure

if TYPE_CHECKING:
    from pathlib import Path


def _record_probes(monkeypatch: pytest.MonkeyPatch, verdict: model_check.Validation) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []

    def fake_probe(model: str, uri: str, api_key: str | None, **kwargs: Any) -> model_check.Validation:
        probes.append({"model": model, "uri": uri, "api_key": api_key, **kwargs})
        return verdict

    monkeypatch.setattr(model_check, "probe_generation", fake_probe)
    monkeypatch.setenv("NIM_API_KEY", "nim-key")
    return probes


def test_a_shared_red_team_and_detector_model_is_probed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _record_probes(monkeypatch, model_check.Validation(ok=True))
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    asyncio.run(preflight.check_model_endpoints(config))
    assert len(probes) == 1
    assert probes[0]["api_key"] == "nim-key"
    assert probes[0]["timeout"] == 600
    assert probes[0]["extra_body"] == {
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_the_probe_sends_json_mode_only_when_garak_will(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe without ``response_format`` passes an endpoint that then rejects every call in the scan."""
    probes = _record_probes(monkeypatch, model_check.Validation(ok=True))
    config = build_agent_breaker_config(
        target_uri="http://v",
        report_dir="d",
        report_prefix="p",
        overrides={"json_mode": False, "red_team_model_name": "gpt-oss", "detector_model_name": "gpt-oss"},
    )
    asyncio.run(preflight.check_model_endpoints(config))
    assert probes[0]["extra_body"] is None


def test_an_unanswering_model_names_its_role(monkeypatch: pytest.MonkeyPatch) -> None:
    _record_probes(monkeypatch, model_check.Validation(ok=False, reason="timeout", detail="no response within 600s"))
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    with pytest.raises(ModelUnavailableError, match="red-team and detector"):
        asyncio.run(preflight.check_model_endpoints(config))


def test_non_nim_models_are_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _record_probes(monkeypatch, model_check.Validation(ok=True))
    config = build_agent_breaker_config(
        target_uri="http://v",
        report_dir="d",
        report_prefix="p",
        overrides={"red_team_model_type": "openai.OpenAICompatible", "detector_model_name": "judge"},
    )
    asyncio.run(preflight.check_model_endpoints(config))
    assert [probe["model"] for probe in probes] == ["judge"]


def test_a_failed_preflight_stops_garak_from_starting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GARAK_COMMAND", "garak")
    started: list[list[str]] = []

    async def fake_runner(command: list[str], _log: Path, _timeout: float | None, _env: dict[str, str] | None) -> int:
        started.append(command)
        return 0

    async def failing_preflight(_config: dict[str, Any]) -> None:
        raise ModelUnavailableError("down")

    with pytest.raises(ModelUnavailableError):
        asyncio.run(
            run_agent_breaker.run_agent_breaker_scan(
                target_uri="http://victim:8000/x",
                report_dir=str(tmp_path / "r"),
                runner=fake_runner,
                preflight=failing_preflight,
            )
        )
    assert started == []


def test_a_failed_preflight_reaches_the_operator_as_a_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not as an AttackerError, whose advice (raise garak.timeout_s) is wrong for a dead model endpoint."""

    async def failing_scan(**_kwargs: Any) -> run_agent_breaker.ScanResult:
        raise ModelUnavailableError("detector down", remediation="fix the detector")

    monkeypatch.setattr(run_agent_breaker, "run_agent_breaker_scan", failing_scan)
    request = AgentRunInput(round_id="r1", target=TargetInput(name="victim", base_url="http://127.0.0.1:8000/v1"))
    agent = AgentConfig(name="garak-agent-breaker", role="attacker")
    record = asyncio.run(run_agent_breaker.run(request, agent))
    assert record.ok is False
    with pytest.raises(ModelUnavailableError, match="detector down") as exc_info:
        raise_on_attacker_failure([agent], [record])
    assert exc_info.value.remediation == "fix the detector"


@pytest.mark.parametrize(
    ("context", "probed"), [({}, True), ({ARTIFACT_DIR_KEY: "run/garak"}, False)], ids=["standalone", "in-a-run"]
)
def test_models_are_probed_only_outside_a_run_that_preflighted_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, context: dict[str, Any], probed: bool
) -> None:
    preflights: list[Any] = []

    async def fake_scan(**kwargs: Any) -> run_agent_breaker.ScanResult:
        preflights.append(kwargs["preflight"])
        return run_agent_breaker.ScanResult(return_code=0, hits=[{"prompt": "p"}], config_path="c.yaml")

    monkeypatch.setattr(run_agent_breaker, "run_agent_breaker_scan", fake_scan)
    request = AgentRunInput(
        round_id="r1", target=TargetInput(name="victim", base_url="http://127.0.0.1:8000/v1"), context=context
    )
    agent = AgentConfig(name="garak-agent-breaker", role="attacker", config={"report_dir": str(tmp_path)})
    asyncio.run(run_agent_breaker.run(request, agent))

    assert preflights == [preflight.check_model_endpoints if probed else None]
