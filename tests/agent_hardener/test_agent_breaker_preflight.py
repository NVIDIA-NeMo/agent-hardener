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

if TYPE_CHECKING:
    from pathlib import Path


def _record_probes(monkeypatch: pytest.MonkeyPatch, verdict: model_check.Validation) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []

    def fake_probe(model: str, uri: str, api_key: str | None, **kwargs: Any) -> model_check.Validation:
        probes.append({"model": model, "uri": uri, "api_key": api_key, **kwargs})
        return verdict

    monkeypatch.setattr(preflight, "probe_generation", fake_probe)
    monkeypatch.setenv("NIM_API_KEY", "nim-key")
    return probes


def test_a_shared_red_team_and_detector_model_is_probed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _record_probes(monkeypatch, model_check.Validation(ok=True))
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    asyncio.run(preflight.check_model_endpoints(config))
    assert len(probes) == 1
    assert probes[0]["api_key"] == "nim-key"
    assert probes[0]["timeout"] == 600
    assert probes[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


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
