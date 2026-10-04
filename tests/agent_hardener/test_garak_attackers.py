# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_hardener.agents.attackers.agent_breaker import run_agent_breaker
from agent_hardener.agents.attackers.agent_breaker.config import (
    DETECTOR_SPEC,
    PROBE_SPEC,
    apply_runtime_fields,
    build_agent_breaker_config,
)
from agent_hardener.endpoint import EndpointContract
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput

# --- config builder -----------------------------------------------------------------------


def test_build_agent_breaker_config_defaults() -> None:
    config = build_agent_breaker_config(
        target_uri="http://victim:8000/v1/chat/completions",
        report_dir=".agent-hardener/garak_runs",
        report_prefix="agent-breaker",
    )
    plugins = config["plugins"]
    assert plugins["probe_spec"] == PROBE_SPEC
    assert plugins["extended_detectors"] == [DETECTOR_SPEC]
    assert plugins["generators"]["rest"]["RestGenerator"]["uri"] == "http://victim:8000/v1/chat/completions"
    assert config["reporting"] == {"report_dir": ".agent-hardener/garak_runs", "report_prefix": "agent-breaker"}


def test_no_status_code_is_skipped_by_default() -> None:
    """Skipping a status code by default would hide a real victim failure.

    The old ``[422]`` was a NAT-ism; Agent Hardener no longer owns the victim's server, so which codes
    mean "it broke" is the victim's business to declare.
    """
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    assert config["plugins"]["generators"]["rest"]["RestGenerator"]["skip_codes"] == []


def test_a_victim_can_declare_its_own_skip_codes() -> None:
    config = build_agent_breaker_config(
        target_uri="http://v", report_dir="d", report_prefix="p", overrides={"skip_codes": [503]}
    )
    assert config["plugins"]["generators"]["rest"]["RestGenerator"]["skip_codes"] == [503]


def test_attacker_sends_the_same_request_shape_as_replay() -> None:
    """Attacker and replay must send the same request shape.

    If the two drift, a "blocked" verdict stops being evidence about the guardrail and becomes
    evidence that two payload builders disagreed.
    """
    contract = EndpointContract(url="http://v", mode="field", input_field="query")
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p", contract=contract)
    generator = config["plugins"]["generators"]["rest"]["RestGenerator"]
    assert generator["req_template_json_object"] == {"query": "$INPUT"}
    assert generator["response_json_field"] == "$.choices[0].message.content"


def test_default_contract_is_openai_chat_completions() -> None:
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    template = config["plugins"]["generators"]["rest"]["RestGenerator"]["req_template_json_object"]
    assert template["messages"] == [{"role": "user", "content": "$INPUT"}]


def test_build_agent_breaker_config_overrides_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GARAK_DETECTOR_MODEL_NAME", "env-model")
    config = build_agent_breaker_config(
        target_uri="http://v",
        report_dir="d",
        report_prefix="p",
        overrides={"red_team_model_name": "override-model", "max_attempts_per_tool": 9},
    )
    probe = config["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]
    detector = config["plugins"]["detectors"]["agent_breaker"]["AgentBreakerResult"]
    assert probe["red_team_model_name"] == "override-model"  # explicit override wins
    assert probe["max_attempts_per_tool"] == 9
    assert detector["detector_model_name"] == "env-model"  # GARAK_* env fallback


def test_nemotron_models_run_without_reasoning() -> None:
    config = build_agent_breaker_config(target_uri="http://v", report_dir="d", report_prefix="p")
    no_thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    probe = config["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]
    detector = config["plugins"]["detectors"]["agent_breaker"]["AgentBreakerResult"]
    assert probe["red_team_model_config"]["extra_params"] == no_thinking
    assert detector["detector_model_config"]["extra_params"] == no_thinking


def test_non_nemotron_models_get_no_reasoning_toggle() -> None:
    config = build_agent_breaker_config(
        target_uri="http://v",
        report_dir="d",
        report_prefix="p",
        overrides={"red_team_model_name": "openai/gpt-oss-20b", "detector_model_name": "openai/gpt-oss-20b"},
    )
    probe = config["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]
    detector = config["plugins"]["detectors"]["agent_breaker"]["AgentBreakerResult"]
    assert "extra_params" not in probe["red_team_model_config"]
    assert "extra_params" not in detector["detector_model_config"]


def test_apply_runtime_fields_overlays_and_preserves_base() -> None:
    base = build_agent_breaker_config(target_uri="http://old", report_dir="x", report_prefix="y")
    resolved = apply_runtime_fields(base, target_uri="http://new", report_dir="rd", report_prefix="rp")
    assert resolved["plugins"]["generators"]["rest"]["RestGenerator"]["uri"] == "http://new"
    assert resolved["reporting"] == {"report_dir": "rd", "report_prefix": "rp"}
    assert base["plugins"]["generators"]["rest"]["RestGenerator"]["uri"] == "http://old"  # original untouched


# --- target uri resolution ----------------------------------------------------------------


def test_resolve_target_uri_full_override() -> None:
    assert run_agent_breaker.resolve_target_uri("http://x:1", target_uri="http://override:9") == "http://override:9"


def test_resolve_target_uri_port_swap() -> None:
    out = run_agent_breaker.resolve_target_uri("http://127.0.0.1:8000/v1/chat/completions", target_port=9001)
    assert out == "http://127.0.0.1:9001/v1/chat/completions"


def test_resolve_target_uri_default_passthrough() -> None:
    assert run_agent_breaker.resolve_target_uri("http://127.0.0.1:8000/x") == "http://127.0.0.1:8000/x"


# --- garak subprocess env -----------------------------------------------------------------


# --- garak interpreter resolution ---------------------------------------------------------


def test_garak_command_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GARAK_COMMAND", "python -m garak")
    assert run_agent_breaker._garak_command() == ["python", "-m", "garak"]


def test_garak_command_uses_dedicated_venv_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GARAK_COMMAND", raising=False)
    fake_python = tmp_path / "garak-venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.touch()
    monkeypatch.setenv("AGENT_HARDENER_GARAK_PYTHON", str(fake_python))
    assert run_agent_breaker._garak_command() == [str(fake_python), "-m", "garak"]


def test_garak_command_missing_interpreter_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GARAK_COMMAND", raising=False)
    monkeypatch.setenv("AGENT_HARDENER_GARAK_PYTHON", str(tmp_path / "absent" / "python"))
    with pytest.raises(FileNotFoundError, match="garak interpreter not found"):
        run_agent_breaker._garak_command()


def test_run_agent_breaker_scan_passes_aliased_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GARAK_COMMAND", "garak")  # skip dedicated-venv resolution in unit tests
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.setenv("INFERENCE_API_KEY", "inf-key")
    envs: list[dict[str, str] | None] = []

    async def fake_runner(_command: list[str], _log: Path, _timeout: float | None, env: dict[str, str] | None) -> int:
        envs.append(env)
        return 0

    asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri="http://victim:8000/x",
            report_dir=str(tmp_path / "r"),
            report_prefix="agent-breaker",
            runner=fake_runner,
        )
    )
    assert envs[0] is not None
    assert envs[0]["NIM_API_KEY"] == "inf-key"


# --- scan + run ---------------------------------------------------------------------------


def _write_hitlog(report_dir: Path, prefix: str, rows: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{prefix}.uuid.hitlog.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def test_run_agent_breaker_scan_spawns_and_reads_hitlog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GARAK_COMMAND", "garak")  # skip dedicated-venv resolution in unit tests
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    report_dir = tmp_path / "reports"
    commands: list[list[str]] = []
    envs: list[dict[str, str] | None] = []

    async def fake_runner(command: list[str], _log: Path, _timeout: float | None, env: dict[str, str] | None) -> int:
        commands.append(command)
        envs.append(env)
        _write_hitlog(report_dir, "agent-breaker", [{"probe": "agent_breaker.AgentBreaker"}])
        return 0

    result = asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri="http://victim:8000/v1/chat/completions",
            report_dir=str(report_dir),
            report_prefix="agent-breaker",
            runner=fake_runner,
        )
    )
    assert envs[0] is not None  # env is built explicitly, never inherited as-is
    assert envs[0]["NIM_API_KEY"] == "NOT_SET"  # no inference key to alias -> defaulted, not carried
    assert result.return_code == 0
    assert result.hits == [{"probe": "agent_breaker.AgentBreaker"}]
    resolved_path = Path(commands[0][-1])
    assert commands[0][0] == "garak"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["plugins"]["generators"]["rest"]["RestGenerator"]["uri"] == "http://victim:8000/v1/chat/completions"


def test_run_agent_breaker_scan_overlays_user_scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GARAK_COMMAND", "garak")  # skip dedicated-venv resolution in unit tests
    scaffold = tmp_path / "garak-scan.yaml"
    base = build_agent_breaker_config(target_uri="http://placeholder", report_dir="x", report_prefix="y")
    base["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]["red_team_model_name"] = "user-edited"
    scaffold.write_text(yaml.safe_dump(base), encoding="utf-8")

    async def fake_runner(_command: list[str], _log: Path, _timeout: float | None, _env: dict[str, str] | None) -> int:
        return 0

    asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri="http://victim:8000/x",
            config_path=str(scaffold),
            report_dir=str(tmp_path / "r"),
            report_prefix="agent-breaker",
            runner=fake_runner,
        )
    )
    resolved = yaml.safe_load((tmp_path / "r" / "garak-agent-breaker.resolved.yaml").read_text())
    probe = resolved["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]
    assert probe["red_team_model_name"] == "user-edited"  # user edit survives
    assert resolved["plugins"]["generators"]["rest"]["RestGenerator"]["uri"] == "http://victim:8000/x"  # uri overlaid


def test_agent_breaker_run_normalizes_records_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def fake_scan(**kwargs: Any) -> run_agent_breaker.ScanResult:
        assert kwargs["target_uri"] == "http://127.0.0.1:9001/v1/chat/completions"  # port override applied
        return run_agent_breaker.ScanResult(
            return_code=0,
            hits=[{"probe": "agent_breaker.AgentBreaker"}, "raw-hit"],
            config_path="/c.yaml",
            report_path="/r.jsonl",
        )

    monkeypatch.setattr(run_agent_breaker, "run_agent_breaker_scan", fake_scan)
    request = AgentRunInput(
        round_id="round-0001",
        target=TargetInput(name="victim", base_url="http://127.0.0.1:8000/v1/chat/completions"),
    )
    record = asyncio.run(
        run_agent_breaker.run(request, AgentConfig(name="agent-breaker", role="attacker", config={"target_port": 9001}))
    )
    assert record.ok is True
    assert record.metadata["garak_report_path"] == "/r.jsonl"
    assert record.metadata["target_uri"] == "http://127.0.0.1:9001/v1/chat/completions"
    assert record.records == [
        {"probe": "agent_breaker.AgentBreaker", "source": "garak-agent-breaker", "hit_index": 0},
        {"value": "raw-hit", "source": "garak-agent-breaker", "hit_index": 1},
    ]


def test_default_request_timeout_fits_a_sub_agent_turn():
    """One attack turn, not the whole scan — ``garak.timeout_s`` is the separate, outer cap.

    Measured on a healthy DeepAgents victim: a ``read_customer_record`` turn takes 12s,
    ``write_file`` 34s, and a ``task`` turn — which spawns a sub-agent — 141s. At the old 240s a
    single ``task`` attack timed out and failed the whole attacker, scoring every remaining attack
    as unrun rather than as a finding.
    """
    config = build_agent_breaker_config(
        target_uri="http://127.0.0.1:8000/v1/chat/completions",
        report_dir="/tmp",
        report_prefix="p",
    )
    rest = config["plugins"]["generators"]["rest"]["RestGenerator"]
    assert rest["request_timeout"] == 600


def test_request_timeout_is_overridable():
    """Turn duration is a property of the victim, so the cap must be settable per victim."""
    config = build_agent_breaker_config(
        target_uri="http://127.0.0.1:8000/v1/chat/completions",
        report_dir="/tmp",
        report_prefix="p",
        overrides={"request_timeout": 900},
    )
    rest = config["plugins"]["generators"]["rest"]["RestGenerator"]
    assert rest["request_timeout"] == 900
