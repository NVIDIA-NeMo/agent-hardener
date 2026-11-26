# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every model a run calls is checked before the sandbox, as the run will call it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener import model_check
from agent_hardener.errors import ModelUnavailableError
from agent_hardener.manifest import AGENT_BREAKER_IMPLEMENTATION, GUARDRAILS_DEFENDER_IMPLEMENTATION
from agent_hardener.models import SessionConfig
from agent_hardener.preflight.models import preflight_run_models

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ModelProbe

ANALYSIS = (model_check.default_model(), model_check.default_base_url(), "run-key")
FROZEN_RAILS = """\
version = 1

[[components]]
kind = "agent_hardener.pre_tool_verifier"
enabled = true

[components.config.model]
model = "frozen/judge"
base_url = "https://frozen.example/v1"
api_key_env = "AGENT_HARDENER_SAFETY_API_KEY"
"""


def _session(
    tmp_path: Path,
    *,
    guardrails: dict[str, Any] | None = None,
    garak: dict[str, Any] | None = None,
    rails: Path | None = None,
    victim_control: dict[str, Any] | None = None,
) -> SessionConfig:
    defenders = (
        [
            {
                "name": "defender-guardrails",
                "role": "defender",
                "capabilities": "guardrails",
                "implementation": GUARDRAILS_DEFENDER_IMPLEMENTATION,
                "config": guardrails,
            }
        ]
        if guardrails is not None
        else []
    )
    attackers = (
        [
            {
                "name": "garak-agent-breaker",
                "role": "attacker",
                "implementation": AGENT_BREAKER_IMPLEMENTATION,
                "config": garak,
            }
        ]
        if garak is not None
        else []
    )
    return SessionConfig(
        storage={"root_dir": tmp_path},
        target={"name": "victim", "agent_relay_plugins": rails},
        victim={"name": "victim", "role": "victim", "implementation": "pkg:run"},
        attackers=attackers,
        defenders=defenders,
        victim_control=victim_control or {},
    )


def test_a_judge_sharing_the_analysis_model_endpoint_and_key_is_asked_once(
    tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    preflight_run_models(_session(tmp_path, guardrails={}))

    assert model_probes == [ANALYSIS]


def test_the_judge_is_checked_with_the_chosen_safety_model_and_its_own_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    """Helix's --safety-model arrives as the guardrails defender's safety_llm, its key as the safety env var."""
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "judge-key")

    preflight_run_models(_session(tmp_path, guardrails={"safety_llm": "safety/model"}))

    assert model_probes[1] == ("safety/model", model_check.default_base_url(), "judge-key")


def test_frozen_rails_are_checked_with_the_judge_already_in_them(
    tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    """The defender only appends to a rails file that names a judge, so that judge is the one the victim calls."""
    rails = tmp_path / "plugins.toml"
    rails.write_text(FROZEN_RAILS, encoding="utf-8")

    preflight_run_models(_session(tmp_path, guardrails={"safety_llm": "safety/model"}, rails=rails))

    assert model_probes[1] == ("frozen/judge", "https://frozen.example/v1", "run-key")


def test_a_judge_reading_its_own_variable_is_checked_with_the_victims_secret(
    tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    """The sandbox gives the judge only the victim's secrets and the safety key, not the host's environment."""
    rails = tmp_path / "plugins.toml"
    rails.write_text(FROZEN_RAILS.replace("AGENT_HARDENER_SAFETY_API_KEY", "VICTIM_JUDGE_KEY"), encoding="utf-8")
    secrets = tmp_path / "victim.env"
    secrets.write_text("VICTIM_JUDGE_KEY=victim-judge-key\n", encoding="utf-8")
    sandbox = {
        "type": "openshell",
        "config": {"sandbox": "victim", "policy_path": "policy.yaml", "provider_env_file": str(secrets)},
    }

    preflight_run_models(_session(tmp_path, guardrails={}, rails=rails, victim_control=sandbox))

    assert model_probes[1] == ("frozen/judge", "https://frozen.example/v1", "victim-judge-key")


def test_a_run_without_the_guardrails_defender_does_not_check_a_judge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "judge-key")

    preflight_run_models(_session(tmp_path))

    assert model_probes == [ANALYSIS]


def test_garak_models_are_checked_from_the_attacker_settings_with_the_nim_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    """A bad attack key used to pass, then fail with a 403 once the sandbox was up and the attack began."""
    monkeypatch.chdir(tmp_path)  # no garak-scan.yaml scaffold here
    monkeypatch.setenv("NIM_API_KEY", "nim-key")

    preflight_run_models(_session(tmp_path, garak={"red_team_model_name": "attack/model"}))

    assert ("attack/model", "https://integrate.api.nvidia.com/v1/", "nim-key") in model_probes
    assert len(model_probes) == 3  # analysis, red-team, detector


def test_the_first_model_that_will_not_answer_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_probes: list[ModelProbe]
) -> None:
    monkeypatch.setattr(
        model_check,
        "probe_generation",
        lambda model, *_a, **_k: model_check.Validation(ok=model != "safety/model", reason="auth", detail="HTTP 401"),
    )

    with pytest.raises(ModelUnavailableError, match="guardrail judge"):
        preflight_run_models(_session(tmp_path, guardrails={"safety_llm": "safety/model"}))
