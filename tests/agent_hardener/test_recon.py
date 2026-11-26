# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recon: the victim profile handed to the garak attackers before the attack stage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from agent_hardener.agents.attackers.agent_breaker import run_agent_breaker
from agent_hardener.agents.attackers.recon import (
    GARAK_STAGING_DIR,
    RECON_FILENAME,
    run_recon,
    staged_for_garak,
)
from agent_hardener.config import load_config
from agent_hardener.garak_venv import garak_user_data_dir
from agent_hardener.models import RECON_PROFILE_KEY, AgentProfile, GarakSettings, ReconSettings, ReconTool
from agent_hardener.runtime.orchestrator import Orchestrator

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_PROFILE = REPO_ROOT / "examples/langgraph-victim/recon.yaml"
TARGET = "http://victim:8000/v1/chat/completions"
PROFILE = AgentProfile(
    agent_purpose="Runs commands.", tools=[ReconTool(name="bash_executor", description="Runs bash.")]
)

_MANIFEST = (
    "agent:\n"
    "  name: demo\n"
    "  project_dir: .\n"
    "  dockerfile: Dockerfile\n"
    "  start_command: /app/serve\n"
    "  binaries: ['/app/**']\n"
    "  secrets: [INFERENCE_API_KEY]\n"
)


# --- settings -----------------------------------------------------------------------------


def test_recon_is_off_by_default() -> None:
    assert GarakSettings().recon.source == "none"


@pytest.mark.parametrize(
    "settings",
    [
        {"source": "none", "file": "recon.yaml"},
        {"source": "static"},
        {"source": "static", "file": "recon.yaml", "profile": {"tools": [{"name": "t"}]}},
        {"source": "discovery"},
    ],
)
def test_recon_settings_reject_an_inconsistent_source(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReconSettings.model_validate(settings)


def test_a_relative_recon_file_resolves_against_the_manifest_and_stays_out_of_the_attacker(tmp_path: Path) -> None:
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text(_MANIFEST + "garak:\n  recon:\n    source: static\n    file: recon.yaml\n", encoding="utf-8")
    config = load_config(str(manifest))
    assert config.garak is not None
    assert config.garak.recon.file == tmp_path / "recon.yaml"
    assert "recon" not in config.attackers[0].config


def test_recon_does_not_change_the_attacker_id(tmp_path: Path) -> None:
    plain = tmp_path / "plain.yaml"
    plain.write_text(_MANIFEST, encoding="utf-8")
    with_recon = tmp_path / "recon-manifest.yaml"
    with_recon.write_text(_MANIFEST + "garak:\n  recon:\n    source: static\n    file: recon.yaml\n", encoding="utf-8")
    assert load_config(str(plain)).attackers[0].agent_id == load_config(str(with_recon)).attackers[0].agent_id


# --- sources ------------------------------------------------------------------------------


def test_static_recon_reads_the_mock_profile() -> None:
    profile = asyncio.run(run_recon(ReconSettings(source="static", file=MOCK_PROFILE), target_url=TARGET))
    assert profile is not None
    assert [tool.name for tool in profile.tools] == ["bash_executor", "python_executor"]
    assert profile.agent_purpose


def test_static_recon_takes_an_inline_profile() -> None:
    assert asyncio.run(run_recon(ReconSettings(source="static", profile=PROFILE), target_url=TARGET)) == PROFILE


def test_static_recon_refuses_a_profile_without_tools(tmp_path: Path) -> None:
    empty = tmp_path / "recon.yaml"
    empty.write_text("agent_purpose: x\ntools: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no tools"):
        asyncio.run(run_recon(ReconSettings(source="static", file=empty), target_url=TARGET))


def test_static_recon_refuses_a_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    listed = tmp_path / "recon.yaml"
    listed.write_text("- name: t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must hold a mapping"):
        asyncio.run(run_recon(ReconSettings(source="static", file=listed), target_url=TARGET))


# --- handing the profile to garak ---------------------------------------------------------


def test_garak_user_data_dir_mirrors_xdg(tmp_path: Path) -> None:
    assert garak_user_data_dir({"XDG_DATA_HOME": str(tmp_path)}) == tmp_path / "garak" / "data"
    home = {"HOME": str(tmp_path), "XDG_DATA_HOME": "relative/ignored"}
    assert garak_user_data_dir(home) == tmp_path / ".local/share/garak/data"


def test_the_profile_is_staged_under_garaks_data_dir_and_removed_after(tmp_path: Path) -> None:
    env = {"XDG_DATA_HOME": str(tmp_path)}
    data_dir = tmp_path / "garak" / "data"
    with staged_for_garak(PROFILE, env) as name:
        assert not Path(name).is_absolute()
        assert name.startswith(f"{GARAK_STAGING_DIR}/")
        assert AgentProfile.model_validate(yaml.safe_load((data_dir / name).read_text(encoding="utf-8"))) == PROFILE
    assert not (data_dir / name).exists()
    assert list((data_dir / GARAK_STAGING_DIR).iterdir()) == []


def _scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: AgentProfile | None) -> dict[str, Any]:
    """Run one scan with a fake garak; return its resolved config and whether the profile was there."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    seen: dict[str, Any] = {}

    async def fake_runner(command: list[str], _log: Path, _timeout: float | None, _env: dict[str, str] | None) -> int:
        resolved = yaml.safe_load(Path(command[-1]).read_text(encoding="utf-8"))
        block = resolved["plugins"].get("probes", {}).get("agent_breaker", {}).get("AgentBreaker", {})
        seen["agent_config_file"] = block.get("agent_config_file")
        if seen["agent_config_file"]:
            staged = tmp_path / "xdg" / "garak" / "data" / seen["agent_config_file"]
            seen["staged"] = AgentProfile.model_validate(yaml.safe_load(staged.read_text(encoding="utf-8")))
            seen["staged_path"] = staged
        return 0

    asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri=TARGET,
            report_dir=str(tmp_path / "r"),
            garak_command=["garak"],
            runner=fake_runner,
            agent_profile=profile,
        )
    )
    return seen


def test_a_scan_with_a_profile_points_the_probe_at_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _scan(tmp_path, monkeypatch, PROFILE)
    assert seen["staged"] == PROFILE
    assert not seen["staged_path"].exists()


def test_a_scan_without_a_profile_keeps_the_probes_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _scan(tmp_path, monkeypatch, None)["agent_config_file"] is None


def test_the_attacker_reads_the_profile_from_the_run_context(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[AgentProfile | None] = []

    async def fake_scan(**kwargs: Any) -> Any:
        received.append(kwargs["agent_profile"])
        raise RuntimeError("stop after the scan is set up")

    monkeypatch.setattr(run_agent_breaker, "run_agent_breaker_scan", fake_scan)
    request = SimpleNamespace(
        target=SimpleNamespace(base_url=TARGET), context={RECON_PROFILE_KEY: PROFILE.model_dump()}
    )
    agent = SimpleNamespace(config={}, agent_id="a", name="garak-agent-breaker")
    with pytest.raises(RuntimeError, match="stop after"):
        asyncio.run(run_agent_breaker.run(request, agent))  # type: ignore[arg-type]
    assert received == [PROFILE]


# --- once per run -------------------------------------------------------------------------


def _orchestrator(recon: ReconSettings, *, attackers: bool = True, base_url: str | None = TARGET) -> SimpleNamespace:
    config = SimpleNamespace(
        garak=GarakSettings(recon=recon),
        attackers=[object()] if attackers else [],
        target=SimpleNamespace(base_url=base_url),
        context={},
    )
    return SimpleNamespace(config=config)


def test_recon_runs_once_into_the_context_and_the_run_log(tmp_path: Path) -> None:
    orchestrator = _orchestrator(ReconSettings(source="static", file=MOCK_PROFILE))
    asyncio.run(Orchestrator._recon(orchestrator, tmp_path))  # type: ignore[arg-type]
    profile = AgentProfile.model_validate(orchestrator.config.context[RECON_PROFILE_KEY])
    assert [tool.name for tool in profile.tools] == ["bash_executor", "python_executor"]
    kept = yaml.safe_load((tmp_path / RECON_FILENAME).read_text(encoding="utf-8"))
    assert AgentProfile.model_validate(kept) == profile


@pytest.mark.parametrize(
    ("recon", "attackers", "base_url"),
    [
        (ReconSettings(), True, TARGET),
        (ReconSettings(), True, None),
        (ReconSettings(source="static", file=MOCK_PROFILE), False, TARGET),
    ],
)
def test_no_recon_without_a_source_or_attackers(
    tmp_path: Path, recon: ReconSettings, attackers: bool, base_url: str | None
) -> None:
    orchestrator = _orchestrator(recon, attackers=attackers, base_url=base_url)
    asyncio.run(Orchestrator._recon(orchestrator, tmp_path))  # type: ignore[arg-type]
    assert RECON_PROFILE_KEY not in orchestrator.config.context
    assert not (tmp_path / RECON_FILENAME).exists()
