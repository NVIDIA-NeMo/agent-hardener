# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the structured error dump + the CLI error boundary."""

from __future__ import annotations

import json

import pytest
import typer

from agent_hardener.cli._errors import cli_error_boundary
from agent_hardener.errors import (
    ERROR_FILE_ENVVAR,
    AgentHardenerError,
    SandboxProvisioningError,
    VictimBuildError,
    emit_run_error,
    write_run_error,
)


def test_agent_hardener_error_carries_category_and_default_remediation() -> None:
    exc = SandboxProvisioningError("sandbox down")
    assert exc.category == "sandbox"
    assert exc.remediation  # a category default is filled in
    assert VictimBuildError("bad project").category == "victim_unavailable"


def test_emit_run_error_writes_structured_dump(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    emit_run_error(VictimBuildError("missing project_dir", remediation="fix project_dir"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["category"] == "victim_unavailable"
    assert payload["message"] == "missing project_dir"
    assert payload["remediation"] == "fix project_dir"


def test_emit_run_error_is_noop_without_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ERROR_FILE_ENVVAR, raising=False)
    # Must not raise when the caller did not request a dump.
    emit_run_error(AgentHardenerError("boom"))


def test_write_run_error_serializes_non_exception_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    write_run_error("sandbox", "ok=false", remediation="check docker")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"category": "sandbox", "message": "ok=false", "remediation": "check docker", "stack": ""}


def _concrete_errors(base: type[AgentHardenerError] = AgentHardenerError) -> list[type[AgentHardenerError]]:
    return [cls for sub in base.__subclasses__() for cls in (sub, *_concrete_errors(sub))]


@pytest.mark.parametrize("error_type", _concrete_errors(), ids=lambda cls: cls.__name__)
def test_boundary_writes_the_contract_the_platform_reads(
    error_type: type[AgentHardenerError], tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every classified error reaches the caller with the cause and a fix, never a bare category."""
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    with pytest.raises(typer.Exit) as exc_info, cli_error_boundary():
        raise error_type("it broke")
    assert exc_info.value.exit_code == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"category", "message", "remediation", "stack"}
    assert payload["category"] == error_type.category != AgentHardenerError.category
    assert payload["message"] == "it broke"
    assert payload["remediation"].strip()


def test_boundary_prints_bracketed_text_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    """A failure code in brackets is data; read as a rich style tag it vanished from the console."""
    with pytest.raises(typer.Exit), cli_error_boundary():
        raise SandboxProvisioningError("did not complete: [victim_unreachable] refused", remediation="see [log]")
    out = capsys.readouterr().out
    assert "[victim_unreachable] refused" in out
    assert "see [log]" in out


def test_boundary_serializes_unexpected_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    with pytest.raises(typer.Exit), cli_error_boundary():
        raise ValueError("weird")
    assert json.loads(path.read_text(encoding="utf-8"))["category"] == "unexpected"


def test_boundary_passes_typer_exit_through(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    # An already-handled exit (e.g. a fail() preflight) is control flow, not a new failure to dump.
    with pytest.raises(typer.Exit), cli_error_boundary():
        raise typer.Exit(1)
    assert not path.exists()
