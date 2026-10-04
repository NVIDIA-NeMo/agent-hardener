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


def test_boundary_serializes_agent_hardener_error_and_exits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run-error.json"
    monkeypatch.setenv(ERROR_FILE_ENVVAR, str(path))
    with pytest.raises(typer.Exit) as exc_info, cli_error_boundary():
        raise SandboxProvisioningError("docker down")
    assert exc_info.value.exit_code == 1
    assert json.loads(path.read_text(encoding="utf-8"))["category"] == "sandbox"


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
