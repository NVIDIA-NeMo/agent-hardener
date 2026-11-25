# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A hung OpenShell/Docker command is killed at ``command_timeout_s`` and reported as a named failure."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from agent_hardener.openshell.command import OpenShellCommandExecutor
from agent_hardener.openshell.config import (
    TIMEOUT_EXIT_CODE,
    OpenShellConfig,
    SubprocessCommandRunner,
)
from agent_hardener.openshell.lifecycle import OpenShellLifecycle
from agent_hardener.openshell.support import is_transient_transport_failure

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_SETTING = "victim_control.config.command_timeout_s"
_HANG = ["-c", "import sys, time; print('building', flush=True); time.sleep(30)"]


def _executor(tmp_path: Path, timeout_s: float) -> OpenShellCommandExecutor:
    config = OpenShellConfig(
        gateway="gw",
        sandbox="sb",
        policy_path=tmp_path / "policy.yaml",
        openshell_bin=sys.executable,
        command_timeout_s=timeout_s,
    )
    return OpenShellCommandExecutor(config, SubprocessCommandRunner(timeout_s), log_dir=tmp_path)


def test_runner_kills_hung_command_and_keeps_partial_output() -> None:
    result = SubprocessCommandRunner(timeout_s=0.5).run([sys.executable, *_HANG])

    assert result.returncode == TIMEOUT_EXIT_CODE
    assert result.stdout.startswith("building")
    assert "timed out after 0.5s" in result.stdout
    assert _SETTING in result.stdout


@pytest.mark.parametrize("logged", [False, True], ids=["captured", "logged"])
def test_executor_reports_timeout_as_failed_result(tmp_path: Path, logged: bool) -> None:
    executor = _executor(tmp_path, timeout_s=0.5)

    result = executor.run_logged(_HANG) if logged else executor.run(_HANG)

    assert not result.ok
    assert "timed out after 0.5s" in result.output
    assert _SETTING in result.output
    # A sandbox create that ran out its bound must not be retried as a transport blip.
    assert not is_transient_transport_failure(result.output)


def test_command_timeout_is_configurable_and_reaches_the_default_runner(tmp_path: Path) -> None:
    default = OpenShellConfig.from_mapping({"policy_path": str(tmp_path / "p.yaml")})
    custom = OpenShellConfig.from_mapping({"policy_path": str(tmp_path / "p.yaml"), "command_timeout_s": 3600})

    assert default.command_timeout_s == 1800.0
    runner = OpenShellLifecycle(custom).runner
    assert isinstance(runner, SubprocessCommandRunner)
    assert runner.timeout_s == 3600.0
