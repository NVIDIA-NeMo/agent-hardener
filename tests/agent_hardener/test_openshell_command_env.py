# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-command credentials reach real child processes without changing the parent."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

from agent_hardener.openshell.command import OpenShellCommandExecutor
from agent_hardener.openshell.config import OpenShellConfig, SubprocessCommandRunner

if TYPE_CHECKING:
    from pathlib import Path


_CHECK_CHILD_ENV = (
    "import os; "
    "assert os.environ['AGENT_HARDENER_SAFETY_API_KEY'] == os.environ['EXPECTED_JUDGE_KEY']; "
    "assert os.environ['INFERENCE_API_KEY'] == os.environ['EXPECTED_VICTIM_KEY']; "
    "print('child-env-ok')"
)


@pytest.fixture
def child_credentials(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.delenv("AGENT_HARDENER_SAFETY_API_KEY", raising=False)
    monkeypatch.setenv("INFERENCE_API_KEY", "parent-analysis-key")
    return {
        **os.environ,
        "AGENT_HARDENER_SAFETY_API_KEY": "child-judge-key",
        "INFERENCE_API_KEY": "child-victim-key",
        "EXPECTED_JUDGE_KEY": "child-judge-key",
        "EXPECTED_VICTIM_KEY": "child-victim-key",
    }


def test_subprocess_runner_passes_child_credentials(child_credentials: dict[str, str]) -> None:
    result = SubprocessCommandRunner().run([sys.executable, "-c", _CHECK_CHILD_ENV], env=child_credentials)

    assert result.returncode == 0
    assert result.stdout == "child-env-ok\n"
    assert os.environ["INFERENCE_API_KEY"] == "parent-analysis-key"
    assert "AGENT_HARDENER_SAFETY_API_KEY" not in os.environ


def test_executor_passes_child_credentials_without_recording_values(
    tmp_path: Path, child_credentials: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    executor = OpenShellCommandExecutor(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", openshell_bin=sys.executable),
        SubprocessCommandRunner(),
        log_dir=tmp_path,
    )

    with caplog.at_level("INFO", logger="agent_hardener.session"):
        result = executor.run(["-c", _CHECK_CHILD_ENV], env=child_credentials)

    assert result.ok
    assert result.output == "child-env-ok\n"
    for value in ("child-judge-key", "child-victim-key"):
        assert value not in str(result.command) + result.output + caplog.text
    assert os.environ["INFERENCE_API_KEY"] == "parent-analysis-key"
    assert "AGENT_HARDENER_SAFETY_API_KEY" not in os.environ
