# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the victim-deployment uploaders."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from agent_hardener.openshell.lifecycle import OpenShellCommandResult, OpenShellConfig
from agent_hardener.runtime.adapters import OpenshellUploader, RelayPolicyUploader

if TYPE_CHECKING:
    from pathlib import Path


class FakeLifecycle:
    """Records lifecycle calls and returns canned results, so uploaders can be tested in isolation."""

    def __init__(
        self,
        config: OpenShellConfig,
        *,
        apply_ok: bool = True,
        upload_ok: bool = True,
        active_policy_output: str = "",
    ) -> None:
        self.config = config
        self.runner = object()
        self.calls: list[tuple[object, ...]] = []
        self._apply_ok = apply_ok
        self._upload_ok = upload_ok
        self._active_policy_output = active_policy_output

    def apply_policy(self, policy_path: Path) -> OpenShellCommandResult:
        self.calls.append(("apply_policy", str(policy_path)))
        return OpenShellCommandResult(ok=self._apply_ok, command=["policy", "set"], output="applied")

    def restart(self, policy_path: Path | None = None, wait_for_health: bool = True) -> OpenShellCommandResult:
        self.calls.append(("restart", None if policy_path is None else str(policy_path), wait_for_health))
        return OpenShellCommandResult(ok=True, command=["restart"], output="restarted")

    def active_policy(self) -> OpenShellCommandResult:
        self.calls.append(("active_policy",))
        return OpenShellCommandResult(ok=True, command=["policy", "get"], output=self._active_policy_output)

    def upload_file(self, local_path: Path, destination: str) -> OpenShellCommandResult:
        self.calls.append(("upload_file", str(local_path), destination))
        return OpenShellCommandResult(ok=self._upload_ok, command=["upload"], output="uploaded")

    def restart_victim(self, wait_for_health: bool = True) -> OpenShellCommandResult:
        self.calls.append(("restart_victim",))
        return OpenShellCommandResult(ok=True, command=["restart-victim"], output="victim restarted")


def _config(policy_path: Path, uploads: list[str] | None = None) -> OpenShellConfig:
    return OpenShellConfig(gateway="gw", sandbox="sb", policy_path=policy_path, uploads=uploads or [])


@pytest.mark.unit
def test_openshell_uploader_applies_in_place_and_verifies(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("version: 1\n", encoding="utf-8")
    lifecycle = FakeLifecycle(_config(tmp_path / "policy.yaml"), active_policy_output="version: 1\n")

    result = asyncio.run(OpenshellUploader(lifecycle).upload(candidate, recreate=False))

    assert result.ok is True
    assert lifecycle.calls == [("apply_policy", str(candidate)), ("active_policy",)]


@pytest.mark.unit
def test_openshell_uploader_recreates_when_requested(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("version: 1\n", encoding="utf-8")
    # policy_path == candidate so _recreate reuses this lifecycle instead of building a real one.
    lifecycle = FakeLifecycle(_config(candidate), active_policy_output="version: 1\n")

    result = asyncio.run(OpenshellUploader(lifecycle).upload(candidate, recreate=True))

    assert result.ok is True
    assert lifecycle.calls == [("restart", str(candidate), True), ("active_policy",)]


@pytest.mark.unit
def test_openshell_uploader_reports_active_policy_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("version: 1\n", encoding="utf-8")
    lifecycle = FakeLifecycle(_config(tmp_path / "policy.yaml"), active_policy_output="version: 2\n")

    result = asyncio.run(OpenshellUploader(lifecycle).upload(candidate, recreate=False))

    assert result.ok is False
    assert "does not match" in result.output


@pytest.mark.unit
def test_openshell_uploader_returns_apply_failure_without_verifying(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("version: 1\n", encoding="utf-8")
    lifecycle = FakeLifecycle(_config(tmp_path / "policy.yaml"), apply_ok=False)

    result = asyncio.run(OpenshellUploader(lifecycle).upload(candidate, recreate=False))

    assert result.ok is False
    assert lifecycle.calls == [("apply_policy", str(candidate))]  # no active_policy verification after a failed apply


@pytest.mark.unit
def test_relay_uploader_uploads_then_restarts_victim(tmp_path: Path) -> None:
    # The uploaded file is the active-state copy; the upload source is the same name in another dir.
    candidate = tmp_path / "active-state" / "workflow.yaml"
    upload_source = tmp_path / "build" / "workflow.yaml"
    lifecycle = FakeLifecycle(_config(tmp_path / "policy.yaml", uploads=[f"{upload_source}:/app/workflow.yaml"]))

    result = asyncio.run(RelayPolicyUploader(lifecycle).upload(candidate))

    assert result.ok is True
    assert lifecycle.calls == [
        ("upload_file", str(candidate), "/app/workflow.yaml"),
        ("restart_victim",),
    ]


@pytest.mark.unit
def test_relay_uploader_recreates_when_no_upload_destination(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    lifecycle = FakeLifecycle(_config(tmp_path / "policy.yaml", uploads=[]))

    result = asyncio.run(RelayPolicyUploader(lifecycle).upload(workflow))

    assert result.ok is True
    assert lifecycle.calls == [("restart", None, True)]


@pytest.mark.unit
def test_relay_uploader_returns_failed_upload_without_restart(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    lifecycle = FakeLifecycle(
        _config(tmp_path / "policy.yaml", uploads=[f"{workflow}:/app/workflow.yaml"]), upload_ok=False
    )

    result = asyncio.run(RelayPolicyUploader(lifecycle).upload(workflow))

    assert result.ok is False
    assert lifecycle.calls == [("upload_file", str(workflow), "/app/workflow.yaml")]
