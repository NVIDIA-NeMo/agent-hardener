# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenShell gateway registration and its wiring into `agent-hardener setup`."""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from agent_hardener.cli import app
from agent_hardener.openshell import gateway

pytestmark = pytest.mark.unit

runner = CliRunner()


def _cp(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class FakeRun:
    """Records commands and scripts responses for the module-level `_run` seam."""

    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "info"]:
            return _cp(0)
        if cmd[:3] == ["docker", "context", "inspect"]:
            return _cp(0, stdout="unix:///var/run/docker.sock\n")
        if cmd[:3] == ["openshell", "status", "--gateway"]:
            return _cp(0, stdout="Status: Connected" if self.connected else "Status: not connected")
        return _cp(0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.time, "sleep", lambda _s: None)


def _which(present: set[str]) -> object:
    return lambda name: f"/usr/bin/{name}" if name in present else None


def test_missing_openshell_cli_returns_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.shutil, "which", _which(set()))
    fake = FakeRun()
    monkeypatch.setattr(gateway, "_run", fake)

    result = gateway.register_local_gateway()

    assert not result.ok
    assert "install it" in result.message
    assert fake.calls == []  # no commands run when the CLI is absent


def test_docker_unreachable_returns_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell"}))  # docker not on PATH
    monkeypatch.setattr(gateway, "_run", FakeRun())

    result = gateway.register_local_gateway()

    assert not result.ok
    assert "Docker" in result.message


def test_macos_registers_and_reports_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker"}))
    fake = FakeRun(connected=True)
    monkeypatch.setattr(gateway, "_run", fake)

    result = gateway.register_local_gateway()

    assert result.ok
    assert ["launchctl", "setenv", "OPENSHELL_DRIVERS", "docker"] in fake.calls
    assert ["launchctl", "setenv", "DOCKER_HOST", "unix:///var/run/docker.sock"] in fake.calls
    assert ["openshell", "gateway", "add", gateway.GATEWAY_ENDPOINT, "--local", "--name", gateway.GATEWAY_NAME] in (
        fake.calls
    )
    assert not any(cmd[:1] == ["systemctl"] or cmd[:1] == ["sudo"] for cmd in fake.calls)


def test_not_connected_returns_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker"}))
    monkeypatch.setattr(gateway, "_run", FakeRun(connected=False))

    result = gateway.register_local_gateway()

    assert not result.ok
    assert "not connected" in result.message


def test_setup_runs_gateway_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """`agent-hardener setup` provisions garak then registers the gateway; a bad gateway is not fatal."""
    monkeypatch.setattr("agent_hardener.garak_venv.provision_garak_venv", lambda **_k: "/tmp/garak/bin/python")
    calls: list[str] = []

    def fake_register(*, wait: bool = True) -> gateway.GatewayResult:
        calls.append("register")
        return gateway.GatewayResult(ok=False, message="Gateway not connected yet — check: ...")

    monkeypatch.setattr("agent_hardener.cli.setup.register_local_gateway", fake_register)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0  # gateway failure warns, does not fail setup
    assert calls == ["register"]
