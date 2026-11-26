# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenShell gateway registration and its wiring into `agent-hardener setup`."""

from __future__ import annotations

import subprocess
import tomllib
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from agent_hardener.cli import app
from agent_hardener.openshell import gateway, support

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRun:
    """Records commands and scripts responses for the module-level `_run` seam."""

    def __init__(
        self,
        *,
        connected: bool = True,
        overrides: dict[tuple[str, ...], subprocess.CompletedProcess[str] | Exception] | None = None,
    ) -> None:
        self.connected = connected
        self.overrides = overrides or {}
        self.calls: list[list[str]] = []

    def __call__(
        self, cmd: list[str], *, stdin: str | None = None, timeout_s: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        override = self.overrides.get(tuple(cmd[:3]))
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return override
        if cmd[:2] == ["docker", "info"]:
            return _cp(0)
        if cmd[:2] == ["openshell", "--version"]:
            return _cp(0, stdout="openshell 0.1.6\n")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return _cp(0, stdout="unix:///var/run/docker.sock\n")
        if cmd[:3] == ["openshell", "status", "--gateway"]:
            return _cp(0, stdout="Status: Connected" if self.connected else "Status: not connected")
        return _cp(0)


@pytest.fixture(autouse=True)
def _instant_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleeping advances a fake clock, so the connect wait times out without real waiting."""
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(support.time, "sleep", sleep)
    monkeypatch.setattr(support.time, "monotonic", lambda: now[0])


@pytest.fixture(autouse=True)
def _gateway_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "gateway.toml"
    monkeypatch.setattr(gateway, "GATEWAY_CONFIG", path)
    monkeypatch.setattr(gateway.write_gateway_config, "__defaults__", (path,))
    return path


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


@pytest.mark.parametrize(
    ("output", "problem"),
    [
        ("openshell 0.1.6\n", None),
        ("openshell 0.2.0\n", None),
        ("openshell 0.1.4\n", "OpenShell 0.1.4 is older than the required v0.1.6"),
        ("dev build\n", "could not read the OpenShell version"),
    ],
)
def test_outdated_openshell_accepts_the_pinned_release_or_newer(
    monkeypatch: pytest.MonkeyPatch, output: str, problem: str | None
) -> None:
    monkeypatch.setattr(gateway, "_run", FakeRun(overrides={("openshell", "--version"): _cp(0, stdout=output)}))

    found = gateway.outdated_openshell()

    assert found == problem if problem is None else problem in (found or "")


def test_setup_refuses_an_outdated_openshell_with_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker"}))
    fake = FakeRun(overrides={("openshell", "--version"): _cp(0, stdout="openshell 0.1.4\n")})
    monkeypatch.setattr(gateway, "_run", fake)

    result = gateway.register_local_gateway()

    assert not result.ok
    assert gateway.INSTALL_HINT in result.message
    assert ["docker", "info"] not in fake.calls


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
    assert ["brew", "services", "restart", gateway.BREW_SERVICE] in fake.calls
    assert not any(cmd[:1] == ["launchctl"] for cmd in fake.calls)  # driver settings live in gateway.toml
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


def test_linux_network_create_failure_fails_with_docker_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "linux")
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker", "systemctl"}))
    fake = FakeRun(
        overrides={
            ("docker", "network", "inspect"): _cp(1, stderr="No such network"),
            ("docker", "network", "create"): _cp(1, stderr="permission denied on docker.sock"),
        }
    )
    monkeypatch.setattr(gateway, "_run", fake)

    result = gateway.register_local_gateway()

    assert not result.ok
    assert "permission denied on docker.sock" in result.message
    assert not any(cmd[:3] == ["openshell", "gateway", "add"] for cmd in fake.calls)  # stops before registering


def test_restart_timeout_is_named_even_when_the_gateway_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker"}))
    hung = subprocess.TimeoutExpired(["brew"], gateway.SETUP_COMMAND_TIMEOUT_S)
    monkeypatch.setattr(gateway, "_run", FakeRun(overrides={("brew", "services", "restart"): hung}))

    result = gateway.register_local_gateway()

    # Connected, but possibly on the old config: the stale restart must not be silent.
    assert result.ok
    assert f"restarting {gateway.BREW_SERVICE} timed out after 30s" in result.message


def test_gateway_add_failure_explains_a_gateway_that_never_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    monkeypatch.setattr(gateway.shutil, "which", _which({"openshell", "docker"}))
    fake = FakeRun(connected=False, overrides={("openshell", "gateway", "add"): _cp(2, stderr="invalid endpoint")})
    monkeypatch.setattr(gateway, "_run", fake)

    result = gateway.register_local_gateway()

    assert not result.ok
    assert "registering gateway 'auto-defender' failed (exit 2): invalid endpoint" in result.message
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


def test_write_gateway_config_sets_docker_driver_and_keeps_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "gateway.toml"
    path.write_text('[openshell]\nversion = 2\n\n[openshell.gateway]\nlog_level = "debug"\n', encoding="utf-8")

    gateway.write_gateway_config("unix:///Users/me/.docker/run/docker.sock", path)

    config = tomllib.loads(path.read_text(encoding="utf-8"))["openshell"]
    assert config["version"] == 2
    assert config["gateway"] == {"log_level": "debug", "compute_driver": "docker"}
    assert config["drivers"]["docker"]["socket_path"] == "/Users/me/.docker/run/docker.sock"


def test_write_gateway_config_points_docker_desktop_supervisors_at_the_host_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    path = tmp_path / "gateway.toml"

    gateway.write_gateway_config("unix:///var/run/docker.sock", path)

    docker = tomllib.loads(path.read_text(encoding="utf-8"))["openshell"]["drivers"]["docker"]
    assert docker["grpc_endpoint"] == "https://host.docker.internal:17670"


@pytest.mark.parametrize(
    ("getent", "alias"),
    [
        ("192.168.5.2     host.docker.internal\n192.168.5.2     host.lima.internal\n", "host.docker.internal"),
        ("192.168.5.2     host.lima.internal\n", "host.lima.internal"),
    ],
    ids=["both-resolve", "only-lima"],
)
def test_colima_supervisors_use_the_first_host_alias_its_vm_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, getent: str, alias: str
) -> None:
    """Older Colima releases resolve only host.lima.internal inside the VM."""
    monkeypatch.setattr(gateway.sys, "platform", "darwin")
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=getent, stderr="")

    monkeypatch.setattr(gateway, "_run", run)
    path = tmp_path / "gateway.toml"

    gateway.write_gateway_config("unix:///Users/me/.colima/work/docker.sock", path)

    docker = tomllib.loads(path.read_text(encoding="utf-8"))["openshell"]["drivers"]["docker"]
    assert docker["grpc_endpoint"] == f"https://{alias}:17670"
    assert calls[0][:4] == ["colima", "ssh", "--profile", "work"]


def test_local_compute_driver_reads_what_setup_wrote(tmp_path: Path) -> None:
    """The preflight decides whether the local gateway needs this host's Docker from this value."""
    path = tmp_path / "gateway.toml"
    assert gateway.local_compute_driver(path) is None

    gateway.write_gateway_config("unix:///var/run/docker.sock", path)

    assert gateway.local_compute_driver(path) == gateway.DRIVER
