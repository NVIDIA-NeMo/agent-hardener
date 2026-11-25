# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every host tool a run shells out to is checked before the sandbox, the way the run will resolve it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.errors import ConfigError, SandboxProvisioningError
from agent_hardener.manifest import AGENT_BREAKER_IMPLEMENTATION
from agent_hardener.models import SessionConfig
from agent_hardener.preflight import host
from agent_hardener.preflight import run as preflight_run

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ModelProbe


@dataclass
class HostProbes:
    """Scripted answers for the host probes, and what each was asked."""

    on_path: set[str] = field(default_factory=lambda: {"openshell"})
    docker_up: bool = True
    gateway_up: bool = True
    compose_error: Exception | None = None
    driver: str | None = None
    sandbox_ready: bool = False
    recorded_image: str | None = None
    gateways: list[tuple[str, str]] = field(default_factory=list)
    docker_probes: int = 0

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if name in self.on_path else None

    def docker_reachable(self, *, timeout_s: float | None = None) -> bool:
        self.docker_probes += 1
        return self.docker_up

    def gateway_connected(self, gateway: str, *, openshell_bin: str, timeout_s: float | None = None) -> bool:
        self.gateways.append((gateway, openshell_bin))
        return self.gateway_up

    def resolve_compose_command(self) -> list[str]:
        if self.compose_error is not None:
            raise self.compose_error
        return ["docker", "compose"]


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> HostProbes:
    scripted = HostProbes()
    monkeypatch.setattr(host.shutil, "which", scripted.which)
    monkeypatch.setattr(host, "docker_reachable", scripted.docker_reachable)
    monkeypatch.setattr(host, "gateway_connected", scripted.gateway_connected)
    monkeypatch.setattr(host, "resolve_compose_command", scripted.resolve_compose_command)
    monkeypatch.setattr(host, "local_compute_driver", lambda: scripted.driver)
    monkeypatch.setattr(host, "sandbox_is_ready", lambda _config: scripted.sandbox_ready)
    monkeypatch.setattr(host, "recorded_victim_image", lambda _session, _config: scripted.recorded_image)
    return scripted


def _session(
    tmp_path: Path,
    *,
    openshell: dict[str, Any] | None = None,
    backends: list[dict[str, Any]] | None = None,
    agent_breaker: bool = False,
) -> SessionConfig:
    victim_control: dict[str, Any] = {}
    if openshell is not None or backends is not None:
        config: dict[str, Any] = {"policy_path": "policy.yaml", "cwd": str(tmp_path), **(openshell or {})}
        if backends is not None:
            config["relay_victim"] = {
                "project_dir": ".",
                "dockerfile": "Dockerfile",
                "victim_binaries": ["/usr/bin/python3"],
                "backends": backends,
            }
        victim_control = {"type": "openshell", "config": config}
    attackers = (
        [{"name": "garak", "role": "attacker", "implementation": AGENT_BREAKER_IMPLEMENTATION, "config": {}}]
        if agent_breaker
        else []
    )
    return SessionConfig(
        storage={"root_dir": tmp_path},
        target={"name": "victim"},
        victim={"name": "victim", "role": "victim", "implementation": "pkg:run"},
        attackers=attackers,
        victim_control=victim_control,
    )


def test_a_run_without_openshell_or_garak_probes_nothing(tmp_path: Path, probes: HostProbes) -> None:
    probes.on_path = set()

    host.preflight_host(_session(tmp_path))

    assert probes.gateways == []
    assert probes.docker_probes == 0


def test_a_ready_host_passes_and_checks_the_configured_gateway(tmp_path: Path, probes: HostProbes) -> None:
    probes.on_path = {"/opt/openshell"}

    host.preflight_host(_session(tmp_path, openshell={"gateway": "gw-1", "openshell_bin": "/opt/openshell"}))

    assert probes.gateways == [("gw-1", "/opt/openshell")]
    assert probes.docker_probes == 1


def test_missing_openshell_cli_fails_before_any_probe(tmp_path: Path, probes: HostProbes) -> None:
    probes.on_path = set()

    with pytest.raises(SandboxProvisioningError, match="OpenShell CLI not found") as caught:
        host.preflight_host(_session(tmp_path, openshell={}))

    assert "agent-hardener setup" in caught.value.remediation
    assert probes.gateways == []


def test_stopped_docker_fails_a_local_build(tmp_path: Path, probes: HostProbes) -> None:
    probes.docker_up = False

    with pytest.raises(SandboxProvisioningError, match="Docker daemon not reachable"):
        host.preflight_host(_session(tmp_path, openshell={}))


def test_a_configured_image_on_a_remote_gateway_does_not_need_docker(tmp_path: Path, probes: HostProbes) -> None:
    probes.docker_up = False

    host.preflight_host(_session(tmp_path, openshell={"gateway": "k8s", "image": "registry.example.com/agent:1"}))

    assert probes.docker_probes == 0


def test_the_local_docker_gateway_needs_docker_even_with_a_prebuilt_image(tmp_path: Path, probes: HostProbes) -> None:
    """Its compute driver runs every sandbox as a container on this host."""
    probes.docker_up = False
    probes.driver = "docker"

    with pytest.raises(SandboxProvisioningError, match="Docker daemon not reachable"):
        host.preflight_host(_session(tmp_path, openshell={"image": "registry.example.com/agent:1"}))


def test_reusing_a_ready_sandbox_on_its_recorded_image_does_not_need_docker(tmp_path: Path, probes: HostProbes) -> None:
    probes.docker_up = False
    probes.sandbox_ready = True
    probes.recorded_image = "registry.example.com/agent:1"

    host.preflight_host(_session(tmp_path, openshell={"gateway": "k8s"}), reuse=True)

    assert probes.docker_probes == 0


@pytest.mark.parametrize(
    ("reuse", "ready", "recorded"),
    [(False, True, "registry.example.com/agent:1"), (True, False, "registry.example.com/agent:1"), (True, True, None)],
    ids=["no-reuse", "sandbox-not-ready", "agent-changed"],
)
def test_a_recorded_image_the_run_will_not_adopt_still_needs_docker(
    tmp_path: Path, probes: HostProbes, reuse: bool, ready: bool, recorded: str | None
) -> None:
    """Without all three the run builds the victim locally, so a host without Docker must fail up front."""
    probes.docker_up = False
    probes.sandbox_ready = ready
    probes.recorded_image = recorded

    with pytest.raises(SandboxProvisioningError, match="Docker daemon not reachable"):
        host.preflight_host(_session(tmp_path, openshell={"gateway": "k8s"}), reuse=reuse)


def test_disconnected_gateway_fails_with_its_name(tmp_path: Path, probes: HostProbes) -> None:
    probes.gateway_up = False

    with pytest.raises(SandboxProvisioningError, match="'gw-1' is not connected") as caught:
        host.preflight_host(_session(tmp_path, openshell={"gateway": "gw-1"}))

    assert "agent-hardener setup" in caught.value.remediation


def test_compose_backends_need_docker_even_with_a_prebuilt_image(tmp_path: Path, probes: HostProbes) -> None:
    probes.docker_up = False
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")

    with pytest.raises(SandboxProvisioningError, match="Docker daemon"):
        host.preflight_host(
            _session(
                tmp_path,
                openshell={"image": "registry.example.com/agent:1"},
                backends=[{"name": "api", "compose_file": "compose.yaml"}],
            )
        )


def test_missing_compose_file_names_the_backend_and_path(tmp_path: Path, probes: HostProbes) -> None:
    with pytest.raises(ConfigError, match="backend 'api' not found") as caught:
        host.preflight_host(_session(tmp_path, backends=[{"name": "api", "compose_file": "absent.yaml"}]))

    assert str(tmp_path / "absent.yaml") in str(caught.value)


def test_unresolvable_compose_command_fails(tmp_path: Path, probes: HostProbes) -> None:
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    probes.compose_error = SandboxProvisioningError("Docker Compose not found")

    with pytest.raises(SandboxProvisioningError, match="Docker Compose not found"):
        host.preflight_host(_session(tmp_path, backends=[{"name": "api", "compose_file": "compose.yaml"}]))


def test_backends_without_a_compose_file_need_no_compose(tmp_path: Path, probes: HostProbes) -> None:
    probes.compose_error = SandboxProvisioningError("Docker Compose not found")

    host.preflight_host(_session(tmp_path, backends=[{"name": "external-api"}]))


def test_unprovisioned_garak_venv_fails(tmp_path: Path, probes: HostProbes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GARAK_COMMAND", raising=False)
    monkeypatch.setenv("AGENT_HARDENER_GARAK_PYTHON", str(tmp_path / "absent" / "python"))

    with pytest.raises(ConfigError, match="garak interpreter not found") as caught:
        host.preflight_host(_session(tmp_path, agent_breaker=True))

    assert "agent-hardener setup" in caught.value.remediation


def test_garak_command_override_must_resolve(
    tmp_path: Path, probes: HostProbes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GARAK_COMMAND", "no-such-garak --flag")

    with pytest.raises(ConfigError, match="'no-such-garak'"):
        host.preflight_host(_session(tmp_path, agent_breaker=True))

    probes.on_path.add("no-such-garak")
    host.preflight_host(_session(tmp_path, agent_breaker=True))


def test_preflight_run_stops_at_the_host_before_probing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_probes: list[ModelProbe]
) -> None:
    def broken_host(_session: SessionConfig, *, reuse: bool) -> None:
        raise SandboxProvisioningError("docker down")

    monkeypatch.setattr(preflight_run, "preflight_host", broken_host)

    with pytest.raises(SandboxProvisioningError):
        preflight_run.preflight_run(_session(tmp_path))
    assert model_probes == []

    monkeypatch.setattr(preflight_run, "preflight_host", lambda _session, **_options: None)
    preflight_run.preflight_run(_session(tmp_path))
    assert model_probes
