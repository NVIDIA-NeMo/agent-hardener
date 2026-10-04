# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for egress discovery (sources, discoverer, patterns) and policy injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from agent_hardener.openshell.egress import (
    CodeEgressSource,
    DiscoveredEndpoint,
    EgressDiscoverer,
    ManualEgressSource,
    OpenShellLogEgressSource,
    PolicyEndpoint,
    host_pattern,
    parse_urls,
)
from agent_hardener.openshell.egress.discoverer import resolve_private_ips
from agent_hardener.openshell.policy_egress import build_discovered_egress_block, inject_discovered_egress

# Default test resolver: no DNS, no SSRF exemptions (keeps unit tests offline/deterministic).
_NO_RESOLVE = lambda _host: []  # noqa: E731

pytestmark = pytest.mark.unit


# --- URL parsing --------------------------------------------------------------------------


def test_parse_urls_scheme_ports() -> None:
    endpoints = parse_urls("see https://api.nvidia.com/v1 and http://example.org/x and https://h.io:8443/y")
    assert DiscoveredEndpoint("api.nvidia.com", 443) in endpoints
    assert DiscoveredEndpoint("example.org", 80) in endpoints
    assert DiscoveredEndpoint("h.io", 8443) in endpoints


def test_parse_urls_skips_loopback_and_ips() -> None:
    endpoints = parse_urls("http://localhost:8086 http://127.0.0.1 https://host.docker.internal:8086 https://10.0.0.5")
    assert endpoints == set()


# --- host_pattern -------------------------------------------------------------------------


def test_host_pattern_rules() -> None:
    assert host_pattern("integrate.api.nvidia.com") == "*.api.nvidia.com"  # 3+ labels -> parent wildcard
    assert host_pattern("raw.githubusercontent.com") == "*.githubusercontent.com"
    assert host_pattern("tavily.com") == "tavily.com"  # 2 labels -> exact (never a TLD wildcard)


def test_host_pattern_never_tld_wildcard() -> None:
    # No mapping should ever produce a bare `*` or a single-label TLD wildcard like `*.com`.
    for host in ("a.b.com", "x.y.z.co.uk", "foo.io", "bar.com"):
        pattern = host_pattern(host)
        assert pattern != "*"
        assert not (pattern.startswith("*.") and pattern.count(".") == 1)


# --- sources ------------------------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "pkg").mkdir(parents=True)
    (project / "workflow.yaml").write_text(
        "llms:\n  nim:\n    base_url: https://integrate.api.nvidia.com/v1/\n", encoding="utf-8"
    )
    (project / "pkg" / "tool.py").write_text(
        'BASE = "https://api.tavily.com/search"\nWEATHER = "http://wttr.in/London"\n', encoding="utf-8"
    )
    (project / ".venv").mkdir()
    (project / ".venv" / "junk.py").write_text('X = "https://should-be-ignored.example.com"\n', encoding="utf-8")
    return project


def test_source_scans_config_files_not_just_python(tmp_path: Path) -> None:
    """A Relay-connected agent keeps endpoints in whatever format its framework uses.

    Under NAT there was one known workflow YAML to read ``base_url`` from. There is no such file
    now, so config formats are scanned generically — missing a host is not cosmetic, because the
    sandbox is default-deny and the agent's call is dropped mid-run.
    """
    found = CodeEgressSource(_project(tmp_path)).discover()
    assert DiscoveredEndpoint("integrate.api.nvidia.com", 443) in found  # from workflow.yaml


def test_code_source_scans_py_and_skips_ignored_dirs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    found = {ep.host for ep in CodeEgressSource(project).discover()}
    assert "api.tavily.com" in found
    assert "wttr.in" in found
    assert "should-be-ignored.example.com" not in found  # under .venv -> skipped


def test_code_source_respects_max_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr("agent_hardener.openshell.egress.sources._MAX_CODE_FILES", 0)
    assert CodeEgressSource(project).discover() == set()  # cap reached before scanning


def test_code_source_skips_unreadable_paths(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "isadir.py").mkdir()  # matches *.py but read_text raises IsADirectoryError (OSError)
    assert CodeEgressSource(project).discover() == set()  # skipped, no crash


def test_manual_source_urls_and_bare_hosts() -> None:
    found = ManualEgressSource(["https://x.example.com", "api.foo.com", "svc.bar.com:9000", "  "]).discover()
    assert DiscoveredEndpoint("x.example.com", 443) in found
    assert DiscoveredEndpoint("api.foo.com", 443) in found
    assert DiscoveredEndpoint("svc.bar.com", 9000) in found


def test_log_source_stub_is_empty() -> None:
    assert OpenShellLogEgressSource(sandbox="s", gateway="g").discover() == set()


# --- discoverer ---------------------------------------------------------------------------


def test_discoverer_aggregates_and_maps_patterns(tmp_path: Path) -> None:
    project = _project(tmp_path)
    discoverer = EgressDiscoverer(
        [
            CodeEgressSource(project),
            ManualEgressSource(["https://extra.acme.com"]),
        ],
        resolver=_NO_RESOLVE,
    )
    eps = {(e.host, e.port) for e in discoverer.policy_endpoints()}
    assert ("*.api.nvidia.com", 443) in eps  # integrate.api.nvidia.com -> parent wildcard
    assert ("*.tavily.com", 443) in eps  # api.tavily.com (3 labels) -> parent wildcard
    assert ("wttr.in", 80) in eps  # http -> port 80, 2-label host stays exact
    assert ("*.acme.com", 443) in eps


def test_discoverer_adds_ssrf_allowed_ips_for_private_hosts(tmp_path: Path) -> None:
    project = _project(tmp_path)
    # Fake resolver: the nvidia host resolves to an internal IP; others don't resolve.
    resolver = lambda host: ["192.0.2.10"] if host == "integrate.api.nvidia.com" else []  # noqa: E731
    discoverer = EgressDiscoverer([CodeEgressSource(project)], resolver=resolver)
    by_host = {e.host: e for e in discoverer.policy_endpoints()}
    assert by_host["*.api.nvidia.com"].allowed_ips == ("192.0.2.10",)  # SSRF exemption auto-added


def test_resolve_private_ips_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # getaddrinfo returns one private + one public IP; only the private one is kept.
    def fake_getaddrinfo(host, _port):
        return [(2, 1, 6, "", ("10.0.0.5", 0)), (2, 1, 6, "", ("8.8.8.8", 0))]

    monkeypatch.setattr("agent_hardener.openshell.egress.discoverer.socket.getaddrinfo", fake_getaddrinfo)
    assert resolve_private_ips("anything") == ["10.0.0.5"]


def test_resolve_private_ips_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no DNS")

    monkeypatch.setattr("agent_hardener.openshell.egress.discoverer.socket.getaddrinfo", boom)
    assert resolve_private_ips("anything") == []


# --- policy injection ---------------------------------------------------------------------


def _policy(tmp_path: Path) -> Path:
    src = tmp_path / "policy.yaml"
    src.write_text(yaml.safe_dump({"version": 1, "network_policies": {"existing": {"name": "x"}}}), encoding="utf-8")
    return src


def test_build_discovered_egress_block_shape() -> None:
    block = build_discovered_egress_block(
        [PolicyEndpoint("*.nvidia.com", 443, ("192.0.2.0/24",)), PolicyEndpoint("wttr.in", 80)],
        ["/app/project/.venv/bin/**"],
    )
    assert block["name"] == "relay-victim-discovered-egress"
    assert block["endpoints"][0] == {
        "host": "*.nvidia.com",
        "port": 443,
        "protocol": "rest",
        "enforcement": "enforce",
        "access": "full",
        "allowed_ips": ["192.0.2.0/24"],  # SSRF exemption carried through
    }
    assert "allowed_ips" not in block["endpoints"][1]  # omitted when none
    assert block["binaries"] == [{"path": "/app/project/.venv/bin/**"}]


def test_inject_discovered_egress_adds_and_is_idempotent(tmp_path: Path) -> None:
    dest = tmp_path / "out.yaml"
    eps = [PolicyEndpoint("*.nvidia.com", 443, ("192.0.2.10",))]
    inject_discovered_egress(_policy(tmp_path), eps, ["/x/**"], dest)
    inject_discovered_egress(dest, eps, ["/x/**"], dest)
    policy = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert "existing" in policy["network_policies"]
    block = policy["network_policies"]["relay_victim_discovered_egress"]
    assert len(block["endpoints"]) == 1
    assert block["endpoints"][0]["host"] == "*.nvidia.com"
    assert block["endpoints"][0]["allowed_ips"] == ["192.0.2.10"]


def test_inject_discovered_egress_no_endpoints_copies_through(tmp_path: Path) -> None:
    dest = tmp_path / "out.yaml"
    inject_discovered_egress(_policy(tmp_path), [], ["/x/**"], dest)
    policy = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert "relay_victim_discovered_egress" not in policy["network_policies"]
