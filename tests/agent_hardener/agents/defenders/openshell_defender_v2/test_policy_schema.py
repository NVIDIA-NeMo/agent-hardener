# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the policy load/dump path."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.errors import PolicyParseError
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import (
    dump_policy_yaml,
    load_policy_from_yaml,
)

pytestmark = pytest.mark.unit

_POLICY_YAML = """\
version: 1
network_policies:
  github:
    name: github
    endpoints:
    - host: api.github.com
      port: 443
      protocol: rest
      rules:
      - allow:
          method: GET
          path: /repos/*/*/issues
    binaries:
    - path: /usr/bin/curl
"""


def test_round_trip_preserves_structure() -> None:
    policy = load_policy_from_yaml(_POLICY_YAML)
    dumped_yaml = dump_policy_yaml(policy)
    dumped = load_policy_from_yaml(dumped_yaml)
    assert dumped.network_policies["github"].endpoints[0].host == "api.github.com"
    assert dumped.network_policies["github"].binaries[0].path == "/usr/bin/curl"
    assert "network_middlewares" not in dumped_yaml


def test_empty_policy_text_yields_default_policy() -> None:
    policy = load_policy_from_yaml("")
    assert policy.version == 1
    assert policy.network_policies == {}


def test_binaries_normalizes_bare_strings() -> None:
    policy = load_policy_from_yaml("version: 1\nnetwork_policies:\n  a:\n    binaries: ['/bin/bash']\n")
    assert policy.network_policies["a"].binaries[0].path == "/bin/bash"


def test_static_sections_pass_through_opaquely() -> None:
    text = "version: 1\nfilesystem_policy:\n  read_only: ['/etc']\nnetwork_policies: {}\n"
    policy = load_policy_from_yaml(text)
    dumped = dump_policy_yaml(policy)
    assert "read_only" in dumped


@pytest.mark.parametrize(
    "endpoint",
    [
        "{host: a.example.com, port: 443, options: {allow_encoded_slash: true}}",  # unknown field
        "{host: a.example.com, port: 443, protocol: http}",  # unsupported protocol name
        "{host: a.example.com, port: 443, tls: terminate}",  # removed tls mode
        "{host: a.example.com, port: 443, ports: [443]}",
        "{host: a.example.com, port: 443, protocol: rest, access: full, rules: [{allow: {method: GET, path: /}}]}",
        "{host: a.example.com, port: 443, deny_rules: [{method: GET, path: /}]}",
        "{host: a.example.com, port: 443, protocol: tcp, access: full}",
        "{host: a.example.com, port: 443, protocol: rest, tls: skip, access: full}",
    ],
)
def test_rejects_what_openshell_rejects(endpoint: str) -> None:
    with pytest.raises(PolicyParseError):
        load_policy_from_yaml(f"version: 1\nnetwork_policies:\n  a:\n    endpoints: [{endpoint}]\n")


def test_rejects_unknown_entry_field() -> None:
    with pytest.raises(PolicyParseError):
        load_policy_from_yaml("version: 1\nnetwork_policies:\n  a:\n    description: x\n    endpoints: []\n")


def test_rejects_host_port_disagreeing_on_allowed_ips() -> None:
    text = (
        "version: 1\nnetwork_policies:\n"
        "  a:\n    endpoints: [{host: h.example.com, port: 80, allowed_ips: [10.0.0.1/32]}]\n"
        "  b:\n    endpoints: [{host: h.example.com, port: 80}]\n"
    )
    with pytest.raises(PolicyParseError, match="allowed_ips"):
        load_policy_from_yaml(text)


def test_ports_list_round_trips_and_matches_each_port() -> None:
    policy = load_policy_from_yaml(
        "version: 1\nnetwork_policies:\n  a:\n    endpoints: [{host: h.example.com, ports: [80, 8080]}]\n"
    )
    endpoint = policy.network_policies["a"].endpoints[0]
    assert endpoint.matches_port(8080)
    assert not endpoint.matches_port(443)
    assert "8080" in dump_policy_yaml(policy)
