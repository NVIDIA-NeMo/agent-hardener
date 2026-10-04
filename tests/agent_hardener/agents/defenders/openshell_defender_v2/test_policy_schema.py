# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the policy load/dump path."""

from __future__ import annotations

import pytest

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
      protocol: http
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
