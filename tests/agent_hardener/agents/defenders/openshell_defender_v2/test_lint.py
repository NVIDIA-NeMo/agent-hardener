# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``assert_contraction`` must raise on any widening delta and pass through a strict narrowing."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.errors import LintViolationError
from agent_hardener.agents.defenders.openshell_defender_v2.policy.lint import assert_contraction
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml

pytestmark = pytest.mark.unit

_BASE = """\
version: 1
network_policies:
  github:
    name: github
    endpoints:
    - host: api.github.com
      port: 443
      protocol: http
      rules:
      - allow: {method: GET, path: /repos/*/*/issues}
"""


def test_narrowing_with_added_deny_rule_passes() -> None:
    before = load_policy_from_yaml(_BASE)
    after_text = _BASE + "      deny_rules:\n      - {method: DELETE, path: /repos/*/*/issues/*}\n"
    after = load_policy_from_yaml(after_text)
    assert_contraction(before, after)  # should not raise


def test_new_host_raises() -> None:
    before = load_policy_from_yaml(_BASE)
    after_text = _BASE.replace(
        "version: 1\nnetwork_policies:\n  github:",
        "version: 1\nnetwork_policies:\n  evil:\n    name: evil\n    endpoints:\n"
        "    - host: evil.example.com\n      port: 443\n      protocol: http\n      access: full\n  github:",
    )
    after = load_policy_from_yaml(after_text)
    with pytest.raises(LintViolationError, match="new host"):
        assert_contraction(before, after)


def test_expanded_method_raises() -> None:
    before = load_policy_from_yaml(_BASE)
    after_text = _BASE.replace(
        "rules:\n      - allow: {method: GET, path: /repos/*/*/issues}",
        "rules:\n      - allow: {method: GET, path: /repos/*/*/issues}\n      - allow: {method: DELETE, path: /repos/*/*/issues}",
    )
    after = load_policy_from_yaml(after_text)
    with pytest.raises(LintViolationError):
        assert_contraction(before, after)


def test_widened_glob_raises() -> None:
    before = load_policy_from_yaml(_BASE)
    after_text = _BASE.replace("path: /repos/*/*/issues", "path: /**")
    after = load_policy_from_yaml(after_text)
    with pytest.raises(LintViolationError):
        assert_contraction(before, after)


def test_new_binary_raises() -> None:
    before_text = _BASE + "    binaries:\n    - path: /usr/bin/curl\n"
    before = load_policy_from_yaml(before_text)
    after_text = before_text.replace(
        "binaries:\n    - path: /usr/bin/curl", "binaries:\n    - path: /usr/bin/curl\n    - path: /usr/bin/wget"
    )
    after = load_policy_from_yaml(after_text)
    with pytest.raises(LintViolationError, match="new binary"):
        assert_contraction(before, after)


def test_removed_endpoint_is_a_valid_contraction() -> None:
    before = load_policy_from_yaml(_BASE)
    after = load_policy_from_yaml("version: 1\nnetwork_policies: {}\n")
    assert_contraction(before, after)  # should not raise
