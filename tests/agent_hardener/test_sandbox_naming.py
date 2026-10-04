# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mapping a public agent name onto a length-bounded OpenShell sandbox name."""

from __future__ import annotations

import re

import pytest

from agent_hardener.openshell.naming import MAX_SANDBOX_NAME_LEN, sandbox_name

_LEGAL = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")


@pytest.mark.parametrize(
    "agent_name",
    [
        "a",
        "finance",
        "react-agent",
        # The name from the QA repro, which produced a 47-character sandbox before the fix.
        "qa-ah-1fae495c9f-manifest-1faaef",
        "Finance Assistant (v2)",
        "___",
        "x" * 500,
    ],
)
def test_sandbox_name_is_always_legal_and_bounded(agent_name: str) -> None:
    name = sandbox_name(agent_name)
    assert len(name) <= MAX_SANDBOX_NAME_LEN
    assert _LEGAL.match(name), name


def test_sandbox_name_is_deterministic() -> None:
    # Reuse and teardown both address the sandbox by name, so a rerun must derive the same one.
    assert sandbox_name("finance") == sandbox_name("finance")


def test_long_names_sharing_a_prefix_stay_isolated() -> None:
    first = sandbox_name("qa-ah-1fae495c9f-manifest-aaaaaa")
    second = sandbox_name("qa-ah-1fae495c9f-manifest-bbbbbb")
    assert first != second


def test_readable_head_of_the_agent_name_survives() -> None:
    assert sandbox_name("finance").startswith("ah-finance-")
