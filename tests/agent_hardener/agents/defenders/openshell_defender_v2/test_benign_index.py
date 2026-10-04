# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BenignIndex`` endpoint-presence tracking: a bare host/port tuple must still register.

Regression coverage for the bug where a ``RequestTuple`` carrying only ``host``/``port`` (no
method/path/binary/semantic field) left no trace in any of ``BenignIndex``'s field-specific dicts.
``analysis.feasibility.check_remove_endpoint`` and ``analysis.overlap.compute_overlap`` both
derived "was this endpoint ever touched by benign traffic" purely from those dicts, so a bare
tuple was silently invisible to both — the endpoint looked "entirely absent from predicted benign
traffic" and ``remove_endpoint`` fired anyway.

Note: ``relay_victim`` environment endpoints (agent_env/backends/LLM endpoint) no longer flow
through ``BenignIndex`` at all — they're an entirely separate, absolute block-list on
``DefenderContext.protected_endpoints`` (see ``extraction.environment``/``test_environment.py``).
This file only exercises ``BenignIndex``'s own generic bare-tuple behavior, using
``merge_counterexamples`` as a stand-in absorber.
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.analysis.benign_index import (
    build_index,
    merge_counterexamples,
)
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.overlap import compute_overlap
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.selector import select
from agent_hardener.agents.defenders.openshell_defender_v2.models import DefenderContext, RequestTuple
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml
from agent_hardener.models.contracts import DefenderInput

pytestmark = pytest.mark.unit

_BARE_TUPLE = RequestTuple(host="host.docker.internal", port=8089, protocol="rest")

_POLICY = """\
version: 1
network_policies:
  integration_backend:
    name: integration_backend
    endpoints:
    - host: host.docker.internal
      port: 8089
      protocol: rest
      access: full
"""


def test_bare_host_port_tuple_registers_as_a_known_endpoint() -> None:
    index = build_index([])
    merge_counterexamples(index, [_BARE_TUPLE])

    assert (_BARE_TUPLE.host, _BARE_TUPLE.port) in index.endpoints
    # It must NOT fabricate axis data the tuple never carried.
    assert index.methods[(_BARE_TUPLE.host, _BARE_TUPLE.port)] == set()
    assert index.binaries[(_BARE_TUPLE.host, _BARE_TUPLE.port)] == set()


def test_bare_tuple_registers_a_host_and_port_collision() -> None:
    index = build_index([])
    merge_counterexamples(index, [_BARE_TUPLE])

    overlap = compute_overlap(_BARE_TUPLE, index)

    assert overlap.host_collision is True
    assert overlap.port_collision is True


def test_bare_tuple_only_endpoint_is_never_chosen_for_removal() -> None:
    """The exact regression: an endpoint known only via a bare tuple (no method/path/binary/

    semantic evidence) must not look "untouched" to ``remove_endpoint``'s feasibility check.
    """
    policy = load_policy_from_yaml(_POLICY)
    index = build_index([])
    merge_counterexamples(index, [_BARE_TUPLE])
    cut = RequestTuple(host="host.docker.internal", port=8089, protocol="rest")
    ctx = DefenderContext(
        defender_input=DefenderInput(attack_prompt="", agent_response="", attacked_tool="t"),
        policy=policy,
        cut=cut,
        benign_index=index,
        overlap=compute_overlap(cut, index),
    )

    selection = select(ctx)

    assert selection.chosen != "remove_endpoint"
    assert "remove_endpoint" in selection.infeasible
