# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DefendersManager._merge_and_write_policy``: regression test for the reported incident.

Nine attacks were routed; four defenders independently produced ``ok=True`` candidates against
the same pristine policy snapshot, three of them the same fix (removing the beeceptor rule), one a
different fix (removing the backend rule). Because each candidate was written individually,
whichever defender's result was processed last was the only one that survived on disk — the other
three were silently discarded even though every one reported success. This test drives the exact
same shape of input through ``_merge_and_write_policy`` and asserts every fix survives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.defenders.defenders_manager import DefendersManager
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import dump_policy_yaml, load_policy_from_yaml
from agent_hardener.agents.defenders.openshell_defender_v2.policy.schema import Policy
from agent_hardener.models.contracts import CURRENT_POLICY_KEY, DefenderOutput

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_BASE_POLICY = Policy.model_validate(
    {
        "version": 1,
        "network_policies": {
            "beeceptor": {
                "name": "beeceptor",
                "endpoints": [
                    {"host": "abc123.free.beeceptor.com", "port": 443, "protocol": "http", "access": "full"},
                ],
            },
            "integration_backend": {
                "name": "integration_backend",
                "endpoints": [
                    {"host": "host.docker.internal", "port": 8089, "protocol": "http", "access": "full"},
                ],
            },
        },
    }
)


def _without_entry(entry_key: str) -> str:
    candidate = _BASE_POLICY.model_copy(deep=True)
    del candidate.network_policies[entry_key]
    return dump_policy_yaml(candidate)


def test_all_four_incident_defenders_survive_the_merge(tmp_path: Path) -> None:
    victim_policy_path = tmp_path / "is-integration.yaml"
    victim_policy_path.write_text(dump_policy_yaml(_BASE_POLICY), encoding="utf-8")

    enriched_context = {CURRENT_POLICY_KEY: dump_policy_yaml(_BASE_POLICY)}

    # config-test, db-verify, reset/reset-confirm: three independent analyses, same fix.
    beeceptor_removed = _without_entry("beeceptor")
    # backup pipeline: routed/processed last in the incident, a different fix.
    backend_removed = _without_entry("integration_backend")

    results = [
        DefenderOutput(ok=True, new_policy_yaml=beeceptor_removed, resource_type="network"),
        DefenderOutput(ok=True, new_policy_yaml=beeceptor_removed, resource_type="network"),
        DefenderOutput(ok=True, new_policy_yaml=beeceptor_removed, resource_type="network"),
        DefenderOutput(ok=True, new_policy_yaml=backend_removed, resource_type="network"),
    ]
    task_meta = [(None, None)] * len(results)

    DefendersManager._merge_and_write_policy(
        task_meta,
        results,
        enriched_context,
        relay_plugins_path=None,
        victim_policy_path=victim_policy_path,
        target_relay_plugins=None,
    )

    merged = load_policy_from_yaml(victim_policy_path.read_text(encoding="utf-8"))
    assert "beeceptor" not in merged.network_policies, "the beeceptor fix must not be discarded"
    assert "integration_backend" not in merged.network_policies, "the backend fix must not be discarded"


def test_no_write_when_no_candidates_produced_a_policy(tmp_path: Path) -> None:
    victim_policy_path = tmp_path / "is-integration.yaml"
    victim_policy_path.write_text("sentinel: unchanged\n", encoding="utf-8")
    results = [DefenderOutput(ok=False, error_message="abstained")]

    DefendersManager._merge_and_write_policy(
        [(None, None)],
        results,
        {},
        relay_plugins_path=None,
        victim_policy_path=victim_policy_path,
        target_relay_plugins=None,
    )

    assert victim_policy_path.read_text(encoding="utf-8") == "sentinel: unchanged\n"
