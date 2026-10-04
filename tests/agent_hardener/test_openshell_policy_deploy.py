# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The path from "the defender generated a policy" to "the sandbox received it".

Every assertion here corresponds to a defect that shipped: the branch that built the patch was
unreachable, and the patch that reached the deploy stage could claim a live apply was safe when it
was not. Both were silent — a generated policy was written to disk and discarded, and no test
noticed for two campaigns.
"""

from __future__ import annotations

from pathlib import Path

from agent_hardener.agents.defenders.defenders_manager import _build_policy_patches
from agent_hardener.models.contracts import DefenderOutput
from agent_hardener.runtime.adapters import openshell_policy_patch


def _result(resource_type: str, yaml_text: str = "network_policies: {}\n") -> DefenderOutput:
    return DefenderOutput(ok=True, new_policy_yaml=yaml_text, resource_type=resource_type)


def _candidate(path: str = "/tmp/policy.yaml", *, requires_recreate: bool) -> dict[str, object]:
    return {
        "type": "openshell_policy_candidate",
        "candidate_policy_path": path,
        "changed": True,
        "requires_recreate": requires_recreate,
    }


def test_a_generated_policy_produces_a_deployable_patch(tmp_path: Path) -> None:
    """The regression: this branch was gated on resource_type == "openshell_policy".

    The defender reports the resource it *attacked* — network, filesystem, landlock, process — so
    that condition was never true and no policy ever reached a sandbox.

    ``_build_policy_patches`` itself no longer writes to ``candidate_policy_path`` — concurrent
    defenders each compute a candidate against the same pristine policy, so writing each one here
    would clobber the others; ``DefendersManager._merge_and_write_policy`` merges every candidate
    and writes the result exactly once (see ``test_defenders_manager_policy_merge.py``).
    """
    policy_path = tmp_path / "policy.yaml"

    patches = _build_policy_patches(_result("network"), None, None, policy_path)

    assert len(patches) == 1
    assert patches[0]["type"] == "openshell_policy_candidate"
    assert patches[0]["candidate_policy_path"] == str(policy_path)


def test_every_resource_type_the_defender_emits_is_deployable(tmp_path: Path) -> None:
    """Deployable means the deploy stage can select it, not merely that a list came back.

    Asserting the list is non-empty is not enough: the old fallback returned a bare
    ``{"new_policy_yaml": ...}`` dict, which is truthy and which ``openshell_policy_patch`` can
    never match. That assertion would have passed for the whole time no policy was deploying.
    """
    for resource_type in ("network", "filesystem", "landlock", "process"):
        patches = _build_policy_patches(_result(resource_type), None, None, tmp_path / "p.yaml")
        assert openshell_policy_patch(patches) is not None, f"{resource_type} produced nothing deployable"


def test_only_network_changes_may_hot_reload(tmp_path: Path) -> None:
    """Filesystem, landlock and process edits cannot be applied to a running sandbox."""
    policy_path = tmp_path / "policy.yaml"

    assert _build_policy_patches(_result("network"), None, None, policy_path)[0]["requires_recreate"] is False
    for resource_type in ("filesystem", "landlock", "process"):
        patch = _build_policy_patches(_result(resource_type), None, None, policy_path)[0]
        assert patch["requires_recreate"] is True, f"{resource_type} must force a recreate"


def test_no_policy_yaml_produces_no_patch(tmp_path: Path) -> None:
    assert _build_policy_patches(DefenderOutput(ok=True, resource_type="network"), None, None, tmp_path) == []


def test_a_recreate_from_any_patch_wins(tmp_path: Path) -> None:
    """The defender emits one patch per attack, and they all name the same candidate file.

    That file holds the union of their edits, so taking the first patch's flag deployed an
    all-attacks payload on a one-attack decision — and which patch came first was arbitrary.
    """
    patch = openshell_policy_patch([_candidate(requires_recreate=False), _candidate(requires_recreate=True)])

    assert patch is not None
    assert patch["requires_recreate"] is True


def test_the_order_of_the_patches_does_not_change_the_decision() -> None:
    forward = openshell_policy_patch([_candidate(requires_recreate=False), _candidate(requires_recreate=True)])
    reversed_ = openshell_policy_patch([_candidate(requires_recreate=True), _candidate(requires_recreate=False)])

    assert forward["requires_recreate"] == reversed_["requires_recreate"] is True


def test_all_network_patches_still_hot_reload() -> None:
    patch = openshell_policy_patch([_candidate(requires_recreate=False), _candidate(requires_recreate=False)])

    assert patch["requires_recreate"] is False


def test_merging_does_not_mutate_the_defender_s_own_patch() -> None:
    """Each defender's artifact must keep recording what it individually asked for."""
    original = _candidate(requires_recreate=False)

    openshell_policy_patch([original, _candidate(requires_recreate=True)])

    assert original["requires_recreate"] is False


def test_guardrail_candidate_path_is_absolute(tmp_path: Path) -> None:
    """The deploy stage uploads this path from the victim project's cwd, not this process's.

    A run-dir-relative path resolves against the wrong root there, and the upload fails with "local
    path does not exist" only *after* every guardrail has been generated — so the round reports
    fifteen successful defenders and still leaves the victim unguarded.
    """
    plugins = tmp_path / "victim-active-state" / "plugins.toml"
    plugins.parent.mkdir(parents=True)
    plugins.write_text("version = 1\n", encoding="utf-8")
    result = DefenderOutput(ok=True, new_policy_yaml="version = 1\n", resource_type="relay_guardrail_component")

    patches = _build_policy_patches(
        result,
        relay_plugins_path=plugins,
        target_relay_plugins=Path("/etc/nemo-relay/plugins.toml"),
        victim_policy_path=None,
    )

    (patch,) = patches
    assert Path(patch["candidate_relay_plugins_path"]).is_absolute()
    assert Path(patch["candidate_relay_plugins_path"]).is_file()
