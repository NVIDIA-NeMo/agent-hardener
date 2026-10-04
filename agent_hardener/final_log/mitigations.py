# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the mitigations artifact for a completed run.

Bundles the before/after OpenShell policy and Relay guardrail set so a downstream consumer (the NeMo plugin saves it
as a job result; Studio renders a git-style diff + recommendations). This reads only run artifacts already on
disk plus the round reports — it does not touch the defenders.

Beyond the whole-file before/after blobs, ``defenses`` enumerates each individually selectable defense — one
per generated ``custom_guardrail_N`` guardrail plus (optionally) the hardened OpenShell policy — paired with
the attack that motivated it. This lets a downstream UI let the user pick a subset to apply and validate.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

from agent_hardener.final_log.helpers import reversed_iterations
from agent_hardener.relay_plugin.config import configured_guardrails
from agent_hardener.swarm_tracker import init_dir, victim_active_state_dir

if TYPE_CHECKING:
    from agent_hardener.models.reports import RoundReport

_CUSTOM_GUARDRAIL_RE = re.compile(r"^custom_guardrail_\d+$")
_PROMPT_EXCERPT_LEN = 400


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except OSError:
        return None


def _hardened_policy(reports: list[RoundReport]) -> str | None:
    """The OpenShell defender's hardened policy, from whichever patch shape it emitted.

    When the session names a ``victim_policy_path`` the defender rewrites that file in place and
    records ``{"candidate_policy_path": ...}``; with no such path it inlines the YAML as
    ``{"new_policy_yaml": ...}``. Reading only the inline shape silently dropped the policy from
    the artifact for every run the product generates, which always sets the path.
    """
    for iteration in reversed_iterations(reports):
        for patch in iteration.policy_patches:
            inline = patch.get("new_policy_yaml")
            if isinstance(inline, str) and inline.strip():
                return inline
            candidate = patch.get("candidate_policy_path")
            if isinstance(candidate, str) and candidate:
                text = _read(Path(candidate))
                if text is not None and text.strip():
                    return text
    return None


def _first_sentence(text: str, limit: int = 160) -> str:
    """A short human summary: the first sentence of *text*, capped at *limit* characters."""
    stripped = " ".join(text.split())
    head = stripped.split(". ", 1)[0]
    return head if len(head) <= limit else head[: limit - 1].rstrip() + "…"


def _guardrail_attacks(reports: list[RoundReport]) -> dict[str, dict[str, str]]:
    """Map each injected ``custom_guardrail_N`` to the attack that motivated it (from the round reports)."""
    linkage: dict[str, dict[str, str]] = {}
    for report in reports:
        for iteration in report.iterations:
            for analysis in iteration.defenders:
                name = analysis.metadata.get("guardrail_name")
                if not isinstance(name, str) or not name:
                    continue
                prompt = analysis.attack_prompt or ""
                linkage[name] = {
                    "prompt_excerpt": prompt[:_PROMPT_EXCERPT_LEN],
                    "attacked_tool": str(analysis.metadata.get("attacked_tool") or ""),
                }
    return linkage


def _policy_attack(reports: list[RoundReport]) -> dict[str, str] | None:
    """The most recent attack the OpenShell (policy) defender responded to, for the policy defense pair."""
    for iteration in reversed_iterations(reports):
        for analysis in iteration.defenders:
            if analysis.metadata.get("resource_type") != "relay_guardrail_component" and analysis.attack_prompt:
                return {"prompt_excerpt": analysis.attack_prompt[:_PROMPT_EXCERPT_LEN], "attacked_tool": ""}
    return None


def _guardrail_defenses(after_workflow: str, reports: list[RoundReport]) -> list[dict[str, Any]]:
    """One selectable defense per ``custom_guardrail_N`` in the hardened workflow, paired with its attack.

    The workflow is the source of truth for which guardrails exist; the round reports supply the attack
    linkage. A guardrail with no matching report entry still surfaces (with an empty ``attack``).
    """
    try:
        config = tomllib.loads(after_workflow)
    except (tomllib.TOMLDecodeError, TypeError):
        return []

    linkage = _guardrail_attacks(reports)
    defenses: list[dict[str, Any]] = []
    for rail in configured_guardrails(config):
        name = str(rail.get("name") or "")
        if not _CUSTOM_GUARDRAIL_RE.match(name):
            continue
        instructions = str(rail.get("system_instructions") or "")
        defenses.append(
            {
                "id": name,
                "kind": "guardrail",
                "target_tool": rail.get("target_tool") or None,
                "summary": _first_sentence(instructions) if instructions else f"Guardrail {name}",
                "config_fragment": tomli_w.dumps({"guardrails": [rail]}).strip(),
                # The defender records which attack motivated the rail, so the report no longer has
                # to re-derive the pairing from round history when it is available.
                "attack": rail.get("attack_id") or linkage.get(name),
            }
        )
    return defenses


def build_mitigations(
    run_dir: Path,
    reports: list[RoundReport],
    *,
    policy_name: str | None,
    guardrails_name: str | None,
) -> dict[str, Any]:
    """Bundle the run's before/after policy + workflow (plus per-defense pairs) for the Studio Mitigations view.

    Baselines are the ``init/`` copies written before round 1; the hardened workflow is the in-place
    ``victim-active-state/`` copy the guardrails defender rewrote; the hardened policy comes from the OpenShell
    defender's policy patch in the round reports. A section is omitted when it has no change.
    ``defenses`` lists each individually selectable guardrail + (optionally) the policy, paired with its attack.
    """
    init = init_dir(run_dir)
    active = victim_active_state_dir(run_dir)
    result: dict[str, Any] = {}
    defenses: list[dict[str, Any]] = []

    if guardrails_name:
        before, after = _read(init / guardrails_name), _read(active / guardrails_name)
        if before is not None and after is not None and before != after:
            result["guardrails"] = {"before": before, "after": after}
            defenses.extend(_guardrail_defenses(after, reports))

    if policy_name:
        before, after = _read(init / policy_name), _hardened_policy(reports)
        if before is not None and after is not None and before != after:
            result["policy"] = {"before": before, "after": after}
            defenses.append(
                {
                    "id": "openshell_policy",
                    "kind": "policy",
                    "target_tool": None,
                    "summary": "OpenShell sandbox policy hardening",
                    "config_fragment": after.strip(),
                    "attack": _policy_attack(reports),
                }
            )

    if defenses:
        result["defenses"] = defenses

    return result
