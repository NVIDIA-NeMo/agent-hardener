# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared plain-text builders for defender policy / guardrail detail.

Used by both the final-report renderer (:mod:`render_plain`) and the live-progress defender renderer
(:mod:`agent_hardener.display.renderers.openshell_policy`) so the two paths format policy detail
identically. Everything here is pure text (``list[str]``); callers wrap it for Rich as needed. This
module holds the low-level builders both paths share — it does not import ``render_plain``, so the
live renderer no longer reaches back into the final-report module.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from agent_hardener.final_log import (
    MAX_POLICY_ITEMS,
    fmt_bool,
    fmt_value,
    load_json,
    short_text,
)
from agent_hardener.relay_plugin.config import configured_guardrails


def _overflow_lines(total: int, noun: str) -> list[str]:
    """The ``... N more <noun>`` line when a list was truncated to ``MAX_POLICY_ITEMS`` (else nothing)."""
    if total <= MAX_POLICY_ITEMS:
        return []
    return [f"        ... {total - MAX_POLICY_ITEMS} more {noun}"]


def render_policy_detail(
    defenders: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    victim_controls: list[dict[str, Any]],
) -> list[str]:
    """Render defender reasoning: candidate summary, statuses, per-patch detail, victim-control.

    Takes already-flattened defender/patch/victim-control lists so both the final report and the
    live defender renderer feed it directly, with no fabricated report shape.
    """
    lines = ["Policy and guardrail results:"]
    openshell_patches = [patch for patch in patches if patch.get("type") == "openshell_policy_candidate"]
    workflow_patches = [patch for patch in patches if patch.get("type") == "victim_workflow_candidate"]
    changed_openshell = [patch for patch in openshell_patches if patch.get("changed") is not False]
    lines.append(f"  OpenShell policy candidate created: {fmt_bool(bool(openshell_patches))}")
    lines.append(
        f"  OpenShell policy candidates changed/applied candidates: {len(changed_openshell)}/{len(openshell_patches)}"
    )
    lines.append(f"  guardrails workflow candidate created: {fmt_bool(bool(workflow_patches))}")
    lines.extend(_render_defender_statuses(defenders))
    if not patches:
        lines.append("  no defender policy or guardrail patches were produced")
    else:
        for patch in patches:
            if isinstance(patch, dict):
                lines.extend(_render_patch(patch))
    for victim_control in victim_controls:
        lines.extend(_render_victim_control(victim_control))
    return lines


def _render_defender_statuses(defenders: list[dict[str, Any]]) -> list[str]:
    if not defenders:
        return []
    succeeded = sum(1 for defender in defenders if defender.get("ok", True))
    lines = [f"  defender agents: {succeeded}/{len(defenders)} succeeded"]
    for defender in defenders:
        patch_count = len(defender.get("policy_patches", [])) if isinstance(defender.get("policy_patches"), list) else 0
        line = (
            f"  - defender {defender.get('agent_name', defender.get('agent_id', '<unknown>'))}: "
            f"ok={fmt_bool(defender.get('ok'))}, patches={patch_count}"
        )
        if defender.get("summary"):
            line += f", summary={short_text(str(defender['summary']), 160)}"
        if defender.get("error"):
            line += f", error={short_text(str(defender['error']), 160)}"
        lines.append(line)
    return lines


def _render_patch(patch: dict[str, Any]) -> list[str]:
    patch_type = str(patch.get("type", "<unknown>"))
    lines = [f"  - patch type: {patch_type}"]
    for key in (
        "candidate_policy_path",
        "candidate_workflow_path",
        "target_workflow_path",
        "requires_recreate",
        "changed",
        "policy_creation_run_log_path",
        "finding_routes_path",
        "aggregated_patch_plan_path",
        "per_finding_modifications_path",
        "diff_path",
    ):
        if key in patch:
            lines.append(f"      {key}: {fmt_value(patch.get(key))}")
    if patch.get("aggregated_patch_plan_path"):
        lines.extend(render_aggregated_plan(Path(str(patch["aggregated_patch_plan_path"]))))
    if patch.get("per_finding_modifications_path"):
        lines.extend(_render_modifications(Path(str(patch["per_finding_modifications_path"]))))
    if patch.get("candidate_policy_path"):
        lines.extend(_render_candidate_policy(Path(str(patch["candidate_policy_path"]))))
    workflow_path = patch.get("target_workflow_path") or patch.get("candidate_workflow_path")
    if workflow_path:
        lines.extend(_render_guardrails_workflow(Path(str(workflow_path))))
    return lines


def render_aggregated_plan(path: Path) -> list[str]:
    """Summarize a generated OpenShell aggregated patch plan (also reused by the round comparison)."""
    if not path.exists():
        return [f"      aggregated plan missing: {path}"]
    plan = load_json(path)
    operations = plan.get("operations") if isinstance(plan.get("operations"), list) else []
    lines = [f"      generated OpenShell operations: {len(operations)}"]
    if plan.get("summary"):
        lines.append(f"      plan summary: {short_text(str(plan['summary']), 220)}")
    for index, operation in enumerate(operations[:MAX_POLICY_ITEMS], start=1):
        if isinstance(operation, dict):
            lines.append(f"        {index}. {_operation_summary(operation)}")
    lines.extend(_overflow_lines(len(operations), "operation(s)"))
    return lines


def _render_modifications(path: Path) -> list[str]:
    if not path.exists():
        return [f"      per-finding modifications missing: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"      per-finding modifications unreadable: {exc}"]
    if not isinstance(data, list):
        return []
    lines = [f"      per-finding LLM policy decisions: {len(data)}"]
    for index, modification in enumerate(data[:MAX_POLICY_ITEMS], start=1):
        if not isinstance(modification, dict):
            continue
        finding = modification.get("finding") if isinstance(modification.get("finding"), dict) else {}
        plan = modification.get("plan") if isinstance(modification.get("plan"), dict) else {}
        operations = modification.get("operations") if isinstance(modification.get("operations"), list) else []
        lines.append(
            "        "
            f"{index}. finding={finding.get('finding_id', '<unknown>')} "
            f"operations={len(operations)} summary={short_text(str(plan.get('summary', '<none>')), 180)}"
        )
    lines.extend(_overflow_lines(len(data), "finding decision(s)"))
    return lines


def _render_candidate_policy(path: Path) -> list[str]:
    if not path.exists():
        return [f"      candidate policy missing: {path}"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"      candidate policy unreadable: {exc}"]
    if not isinstance(data, dict):
        return []
    lines = ["      candidate policy contents:"]
    filesystem = data.get("filesystem_policy") if isinstance(data.get("filesystem_policy"), dict) else {}
    if filesystem:
        read_only = filesystem.get("read_only") if isinstance(filesystem.get("read_only"), list) else []
        read_write = filesystem.get("read_write") if isinstance(filesystem.get("read_write"), list) else []
        lines.append(
            "        filesystem: "
            f"include_workdir={fmt_bool(filesystem.get('include_workdir'))}, "
            f"read_only={len(read_only)}, read_write={len(read_write)}"
        )
        if read_write:
            lines.append(f"        writable paths: {', '.join(map(str, read_write[:MAX_POLICY_ITEMS]))}")
    process = data.get("process") if isinstance(data.get("process"), dict) else {}
    if process:
        lines.append(
            "        process: "
            f"user={process.get('run_as_user', '<unset>')} group={process.get('run_as_group', '<unset>')}"
        )
    landlock = data.get("landlock") if isinstance(data.get("landlock"), dict) else {}
    if landlock:
        lines.append(f"        landlock: compatibility={landlock.get('compatibility', '<unset>')}")
    network = data.get("network_policies") if isinstance(data.get("network_policies"), dict) else {}
    lines.append(f"        network policies: {len(network)}")
    endpoint_lines = _network_endpoint_lines(network)
    lines.extend(f"        {line}" for line in endpoint_lines[:MAX_POLICY_ITEMS])
    lines.extend(_overflow_lines(len(endpoint_lines), "endpoint(s)"))
    return lines


def _render_guardrails_workflow(path: Path) -> list[str]:
    """Render the run's Relay guardrail set for the console report."""
    if not path.exists():
        return [f"      guardrail config missing: {path}"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"      guardrail config unreadable: {exc}"]

    guardrails = configured_guardrails(data)
    lines = [f"      guardrail config contents: {len(guardrails)} tool guardrail(s)"]
    for index, rail in enumerate(guardrails[:MAX_POLICY_ITEMS], start=1):
        target = rail.get("target_tool", "<unknown>")
        action = rail.get("action", "<unset>")
        threshold = rail.get("threshold", "<unset>")
        lines.append(
            f"        {index}. guardrail={rail.get('name', '<unnamed>')} target={target} "
            f"action={action} threshold={threshold}"
        )
        instructions = rail.get("system_instructions")
        if isinstance(instructions, str) and instructions.strip():
            lines.append(f"           instructions: {short_text(instructions, 240)}")
    lines.extend(_overflow_lines(len(guardrails), "tool guardrail(s)"))
    return lines


def _render_victim_control(victim_control: dict[str, Any]) -> list[str]:
    lines = [
        "  victim-control apply:",
        f"      ok: {fmt_bool(victim_control.get('ok'))}",
        f"      summary: {fmt_value(victim_control.get('summary'))}",
        f"      sandbox recreate/restart requested: {fmt_bool(victim_control.get('redeploy_intent'))}",
    ]
    metadata = victim_control.get("metadata") if isinstance(victim_control.get("metadata"), dict) else {}
    results = metadata.get("results") if isinstance(metadata.get("results"), list) else []
    if not results:
        lines.append("      adapter results: <none>")
        return lines
    for result in results:
        if not isinstance(result, dict):
            continue
        lines.append(f"      - {result.get('patch_type', '<unknown>')}: ok={fmt_bool(result.get('ok'))}")
        for key in ("status", "candidate_policy_path", "effective_policy_path", "target_workflow_path", "output"):
            if key in result:
                lines.append(f"          {key}: {fmt_value(result.get(key))}")
    return lines


def _operation_summary(operation: dict[str, Any]) -> str:
    parts = [f"op={operation.get('op', '<unknown>')}"]
    for key in ("network_policy", "path", "user", "group", "source_finding_id"):
        if operation.get(key):
            parts.append(f"{key}={operation[key]}")
    match = operation.get("match")
    if isinstance(match, dict):
        host = match.get("host")
        port = match.get("port")
        if host or port:
            parts.append(f"match={host or '*'}:{port or '*'}")
    for key in ("access", "enforcement", "tls", "include_workdir"):
        if key in operation:
            parts.append(f"{key}={operation[key]}")
    if isinstance(operation.get("rules"), list):
        parts.append(f"rules={len(operation['rules'])}")
    if isinstance(operation.get("binaries"), list):
        parts.append(f"binaries={len(operation['binaries'])}")
    return " ".join(parts)


def _network_endpoint_lines(network: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for policy_name, policy in network.items():
        if not isinstance(policy, dict):
            continue
        endpoints = policy.get("endpoints") if isinstance(policy.get("endpoints"), list) else []
        binaries = policy.get("binaries") if isinstance(policy.get("binaries"), list) else []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            host = endpoint.get("host", "<unknown>")
            port = endpoint.get("port", "*")
            access = endpoint.get("access", "<unset>")
            enforcement = endpoint.get("enforcement", "<unset>")
            rules = endpoint.get("rules") if isinstance(endpoint.get("rules"), list) else []
            lines.append(
                f"{policy_name}: {host}:{port} access={access} enforcement={enforcement} "
                f"rules={len(rules)} binaries={len(binaries)}"
            )
    return lines
