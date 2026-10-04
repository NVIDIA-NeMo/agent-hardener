# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering of the Agent Hardener final log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_hardener.display.final._summary_data import (
    collect_defenders,
    collect_validators,
    session_overview,
    validator_summary,
)
from agent_hardener.display.policy_detail import render_aggregated_plan, render_policy_detail
from agent_hardener.display.render_diff import diff_plain_lines, load_yaml_diff_from_patch
from agent_hardener.final_log import (
    MAX_EXAMPLES_PER_SCANNER,
    MAX_POLICY_ITEMS,
    SCANNER_LABELS,
    SCANNERS,
    SCANNERS_WITH_OTHER,
    HitExample,
    ScanStats,
    artifact_paths_from_reports,
    collect_attack_stats,
    collect_final_log_data,
    counter_keys,
    counter_summary,
    fmt_bool,
    is_hitlog_path,
    iterations,
    merge_garak_artifacts,
    policy_patches,
    round_dir_from_report,
    scan_has_data,
    short_text,
)
from agent_hardener.final_log import (
    defenders as iteration_defenders,
)


def render_final_run_log(
    reports: list[Any],
    *,
    marker_path: Path | None = None,
    mission_ids: list[str] | None = None,
) -> str:
    """Render a readable final log for completed Agent Hardener reports."""
    data = collect_final_log_data(reports)
    lines = [
        "Agent Hardener final log",
        "=" * 80,
        *render_session_overview(data.report_dicts, mission_ids),
        "",
        *render_session_comparison(data.report_dicts),
        "",
        *render_attack_results(data.stats),
        "",
        *render_policy_results(data.report_dicts),
        "",
        *_render_yaml_diffs(data.report_dicts),
        "",
        *render_validator_results(data.report_dicts),
        "",
        *render_garak_artifacts(data.stats, data.garak_files),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_session_overview(reports: list[dict[str, Any]], mission_ids: list[str] | None) -> list[str]:
    """Render the mission list, success/failure counts, and per-session rows."""
    if not reports:
        return ["Session overview: no reports loaded"]
    overview = session_overview(reports, mission_ids)
    lines = [
        "Session overview:",
        f"  missions: {', '.join(overview.missions) if overview.missions else '<none>'}",
        f"  sessions: {overview.total} ({overview.succeeded} succeeded, {overview.failed} failed)",
    ]
    lines.extend(
        f"  - {row.round_id}: success={fmt_bool(row.success)}, iterations={row.iterations}, storage={row.storage}"
        for row in overview.rows
    )
    if overview.storage_dirs:
        lines.append(f"  latest storage dir: {overview.storage_dirs[-1]}")
    return lines


def render_session_comparison(reports: list[dict[str, Any]]) -> list[str]:
    """Render the before/after round comparison (attack-hit delta + policy/workflow changes)."""
    if len(reports) < 2:
        return []
    lines = ["Before/after round comparison:"]
    snapshots = []
    for index, report in enumerate(reports, start=1):
        stats = collect_attack_stats([report])
        merge_garak_artifacts(stats, artifact_paths_from_reports([report]))
        snapshots.append((report, stats))
        mission = report.get("mission_id", "<unknown>")
        session = report.get("round_id", "<unknown>")
        lines.append(f"  {index}. {mission}/{session}: success={fmt_bool(report.get('success'))}")
        for scanner in SCANNERS:
            lines.append(f"     {SCANNER_LABELS[scanner]}: {_comparison_attack_text(stats[scanner])}")
        lines.extend(_render_comparison_policy_lines(report))

    first_stats = snapshots[0][1]
    last_stats = snapshots[-1][1]
    lines.append("  before -> after attack hit delta:")
    for scanner in SCANNERS:
        before = first_stats[scanner].displayed_successes
        after = last_stats[scanner].displayed_successes
        delta = after - before
        blocked_text = "reduced" if delta < 0 else "unchanged" if delta == 0 else "increased"
        lines.append(
            f"     {SCANNER_LABELS[scanner]} successful hits: {before} -> {after} ({blocked_text}, delta={delta})"
        )
    return lines


def _comparison_attack_text(scan: ScanStats) -> str:
    return f"successful_hits={scan.displayed_successes}"


def _render_comparison_policy_lines(report: dict[str, Any]) -> list[str]:
    report_iterations = iterations(report)
    if not report_iterations:
        return ["     policy/workflow: no iterations"]
    iteration = report_iterations[-1]
    patches = policy_patches(iteration)
    openshell_patches = [patch for patch in patches if patch.get("type") == "openshell_policy_candidate"]
    workflow_patches = [patch for patch in patches if patch.get("type") == "victim_workflow_candidate"]
    defenders = iteration_defenders(iteration)
    defender_successes = sum(1 for defender in defenders if defender.get("ok", True))
    lines = [
        "     policy/workflow: "
        f"OpenShell candidate={fmt_bool(bool(openshell_patches))}, "
        f"guardrails candidate={fmt_bool(bool(workflow_patches))}, "
        f"defenders={defender_successes}/{len(defenders)} succeeded"
    ]
    round_dir = round_dir_from_report(report)
    if round_dir is not None:
        start_policy = round_dir / "openshell-policy.yaml"
        start_workflow = round_dir / "research_agent_workflow.yaml"
        if start_policy.exists():
            lines.append(f"     start OpenShell policy: {start_policy}")
        if start_workflow.exists():
            lines.append(f"     start guardrails workflow: {start_workflow}")
    for patch in openshell_patches[:1]:
        if patch.get("candidate_policy_path"):
            lines.append(f"     after OpenShell policy: {patch['candidate_policy_path']}")
        if patch.get("aggregated_patch_plan_path"):
            lines.extend(
                f"     {line}" for line in render_aggregated_plan(Path(str(patch["aggregated_patch_plan_path"])))
            )
    for patch in workflow_patches[:1]:
        if patch.get("target_workflow_path"):
            lines.append(f"     after guardrails workflow: {patch['target_workflow_path']}")
        elif patch.get("candidate_workflow_path"):
            lines.append(f"     after guardrails workflow candidate: {patch['candidate_workflow_path']}")
    return lines


def render_attack_results(stats: dict[str, ScanStats]) -> list[str]:
    """Render per-scanner attacker results (hit counts, evidence, probes, hit examples)."""
    lines = ["Attacker results:"]
    for scanner in SCANNERS_WITH_OTHER:
        scan = stats[scanner]
        if not scan_has_data(scan) and scanner == "other":
            continue
        label = SCANNER_LABELS[scanner]
        lines.append(f"  {label}: {scan.displayed_successes} successful exploit hit(s)")
        if scan.agent_hardener_hits or scan.hitlog_hits:
            lines.append(
                f"    evidence counts: Agent Hardener records={scan.agent_hardener_hits}, hitlog rows={scan.hitlog_hits}"
            )
        if scan.yaml_names:
            lines.append(f"    yaml: {', '.join(sorted(scan.yaml_names))}")
        if scan.job_ids:
            lines.append(f"    Garak job ids: {', '.join(sorted(scan.job_ids))}")
        if scan.probes:
            lines.append(f"    probes seen: {counter_keys(scan.probes)}")
        if scan.detectors:
            lines.append(f"    detectors seen: {counter_keys(scan.detectors)}")
        if scan.statuses:
            lines.append(f"    attacker agent status: {counter_summary(scan.statuses)}")
        for example in scan.examples[:MAX_EXAMPLES_PER_SCANNER]:
            lines.extend(_render_hit_example(example))
    return lines


def render_policy_results(reports: list[dict[str, Any]]) -> list[str]:
    """Render defender policy/guardrail detail across all reports' iterations."""
    patches = [patch for report in reports for iteration in iterations(report) for patch in policy_patches(iteration)]
    victim_controls = [
        iteration["victim_control"]
        for report in reports
        for iteration in iterations(report)
        if isinstance(iteration.get("victim_control"), dict)
    ]
    return render_policy_detail(collect_defenders(reports), patches, victim_controls)


def _render_yaml_diffs(reports: list[dict[str, Any]]) -> list[str]:
    """Render the previous → current YAML diffs as plain unified-diff text (non-TTY fallback)."""
    lines = ["YAML changes (previous -> current):"]
    found = False
    for report in reports:
        for iteration in iterations(report):
            for patch in policy_patches(iteration):
                resolved = load_yaml_diff_from_patch(patch)
                if resolved is None:
                    continue
                found = True
                label, diff = resolved
                lines.append(f"  {label}:")
                lines.extend(f"    {line}" for line in diff_plain_lines(diff))
    if not found:
        lines.append("  no YAML changes")
    return lines


def render_validator_results(reports: list[dict[str, Any]]) -> list[str]:
    """Render validator results: per-kind tallies, then one line per validator (same shaping as Rich)."""
    lines = ["Validator results:"]
    validators = collect_validators(reports)
    if not validators:
        return [*lines, "  no validators ran"]
    summary = validator_summary(validators)
    if summary.caption:
        lines.append(f"  by kind: {summary.caption}")
    for row in summary.rows:
        status = "ok" if row.ok else "FAILED"
        detail = f" — {short_text(row.detail.text, 120)}" if row.detail.text else ""
        lines.append(f"  - {row.name} ({row.kind}): {status}{detail}")
    return lines


def render_garak_artifacts(stats: dict[str, ScanStats], garak_files: list[Path]) -> list[str]:
    """Render the discovered Garak report/hitlog files and per-scanner file listings."""
    lines = ["Garak artifacts:"]
    if not garak_files:
        lines.append("  no Garak hitlogs or reports found")
        return lines
    report_files = [path for path in garak_files if path.name.endswith(".report.jsonl")]
    hitlog_files = [path for path in garak_files if is_hitlog_path(path)]
    lines.append(f"  report JSONL files: {len(report_files)}")
    for path in report_files[:MAX_POLICY_ITEMS]:
        lines.append(f"    {path}")
    lines.append(f"  hitlog JSONL files: {len(hitlog_files)}")
    for path in hitlog_files[:MAX_POLICY_ITEMS]:
        lines.append(f"    {path}")
    for scan in stats.values():
        if scan.files:
            lines.append(f"  {scan.scanner} files:")
            for path in sorted(scan.files)[:MAX_POLICY_ITEMS]:
                lines.append(f"    {path}")
    return lines


def _render_hit_example(example: HitExample) -> list[str]:
    lines = [
        "    hit example:",
        f"      file: {example.source_path}",
        f"      probe/detector: {example.probe or '<unknown>'} / {example.detector or '<unknown>'}",
    ]
    if example.score is not None:
        lines.append(f"      score: {example.score}")
    if example.prompt:
        lines.append(f"      prompt: {short_text(example.prompt, 220)}")
    if example.output:
        lines.append(f"      output: {short_text(example.output, 220)}")
    return lines
