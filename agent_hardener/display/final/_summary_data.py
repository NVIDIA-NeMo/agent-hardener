# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, format-agnostic extraction of the final-report summary facts.

The plain-text and Rich renderers need the same per-session overview and the same flat lists of
defender/validator records. Computing them here once (as plain data) keeps the two renderers to
pure formatting and removes the duplicated traversal/aggregation that previously lived in both.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from typing import Any

from agent_hardener.final_log import (
    SCANNERS_WITH_OTHER,
    artifact_paths_from_reports,
    collect_attack_stats,
    counter_summary,
    merge_garak_artifacts,
)
from agent_hardener.final_log import defenders as _defenders
from agent_hardener.final_log import iterations as _iterations


@dataclass(frozen=True)
class Tally:
    """``passed`` of ``total`` (``total`` is None when the validator did not run), plus a short breakdown."""

    passed: int = 0
    total: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class RoundOutcome:
    """A round's headline: attack hits found, how many the replay blocked, how many benign requests still work."""

    attack_hits: int
    blocked: Tally
    benign: Tally


def _breakdown(metadata: dict[str, Any], keys: tuple[tuple[str, str], ...]) -> str:
    """``(3 refused, 2 errors)`` from the non-zero counts among ``keys``; empty when all are zero."""
    parts = [f"{metadata[key]} {label}" for key, label in keys if metadata.get(key)]
    return f"({', '.join(parts)})" if parts else ""


def round_outcome(report: dict[str, Any]) -> RoundOutcome:
    """Attack hits for the round, and the last iteration's attack-replay and benign validator tallies."""
    stats = collect_attack_stats([report])
    merge_garak_artifacts(stats, artifact_paths_from_reports([report]))
    blocked = benign = Tally()
    report_iterations = _iterations(report)
    for validator in report_iterations[-1].get("validators") or [] if report_iterations else []:
        metadata = validator.get("metadata") if isinstance(validator, dict) else None
        if not isinstance(metadata, dict):
            continue
        if validator.get("kind") == "attack" and "total_attacks" in metadata:
            detail = _breakdown(metadata, (("not_blocked_count", "not blocked"), ("error_count", "errors")))
            blocked = Tally(int(metadata.get("blocked_count") or 0), int(metadata["total_attacks"]), detail)
        elif validator.get("kind") == "benign" and "total" in metadata:
            detail = _breakdown(metadata, (("refused_count", "refused"), ("error_count", "errors")))
            benign = Tally(int(metadata.get("complied_count") or 0), int(metadata["total"]), detail)
    hits = sum(stats[scanner].displayed_successes for scanner in SCANNERS_WITH_OTHER)
    return RoundOutcome(attack_hits=hits, blocked=blocked, benign=benign)


@dataclass(frozen=True)
class OverviewRow:
    """One session's headline facts for the overview table/section."""

    round_id: str
    success: Any
    iterations: int
    storage: str
    duration: str
    errors: list[str]
    outcome: RoundOutcome


@dataclass(frozen=True)
class SessionOverview:
    """Aggregate + per-session overview shared by both renderers."""

    missions: list[str]
    total: int
    succeeded: int
    failed: int
    rows: list[OverviewRow]
    storage_dirs: list[str]
    duration: str

    @property
    def errored(self) -> int:
        """Rounds in which at least one stage recorded an error."""
        return sum(1 for row in self.rows if row.errors)

    @property
    def logs_dir(self) -> str | None:
        """Where the run's logs live: the shared parent of every round dir, else the latest round dir."""
        if not self.storage_dirs:
            return None
        parents = {str(PurePath(path).parent) for path in self.storage_dirs}
        return parents.pop() if len(parents) == 1 else self.storage_dirs[-1]


def _timestamp(value: Any) -> datetime | None:
    """Parse a report's ``started_at``/``ended_at`` (a datetime, or its ISO string once dumped to JSON)."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def _seconds(report: dict[str, Any]) -> float | None:
    started, ended = _timestamp(report.get("started_at")), _timestamp(report.get("ended_at"))
    if started is None or ended is None:
        return None
    return max((ended - started).total_seconds(), 0.0)


def format_duration(seconds: float | None) -> str:
    """``38s`` / ``4m 12s`` / ``1h 03m``; empty when unknown."""
    if seconds is None:
        return ""
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _record_error(record: Any, *, ok_means_error: bool) -> str | None:
    """A stage record's error text: its ``error``, else ``failed`` when ``ok`` is False and that means an error."""
    if not isinstance(record, dict):
        return None
    error = record.get("error")
    if error and str(error) != "None":
        return str(error)
    return "failed" if ok_means_error and record.get("ok") is False else None


def round_errors(report: dict[str, Any]) -> list[str]:
    """Every stage error in a round, as ``<stage or agent>: <error>``.

    A validator with ``ok=False`` but no ``error`` ran and judged the defense insufficient: that is a
    result, not an error, so it is not listed. Any other stage reporting ``ok=False`` failed to run.
    """
    errors: list[str] = []

    def add(label: str, record: Any, *, ok_means_error: bool = True) -> None:
        error = _record_error(record, ok_means_error=ok_means_error)
        if error:
            name = record.get("agent_name") or record.get("agent_id") or label
            errors.append(f"{name}: {error}")

    for attack in report.get("attacks") or []:
        add("attacker", attack)
    for iteration in _iterations(report):
        for defender in _defenders(iteration):
            add("defender", defender)
        control = iteration.get("victim_control")
        add("deploy", {**control, "agent_name": "deploy"} if isinstance(control, dict) else None)
        add("victim", iteration.get("victim"))
        for validator in iteration.get("validators") or []:
            add("validator", validator, ok_means_error=False)
    return errors


def session_overview(reports: list[dict[str, Any]], mission_ids: list[str] | None) -> SessionOverview:
    """Compute the mission list, success/failure counts, and per-session rows once for both renderers."""
    succeeded = sum(1 for report in reports if report.get("success") is True)
    missions = mission_ids or sorted({str(report.get("mission_id")) for report in reports if report.get("mission_id")})
    storage_dirs = [str(report.get("storage_dir")) for report in reports if report.get("storage_dir")]
    rows = [
        OverviewRow(
            round_id=str(report.get("round_id", "<unknown>")),
            success=report.get("success"),
            iterations=len(_iterations(report)),
            storage=str(report.get("storage_dir", "<unknown>")),
            duration=format_duration(_seconds(report)),
            errors=round_errors(report),
            outcome=round_outcome(report),
        )
        for report in reports
    ]
    durations = [seconds for seconds in map(_seconds, reports) if seconds is not None]
    return SessionOverview(
        missions=missions,
        total=len(reports),
        succeeded=succeeded,
        failed=len(reports) - succeeded,
        rows=rows,
        storage_dirs=storage_dirs,
        duration=format_duration(sum(durations)) if durations else "",
    )


def collect_defenders(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every defender record across all reports' iterations (dicts only)."""
    return [
        defender
        for report in reports
        for iteration in _iterations(report)
        for defender in _defenders(iteration)
        if isinstance(defender, dict)
    ]


def collect_validators(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every validator record across all reports' iterations (dicts only)."""
    return [
        validator
        for report in reports
        for iteration in _iterations(report)
        for validator in iteration.get("validators", [])
        if isinstance(validator, dict)
    ]


@dataclass(frozen=True)
class DefenderRow:
    """One defender's results across every attack it was routed (the manager runs it once per attack)."""

    name: str
    runs: int
    succeeded: int
    defenses: str
    targets: str
    error: str | None


def _is_policy_patch(patch: Any) -> bool:
    return isinstance(patch, dict) and (patch.get("type") == "openshell_policy_candidate" or "new_policy_yaml" in patch)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _defenses_text(records: list[dict[str, Any]]) -> str:
    """What a defender's runs produced: ``2 guardrails``, ``3 policy rules (l7_deny_rule)``, or ``none``."""
    guardrails: set[str] = set()
    policy_kinds: Counter[str] = Counter()
    other = 0
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        patches = [patch for patch in record.get("policy_patches") or [] if isinstance(patch, dict)]
        if metadata.get("guardrail_name"):
            guardrails.add(str(metadata["guardrail_name"]))
        elif any(_is_policy_patch(patch) for patch in patches):
            policy_kinds[str(metadata.get("resource_type") or "policy")] += 1
        elif any(patch.get("changed") is not False for patch in patches):
            other += 1
    parts = []
    if guardrails:
        parts.append(_plural(len(guardrails), "guardrail"))
    if policy_kinds:
        kinds = ", ".join(kind for kind in sorted(policy_kinds) if kind != "policy")
        parts.append(_plural(sum(policy_kinds.values()), "policy rule") + (f" ({kinds})" if kinds else ""))
    if other:
        parts.append(_plural(other, "other change"))
    return ", ".join(parts) or "none"


def defender_rows(defenders: list[dict[str, Any]]) -> list[DefenderRow]:
    """Group defender records by defender, in first-seen order, and say what each one added and where."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for defender in defenders:
        name = str(defender.get("agent_name") or defender.get("agent_id") or "<unknown>")
        grouped.setdefault(name, []).append(defender)
    rows = []
    for name, records in grouped.items():
        tools = {
            str(record["metadata"]["attacked_tool"])
            for record in records
            if isinstance(record.get("metadata"), dict) and record["metadata"].get("attacked_tool")
        }
        errors = [detail.text for detail in map(result_detail, records) if detail.is_error]
        rows.append(
            DefenderRow(
                name=name,
                runs=len(records),
                succeeded=sum(1 for record in records if record.get("ok", True)),
                defenses=_defenses_text(records),
                targets=", ".join(sorted(tools)),
                error=errors[0] if errors else None,
            )
        )
    return rows


@dataclass(frozen=True)
class ResultDetail:
    """The one detail line for a defender/validator row: its error if any, else its summary."""

    text: str
    is_error: bool


def result_detail(record: dict[str, Any]) -> ResultDetail:
    """Shape a defender/validator record's detail — error takes precedence over summary."""
    error = record.get("error")
    if error and str(error) != "None":
        return ResultDetail(str(error), is_error=True)
    return ResultDetail(str(record.get("summary") or ""), is_error=False)


@dataclass(frozen=True)
class ValidatorRow:
    """One validator's row: shared by the Rich table and the plain lines."""

    name: str
    kind: str
    ok: bool
    detail: ResultDetail


@dataclass(frozen=True)
class ValidatorSummary:
    """Per-validator rows plus the per-kind ok/failed tallies both renderers show as a caption."""

    rows: list[ValidatorRow]
    kind_counts: dict[str, Counter[str]]

    @property
    def caption(self) -> str:
        """``attack: failed=1, ok=5 | benign: ok=3`` — identical in both renderers."""
        return " | ".join(f"{kind}: {counter_summary(counter)}" for kind, counter in sorted(self.kind_counts.items()))


def validator_summary(validators: list[dict[str, Any]]) -> ValidatorSummary:
    """Compute the per-validator rows and per-kind tallies once for both renderers."""
    rows: list[ValidatorRow] = []
    kind_counts: dict[str, Counter[str]] = {}
    for validator in validators:
        kind = str(validator.get("kind") or "unknown")
        ok = bool(validator.get("ok", True))
        kind_counts.setdefault(kind, Counter())["ok" if ok else "failed"] += 1
        rows.append(
            ValidatorRow(
                name=str(validator.get("agent_name") or validator.get("agent_id") or "<unknown>"),
                kind=kind,
                ok=ok,
                detail=result_detail(validator),
            )
        )
    return ValidatorSummary(rows=rows, kind_counts=kind_counts)
