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
from typing import Any

from agent_hardener.final_log import counter_summary
from agent_hardener.final_log import defenders as _defenders
from agent_hardener.final_log import iterations as _iterations


@dataclass(frozen=True)
class OverviewRow:
    """One session's headline facts for the overview table/section."""

    round_id: str
    success: Any
    iterations: int
    storage: str


@dataclass(frozen=True)
class SessionOverview:
    """Aggregate + per-session overview shared by both renderers."""

    missions: list[str]
    total: int
    succeeded: int
    failed: int
    rows: list[OverviewRow]
    storage_dirs: list[str]


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
        )
        for report in reports
    ]
    return SessionOverview(
        missions=missions,
        total=len(reports),
        succeeded=succeeded,
        failed=len(reports) - succeeded,
        rows=rows,
        storage_dirs=storage_dirs,
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
