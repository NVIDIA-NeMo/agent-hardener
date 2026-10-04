# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Garak artifact parsing for the final log."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from agent_hardener.final_log.helpers import _classify_scanner, _extract_text, _int_or_none
from agent_hardener.final_log.models import (
    MAX_EXAMPLES_PER_SCANNER,
    HitExample,
    ScanStats,
)

if TYPE_CHECKING:
    from pathlib import Path


def _is_hitlog_path(path: Path) -> bool:
    return path.name.casefold().endswith(".hitlog.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _flatten_scores(value: Any) -> list[float]:
    scores: list[float] = []
    if value is None:
        return scores
    if isinstance(value, bool):
        scores.append(1.0 if value else 0.0)
    elif isinstance(value, int | float):
        scores.append(float(value))
    elif isinstance(value, str):
        with suppress(ValueError):
            scores.append(float(value))
    elif isinstance(value, dict):
        for nested in value.values():
            scores.extend(_flatten_scores(nested))
    elif isinstance(value, list | tuple | set):
        for nested in value:
            scores.extend(_flatten_scores(nested))
    return scores


def _garak_detector_score_items(row: dict[str, Any]) -> list[tuple[str, float]]:
    detector_results = row.get("detector_results")
    fallback_detector = str(row.get("detector_classname") or row.get("detector") or "<unknown>")
    if isinstance(detector_results, dict):
        items: list[tuple[str, float]] = []
        for detector, raw_scores in detector_results.items():
            items.extend((str(detector), score) for score in _flatten_scores(raw_scores))
        return items
    return [(fallback_detector, score) for score in _flatten_scores(detector_results)]


def _record_probe_detector(scan: ScanStats, record: dict[str, Any]) -> None:
    if record.get("probe"):
        scan.probes[str(record["probe"])] += 1
    if record.get("detector"):
        scan.detectors[str(record["detector"])] += 1


def _append_example(scan: ScanStats, example: HitExample) -> None:
    if len(scan.examples) < MAX_EXAMPLES_PER_SCANNER:
        scan.examples.append(example)


def _example_from_hit_record(record: dict[str, Any], source_path: Path) -> HitExample:
    return HitExample(
        source_path=source_path,
        probe=str(record.get("probe", "")),
        detector=str(record.get("detector", "")),
        prompt=_extract_text(record.get("prompt") or record.get("attack_prompt") or record.get("command")),
        output=_extract_text(record.get("output") or record.get("victim_response") or record.get("result")),
        score=record.get("score"),
    )


def _merge_garak_attempt_row(
    stats: dict[str, ScanStats],
    path: Path,
    row: dict[str, Any],
    setup_scanner: str,
) -> None:
    score_items = _garak_detector_score_items(row)
    if not score_items:
        return
    scanner = _classify_scanner(
        [
            setup_scanner,
            path.name,
            row.get("probe_classname"),
            row.get("probe"),
            row.get("detector_classname"),
            row.get("detector"),
            [detector for detector, _score in score_items],
        ]
    )
    scan = stats[scanner]
    successes = sum(1 for _detector, score in score_items if score > 0)
    failures = len(score_items) - successes
    probe = row.get("probe_classname") or row.get("probe")

    scan.garak_successes += successes
    scan.garak_failures += failures
    scan.garak_total += len(score_items)
    scan.has_garak_eval = True
    scan.eval_rows += 1
    scan.files.add(path)
    if probe:
        scan.probes[str(probe)] += len(score_items)
    for detector, _score in score_items:
        scan.detectors[detector] += 1


def _merge_garak_report(stats: dict[str, ScanStats], path: Path) -> None:
    rows = list(_read_jsonl(path))
    # eval rows are the authoritative probe/detector totals; only fall back to attempt rows for
    # attempt-only reports, otherwise the two would double-count the same scan.
    has_eval = any(row.get("entry_type") == "eval" for row in rows)
    setup_scanner = _classify_scanner([path.name])
    for row in rows:
        entry_type = row.get("entry_type")
        if entry_type == "start_run setup":
            setup_scanner = _classify_scanner(
                [
                    path.name,
                    row.get("plugins.probe_spec"),
                    row.get("plugins.detector_spec"),
                    row.get("reporting.report_prefix"),
                    row.get("transient.report_filename"),
                ]
            )
            continue
        if entry_type == "attempt":
            if not has_eval:
                _merge_garak_attempt_row(stats, path, row, setup_scanner)
            continue
        if entry_type != "eval":
            continue
        scanner = _classify_scanner([setup_scanner, path.name, row.get("probe"), row.get("detector")])
        scan = stats[scanner]
        passed = _int_or_none(row.get("passed"))
        total = _int_or_none(row.get("total"))
        if passed is None or total is None:
            continue
        successes = max(total - passed, 0)
        scan.garak_successes += successes
        scan.garak_failures += passed
        scan.garak_total += total
        scan.has_garak_eval = True
        scan.eval_rows += 1
        scan.files.add(path)
        if row.get("probe"):
            scan.probes[str(row["probe"])] += total
        if row.get("detector"):
            scan.detectors[str(row["detector"])] += total


def _merge_hitlog(stats: dict[str, ScanStats], path: Path) -> None:
    for row in _read_jsonl(path):
        scanner = _classify_scanner([path.name, row.get("probe"), row.get("detector"), row.get("goal"), row])
        scan = stats[scanner]
        scan.hitlog_hits += 1
        scan.files.add(path)
        _record_probe_detector(scan, row)
        _append_example(scan, _example_from_hit_record(row, path))


def _merge_garak_artifacts(stats: dict[str, ScanStats], paths: list[Path]) -> None:
    for path in paths:
        if path.name.endswith(".report.jsonl"):
            _merge_garak_report(stats, path)
        elif _is_hitlog_path(path):
            _merge_hitlog(stats, path)
