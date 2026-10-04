# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate reports and Garak artifacts into the shared FinalLogData."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_hardener.final_log.garak import (
    _append_example,
    _example_from_hit_record,
    _garak_detector_score_items,
    _merge_garak_artifacts,
    _read_jsonl,
    _record_probe_detector,
)
from agent_hardener.final_log.helpers import (
    _as_report_dict,
    _classify_scanner,
    _empty_scan_stats,
    _extract_text,
    _iterations,
    _load_json,
    _policy_patches,
)
from agent_hardener.final_log.models import AttemptTranscript, FinalLogData, ScanStats


def _collect_attack_stats(reports: list[dict[str, Any]]) -> dict[str, ScanStats]:
    stats = _empty_scan_stats()
    for report in reports:
        attacks = report.get("attacks") if isinstance(report.get("attacks"), list) else []
        for attack in attacks:
            if not isinstance(attack, dict):
                continue
            metadata = attack.get("metadata") if isinstance(attack.get("metadata"), dict) else {}
            records = attack.get("records") if isinstance(attack.get("records"), list) else []
            scanner = _classify_scanner([attack.get("agent_name"), metadata.get("yaml_name"), records])
            scan = stats[scanner]
            scan.agent_hardener_hits += len(records)
            scan.statuses["ok" if attack.get("ok", True) else "failed"] += 1
            if metadata.get("yaml_name"):
                scan.yaml_names.add(str(metadata["yaml_name"]))
            if metadata.get("garak_job_id"):
                scan.job_ids.add(str(metadata["garak_job_id"]))
            for record in records:
                if isinstance(record, dict):
                    _record_probe_detector(scan, record)
                    _append_example(scan, _example_from_hit_record(record, Path("<agent-hardener-report>")))
        for iteration in _iterations(report):
            _collect_attack_stats_from_policy_routes(stats, iteration)
    return stats


def _collect_attack_stats_from_policy_routes(stats: dict[str, ScanStats], iteration: dict[str, Any]) -> None:
    for patch in _policy_patches(iteration):
        for path_key in ("finding_routes_path", "per_finding_modifications_path"):
            path_value = patch.get(path_key)
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.exists():
                continue
            data = _load_json(path)
            if isinstance(data.get("findings"), list):
                for finding in data["findings"]:
                    if isinstance(finding, dict):
                        scanner = _classify_scanner([finding])
                        stats[scanner].files.add(path)


def _artifact_paths_from_reports(report_dicts: list[dict[str, Any]]) -> list[Path]:
    """Collect Garak artifact paths declared by agents via the artifacts field."""
    paths: set[Path] = set()
    for report in report_dicts:
        attacks = report.get("attacks") if isinstance(report.get("attacks"), list) else []
        for attack in attacks:
            if not isinstance(attack, dict):
                continue
            for artifact in attack.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                if artifact.get("type") in {"garak_report", "garak_hitlog"} and artifact.get("path"):
                    path = Path(str(artifact["path"]))
                    if path.exists():
                        paths.add(path)
    return sorted(paths)


def _collect_final_log_data(reports: list[Any]) -> FinalLogData:
    """Parse reports + Garak artifacts into the shared :class:`FinalLogData`."""
    report_dicts = [_as_report_dict(report) for report in reports]
    stats = _collect_attack_stats(report_dicts)
    garak_files = _artifact_paths_from_reports(report_dicts)
    _merge_garak_artifacts(stats, garak_files)
    return FinalLogData(report_dicts=report_dicts, stats=stats, garak_files=garak_files)


def _extract_outputs_text(outputs: Any) -> str:
    """Join the victim response text from a Garak attempt's ``outputs`` (a list of output objects)."""
    if isinstance(outputs, list):
        return "\n".join(text for text in (_extract_text(item) for item in outputs) if text)
    return _extract_text(outputs)


def _collect_attempt_transcripts(garak_files: list[Path]) -> list[AttemptTranscript]:
    """Read every ``entry_type == "attempt"`` row from the Garak report JSONL files.

    The report (unlike the hitlog) records ALL attempts — complied and refused — so this is the
    authoritative source for each attempt's full attack prompt and victim response.
    """
    transcripts: list[AttemptTranscript] = []
    for path in garak_files:
        if not path.name.endswith(".report.jsonl"):
            continue
        setup_scanner = _classify_scanner([path.name])
        for row in _read_jsonl(path):
            entry_type = row.get("entry_type")
            if entry_type == "start_run setup":
                setup_scanner = _classify_scanner(
                    [path.name, row.get("plugins.probe_spec"), row.get("reporting.report_prefix")]
                )
                continue
            if entry_type != "attempt":
                continue
            score_items = _garak_detector_score_items(row)
            transcripts.append(
                AttemptTranscript(
                    scanner=_classify_scanner([setup_scanner, path.name, row.get("probe_classname"), row.get("probe")]),
                    probe=str(row.get("probe_classname") or row.get("probe") or "<unknown>"),
                    prompt=_extract_text(row.get("prompt")),
                    response=_extract_outputs_text(row.get("outputs")),
                    hit=any(score > 0 for _detector, score in score_items),
                )
            )
    return transcripts
