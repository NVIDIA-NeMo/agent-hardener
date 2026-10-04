# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leaf helpers shared across final-log parsing and rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_hardener.final_log.models import SCANNER_LABELS, ScanStats
from agent_hardener.storage import to_jsonable

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterator

    from agent_hardener.models.reports import RoundIterationReport, RoundReport


def reversed_iterations(reports: list[RoundReport]) -> Iterator[RoundIterationReport]:
    """Yield every iteration newest-first (last report, last iteration first) for last-match scans."""
    for report in reversed(reports):
        yield from reversed(report.iterations)


def _as_report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    model_dump = getattr(report, "model_dump", None)
    data = model_dump(mode="json") if callable(model_dump) else to_jsonable(report)
    if not isinstance(data, dict) and hasattr(report, "__dict__"):
        data = {key: to_jsonable(value) for key, value in vars(report).items() if not key.startswith("_")}
    if not isinstance(data, dict):
        data = {
            key: to_jsonable(getattr(report, key))
            for key in ("round_id", "mission_id", "success", "storage_dir", "attacks", "iterations")
            if hasattr(report, key)
        }
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc), "_path": str(path)}
    return data if isinstance(data, dict) else {"_value": data, "_path": str(path)}


def _path_mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _iterations(report: dict[str, Any]) -> list[dict[str, Any]]:
    iterations = report.get("iterations") if isinstance(report.get("iterations"), list) else []
    return [iteration for iteration in iterations if isinstance(iteration, dict)]


def _policy_patches(iteration: dict[str, Any]) -> list[dict[str, Any]]:
    patches = iteration.get("policy_patches") if isinstance(iteration.get("policy_patches"), list) else []
    return [patch for patch in patches if isinstance(patch, dict)]


def _defenders(iteration: dict[str, Any]) -> list[dict[str, Any]]:
    defenders = iteration.get("defenders") if isinstance(iteration.get("defenders"), list) else []
    if not defenders and isinstance(iteration.get("defender_analyses"), list):
        defenders = iteration["defender_analyses"]
    return [defender for defender in defenders if isinstance(defender, dict)]


def _round_dir_from_report(report: dict[str, Any]) -> Path | None:
    storage_dir = report.get("storage_dir")
    if not storage_dir:
        return None
    path = Path(str(storage_dir))
    # Swarm Tracker uses ``round_<N>`` and the report's storage_dir IS the round dir; older runs nested
    # it under a ``loop-<N>`` parent. Accept both (but NOT the ``round-<N>`` round id), checking the dir
    # itself before its parents.
    for candidate in (path, *path.parents):
        if candidate.name.startswith(("round_", "loop_", "loop-")):
            return candidate
    return None


def _empty_scan_stats() -> dict[str, ScanStats]:
    return {scanner: ScanStats(scanner=scanner) for scanner in SCANNER_LABELS}


def _scan_has_data(scan: ScanStats) -> bool:
    return any(
        (
            scan.agent_hardener_hits,
            scan.hitlog_hits,
            scan.garak_total,
            scan.eval_rows,
            scan.statuses,
            scan.files,
            scan.job_ids,
            scan.yaml_names,
        )
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "<unknown>"
    return str(value)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return _fmt_bool(value)
    return _short_text(str(value), 260)


def _counter_summary(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _counter_keys(counter: Counter[str]) -> str:
    return ", ".join(sorted(counter))


def _short_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        turns = value.get("turns")
        if isinstance(turns, list):
            parts = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                content = turn.get("content")
                text = content.get("text") if isinstance(content, dict) else None
                if isinstance(text, str) and text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
        text = value.get("text")
        if isinstance(text, str):
            return text
    return _short_text(json.dumps(value, sort_keys=True, default=str), 400)


def _classify_scanner(values: list[Any]) -> str:
    text = " ".join(_flatten_text(value) for value in values).casefold()
    normalized = text.replace("-", "_").replace(" ", "_")
    if any(term in normalized for term in ("agent_breaker", "agentbreaker")):
        return "agent_breaker"
    if any(term in normalized for term in ("indirect", "latentinjection", "latent_injection", "prompt_injection")):
        return "indirect"
    return "other"


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return str(value)
    if isinstance(value, dict):
        return " ".join(f"{key}={_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list | tuple | set):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)
