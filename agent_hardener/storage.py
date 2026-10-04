# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON storage helpers for session artifacts."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SAFE_DIR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def create_mission_dir(root_dir: Path, mission_id: str) -> Path:
    """Create and return a mission-specific directory."""
    mission_dir = _safe_child_dir(root_dir, mission_id, "mission_id")
    mission_dir.mkdir(parents=True, exist_ok=False)
    return mission_dir


def create_session_dir(root_dir: Path, round_id: str) -> Path:
    """Create and return a session-specific directory."""
    round_dir = _safe_child_dir(root_dir, round_id, "round_id")
    round_dir.mkdir(parents=True, exist_ok=False)
    return round_dir


def write_json(path: Path, payload: Any) -> None:
    """Write payload as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(payload: Any) -> Any:  # noqa: PLR0911 - flat dispatch is clearer than nesting.
    """Convert Pydantic models and common runtime values into JSON-safe data."""
    if payload is None or isinstance(payload, str | int | float | bool):
        return payload
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {str(key): to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [to_jsonable(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, datetime | date):
        return payload.isoformat()
    return payload


def find_latest_hitlog(report_dir: Path) -> Path | None:
    """Return the most recent garak hitlog under report_dir (searched recursively), or None.

    Garak hitlogs are written run-scoped under ``run-logs/<run_id>/round_<N>/garak/``, so the
    search recurses; newest-by-mtime selects the latest run's latest round.
    """
    if not report_dir.exists():
        return None
    candidates = [*report_dir.rglob("*.hitlog.jsonl"), *report_dir.rglob("*.hitLog.jsonl")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def _safe_child_dir(root_dir: Path, name: str, label: str) -> Path:
    if not _SAFE_DIR_NAME.fullmatch(name):
        msg = f"{label} must be a safe path segment"
        raise ValueError(msg)
    return root_dir / name
