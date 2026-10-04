# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the synth artifacts to disk: profile.json, requests.csv, sources/, input_hash.txt."""

from __future__ import annotations

import csv
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest

    from ..state import ProfileWriterState

logger = logging.getLogger(__name__)

CSV_HEADER = ("tool", "payload", "label", "rationale", "persona")


async def write(state: ProfileWriterState) -> dict[str, object]:
    """Write all artifacts; surface a partial-write state on I/O failure."""
    if state.profile is None:
        return {
            "written_files": [],
            "errors": ["profile_writer: no profile to write"],
            "source_note": "skipped (no profile)",
        }

    written: list[Path] = []
    try:
        state.target_dir.mkdir(parents=True, exist_ok=True)
        written.append(_write_profile(state.target_dir, state))
        written.append(_write_requests_csv(state.target_dir, state.requests))
        written.extend(_write_source_notes(state.target_dir, state.source_notes))
        written.append(_write_input_hash(state.target_dir, state.profile.input_hash))
    except OSError as exc:
        logger.exception("profile_writer file I/O failed")
        return {
            "written_files": written,
            "errors": [f"profile_writer: {exc}"],
            "source_note": f"partial write to {state.target_dir} ({len(written)} file(s))",
        }

    logger.info("profile_writer wrote %d artifact(s) to %s", len(written), state.target_dir)
    return {
        "written_files": written,
        "source_note": f"wrote {len(written)} artifact(s) to {state.target_dir}",
    }


def _write_profile(target_dir: Path, state: ProfileWriterState) -> Path:
    path = target_dir / "profile.json"
    assert state.profile is not None
    path.write_text(state.profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def _write_requests_csv(target_dir: Path, requests: list[GeneratedRequest]) -> Path:
    path = target_dir / "requests.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for row in requests:
            writer.writerow([row.tool, row.payload, row.label, row.rationale, row.persona or ""])
    return path


def _write_source_notes(target_dir: Path, notes: dict[str, str]) -> list[Path]:
    if not notes:
        return []
    sources_dir = target_dir / "sources"
    sources_dir.mkdir(exist_ok=True)
    out: list[Path] = []
    for source, note in notes.items():
        path = sources_dir / f"{source}.json"
        path.write_text(json.dumps({"note": note}, indent=2), encoding="utf-8")
        out.append(path)
    return out


def _write_input_hash(target_dir: Path, input_hash: str) -> Path:
    path = target_dir / "input_hash.txt"
    path.write_text(input_hash + "\n", encoding="utf-8")
    return path
