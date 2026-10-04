# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from agent_hardener.loggers import JsonlEventLogger, build_round_logger
from agent_hardener.storage import create_mission_dir, create_session_dir, to_jsonable

if TYPE_CHECKING:
    from pathlib import Path


def test_to_jsonable_handles_paths_datetimes_and_tuples(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path,
        "time": datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        "items": (tmp_path / "a",),
    }

    assert to_jsonable(payload) == {
        "path": str(tmp_path),
        "time": "2026-05-02T12:00:00+00:00",
        "items": [str(tmp_path / "a")],
    }


def test_storage_dirs_reject_unsafe_names(tmp_path: Path) -> None:
    for bad_name in ["", ".", "..", "../outside", "/tmp/outside", "nested/name", r"nested\name"]:
        with pytest.raises(ValueError, match="mission_id must be a safe path segment"):
            create_mission_dir(tmp_path, bad_name)
        with pytest.raises(ValueError, match="round_id must be a safe path segment"):
            create_session_dir(tmp_path, bad_name)


def test_storage_dirs_accept_safe_names(tmp_path: Path) -> None:
    mission_dir = create_mission_dir(tmp_path, "mission_20260504T000000Z-demo.1")
    round_dir = create_session_dir(mission_dir, "round-0001")

    assert round_dir == tmp_path / "mission_20260504T000000Z-demo.1" / "round-0001"
    assert round_dir.exists()


def test_build_round_logger_uses_single_logger_and_replaces_handlers(tmp_path: Path) -> None:
    dir_a = tmp_path / "session_a"
    dir_a.mkdir()
    dir_b = tmp_path / "session_b"
    dir_b.mkdir()

    first = build_round_logger("session-a", dir_a)
    first_handlers = list(first.handlers)
    second = build_round_logger("session-b", dir_b)

    assert second is first
    assert len(second.handlers) == 2
    assert all(new_handler is not old_handler for new_handler in second.handlers for old_handler in first_handlers)


def test_jsonl_event_logger_creates_parent_and_writes_json(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "events.jsonl"
    logger = JsonlEventLogger(log_path)

    logger.emit("test_event", path=tmp_path)

    assert '"event": "test_event"' in log_path.read_text(encoding="utf-8")
    assert str(tmp_path) in log_path.read_text(encoding="utf-8")
