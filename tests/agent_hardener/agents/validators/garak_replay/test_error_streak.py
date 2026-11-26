# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only the same error, uninterrupted, stops a replay."""

from __future__ import annotations

from agent_hardener.agents.validators.garak_replay.error_streak import ErrorStreak


def test_trips_on_the_limit_th_identical_error() -> None:
    streak = ErrorStreak(limit=3)
    for _ in range(2):
        streak.record_error("boom")
    assert streak.abort_reason is None
    streak.record_error("boom")
    assert streak.abort_reason == "stopped after 3 hits in a row failed with: boom"


def test_a_success_or_a_different_error_restarts_the_count() -> None:
    streak = ErrorStreak(limit=3)
    for error in ("boom", "boom", None, "boom", "boom", "other", "boom", "boom"):
        if error is None:
            streak.record_success()
        else:
            streak.record_error(error)
    assert streak.abort_reason is None


def test_once_tripped_the_reason_stays() -> None:
    streak = ErrorStreak(limit=2)
    streak.record_error("boom")
    streak.record_error("boom")
    streak.record_success()
    streak.record_error("other")
    assert streak.abort_reason == "stopped after 2 hits in a row failed with: boom"
