# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay generated benign requests against the victim and judge refusal.

Public API:

- :class:`ReplayJudgeConfig` — runtime configuration for a replay+judge pass
- :func:`run_all` — replay every :class:`GeneratedRequest` and return per-row results
- :func:`format_finding` / :func:`stats_key` — helpers for assembling the final
  :class:`agent_hardener.models.ValidatorReport`
"""

from __future__ import annotations

from .config import ReplayJudgeConfig, ReplayMode
from .reporting import VerdictStatus, format_finding, stats_key
from .runner import run_all

__all__ = [
    "ReplayJudgeConfig",
    "ReplayMode",
    "VerdictStatus",
    "format_finding",
    "run_all",
    "stats_key",
]
