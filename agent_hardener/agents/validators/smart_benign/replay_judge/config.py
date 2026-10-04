# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for the replay + judge pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReplayMode = Literal["openai_chat", "json"]


@dataclass(frozen=True)
class ReplayJudgeConfig:
    """All knobs the replay+judge pass needs at runtime.

    Constructed once per validator invocation by the smart benign validator's
    top-level ``run`` from ``agent.config`` + defaults; not validated here
    (the parser is the validation boundary).
    """

    # Replay
    replay_url: str
    replay_mode: ReplayMode
    model: str
    input_field: str
    response_json_path: str | None
    timeout_seconds: float
    excerpt_chars: int
    concurrency: int

    # Judge
    confidence_cutoff: float
    judge_model: str
    judge_base_url: str
    judge_api_key: str
    judge_max_tokens: int
    judge_timeout_seconds: float
    judge_input_chars: int
    judge_temperature: float | None
