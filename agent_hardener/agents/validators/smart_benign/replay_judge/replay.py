# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP replay against the victim.

Thin adapter around :func:`agent_hardener.agents.validators.replay_http.replay_prompt`
that injects this package's source label and excerpt configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.replay_http import ReplayResult
from agent_hardener.agents.validators.replay_http import replay_prompt as replay_http_prompt

from .reporting import excerpt

if TYPE_CHECKING:
    import httpx

    from .config import ReplayJudgeConfig


async def replay_one(
    payload: str,
    config: ReplayJudgeConfig,
    client: httpx.AsyncClient,
) -> ReplayResult:
    """Replay a single payload against the configured victim endpoint."""
    return await replay_http_prompt(
        payload,
        config,
        client,
        rate_limit_source="smart benign validator victim replay",
        error_excerpt=lambda text: excerpt(text, config.excerpt_chars),
    )
