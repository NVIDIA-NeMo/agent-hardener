# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from agent_hardener.concurrency import agent_concurrency, concurrency_limit, gather_limited_ordered


def test_gather_limited_ordered_caps_concurrency_and_preserves_order() -> None:
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def worker(value: int) -> str:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.01 * (4 - value))
        async with lock:
            active -= 1
        return f"result-{value}"

    results = asyncio.run(gather_limited_ordered([1, 2, 3], 2, worker))

    assert results == ["result-1", "result-2", "result-3"]
    assert max_active == 2


def test_gather_limited_ordered_can_return_exceptions_in_order() -> None:
    async def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("boom")
        return value

    results = asyncio.run(gather_limited_ordered([1, 2, 3], 3, worker, return_exceptions=True))

    assert results[0] == 1
    assert isinstance(results[1], RuntimeError)
    assert results[2] == 3


def test_concurrency_limit_prefers_valid_env_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("AGENT_HARDENER_TEST_CONCURRENCY", "5")

    assert concurrency_limit(2, "AGENT_HARDENER_TEST_CONCURRENCY") == 5


def test_concurrency_limit_falls_back_for_invalid_or_non_positive_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("AGENT_HARDENER_TEST_CONCURRENCY", "bad")

    assert concurrency_limit(4, "AGENT_HARDENER_TEST_CONCURRENCY") == 4
    assert concurrency_limit(0, default=3) == 3


def test_agent_concurrency_resolution(monkeypatch: Any) -> None:
    context = {"run_concurrency": {"benign_validator_rows": 4, "guardrails_findings": 6}}

    assert agent_concurrency({}, context, "guardrails_findings", "AGENT_HARDENER_MISSING") == 6
    assert agent_concurrency({"concurrency": {"guardrails_findings": 3}}, context, "guardrails_findings", "X") == 3
    assert agent_concurrency({"guardrails_findings_concurrency": 7}, context, "guardrails_findings", "X") == 7

    monkeypatch.setenv("AGENT_HARDENER_AGENT_CONCURRENCY", "8")
    assert (
        agent_concurrency(
            {"concurrency": {"guardrails_findings": 3}},
            context,
            "guardrails_findings",
            "AGENT_HARDENER_AGENT_CONCURRENCY",
        )
        == 8
    )
