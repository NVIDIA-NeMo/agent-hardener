# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ExtractionCache``: per-key memoization used to dedupe LLM extraction calls across a run.

Scoped to one ``DefendersManager.run()`` call.
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.extraction_cache import ExtractionCache

pytestmark = pytest.mark.unit


def test_same_key_computes_once() -> None:
    cache = ExtractionCache()
    calls = []

    def compute():
        calls.append(1)
        return "result"

    assert cache.get_or_compute("k", compute) == "result"
    assert cache.get_or_compute("k", compute) == "result"
    assert len(calls) == 1


def test_different_keys_compute_independently() -> None:
    cache = ExtractionCache()
    assert cache.get_or_compute("a", lambda: "A") == "A"
    assert cache.get_or_compute("b", lambda: "B") == "B"


def test_tuple_keys_are_supported() -> None:
    cache = ExtractionCache()
    calls = []

    def compute():
        calls.append(1)
        return "result"

    key = ("benign", ("req-1", "req-2"))
    cache.get_or_compute(key, compute)
    cache.get_or_compute(("benign", ("req-1", "req-2")), compute)
    assert len(calls) == 1
