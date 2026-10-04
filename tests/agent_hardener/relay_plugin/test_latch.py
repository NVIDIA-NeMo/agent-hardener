# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eviction rules for the blocked-request map, tested without a Relay runtime.

The map is process-global in a victim that serves a whole war-game from one process, so the bounds
are the part worth pinning down: an entry that never expires would eventually answer an innocent
request with the blocked message.
"""

from __future__ import annotations

import pytest

from agent_hardener.relay_plugin.latch import BlockedRequests

pytestmark = pytest.mark.unit


def test_a_marked_request_is_blocked_and_others_are_not() -> None:
    blocked = BlockedRequests()
    blocked.mark("req-a")

    assert blocked.is_blocked("req-a")
    assert not blocked.is_blocked("req-b")


def test_discarding_releases_the_request() -> None:
    """The ordinary path: the blocked answer went out, so the entry has done its job."""
    blocked = BlockedRequests()
    blocked.mark("req-a")
    blocked.discard("req-a")

    assert not blocked.is_blocked("req-a")


@pytest.mark.parametrize("key", ["", None])
def test_an_unusable_key_is_never_blocked(key: object) -> None:
    """No key means no request identity, which must read as "allow" rather than "block everything"."""
    blocked = BlockedRequests()
    blocked.mark(key)  # type: ignore[arg-type]

    assert not blocked.is_blocked(key)  # type: ignore[arg-type]


def test_entries_expire() -> None:
    """A request that ends without a follow-up model call never releases its own entry."""
    blocked = BlockedRequests(ttl_seconds=-1.0)
    blocked.mark("req-a")

    assert not blocked.is_blocked("req-a")


def test_the_map_is_bounded() -> None:
    """A long-lived victim must not accumulate entries without limit."""
    blocked = BlockedRequests(max_entries=3)
    for index in range(10):
        blocked.mark(f"req-{index}")

    still_held = [index for index in range(10) if blocked.is_blocked(f"req-{index}")]

    assert len(still_held) == 3
    assert still_held == [7, 8, 9]  # oldest evicted first


def test_re_marking_an_active_request_is_harmless() -> None:
    """Two rails refusing in the same request is ordinary, not a fault."""
    blocked = BlockedRequests()
    blocked.mark("req-a")
    blocked.mark("req-a")

    assert blocked.is_blocked("req-a")
    blocked.discard("req-a")
    assert not blocked.is_blocked("req-a")
