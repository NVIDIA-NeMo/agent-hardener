# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which requests a guardrail has already refused.

The guardrail's intercepts are installed once, for the whole process, so each call has to ask
whether *this* request is the blocked one. Relay offers no request id and no context that survives
between callbacks — a ``ContextVar`` set inside a tool intercept reads back empty in the model
intercept that follows, because Relay drives each callback in its own context. What does survive is
the propagation root uuid, so that is the key and this is the map.

Bounded on purpose. A victim serves a whole war-game from one process, and the ordinary path removes
its own entry as soon as the blocked answer goes out; the cap and the expiry are for the requests
that never come back — a turn that ends after the refusal without asking the model anything. Without
them the map would only ever grow.

Kept free of ``nemo_relay`` imports, like :mod:`agent_hardener.relay_plugin.policy`, so the eviction
rules can be tested without a Relay runtime.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

#: Entries live at most this long. Comfortably longer than any single agent turn (the war-game's own
#: victim timeout is 120s), so expiry never races a request that is still running.
DEFAULT_TTL_SECONDS = 900.0

#: Hard cap on concurrent blocked requests held. Far above any real concurrency for one victim, and
#: small enough that a runaway cannot cost meaningful memory.
DEFAULT_MAX_ENTRIES = 4096


class BlockedRequests:
    """A bounded, expiring set of request keys, safe to touch from several calls at once."""

    def __init__(self, *, ttl_seconds: float = DEFAULT_TTL_SECONDS, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._marked: OrderedDict[str, float] = OrderedDict()

    def mark(self, key: str) -> None:
        """Record that this request has been refused and should go no further."""
        if not key:
            return
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            self._marked[key] = now
            self._marked.move_to_end(key)
            while len(self._marked) > self._max:
                self._marked.popitem(last=False)

    def is_blocked(self, key: str) -> bool:
        if not key:
            return False
        with self._lock:
            marked_at = self._marked.get(key)
            if marked_at is None:
                return False
            if time.monotonic() - marked_at > self._ttl:
                del self._marked[key]
                return False
            return True

    def discard(self, key: str) -> None:
        """Forget a request, once its blocked answer has gone out.

        Releasing here is what bounds the damage if a key ever turns out to be coarser than one
        request — a victim that runs every request under one propagation root would otherwise have
        a single refusal answer every later request with the blocked message. Released, the worst
        case is one wrong answer rather than an outage that lasts until restart.
        """
        if not key:
            return
        with self._lock:
            self._marked.pop(key, None)

    def _evict(self, now: float) -> None:
        """Drop expired entries. Cheap because the map is ordered oldest-first."""
        cutoff = now - self._ttl
        while self._marked:
            key, marked_at = next(iter(self._marked.items()))
            if marked_at > cutoff:
                return
            del self._marked[key]
