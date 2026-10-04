# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe per-key memoization scoped to a single ``DefendersManager.run()`` call.

Defenders are dispatched via ``asyncio.to_thread`` (see ``agent_hardener/runtime/stages/defense.py``),
so concurrent invocations within one run execute on real OS threads — locking here is
``threading.Lock``, not ``asyncio.Lock``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable


class ExtractionCache:
    """Memoizes ``compute()`` results by key, deduplicating concurrent first-callers per key."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._key_locks: dict[Hashable, threading.Lock] = {}
        self._results: dict[Hashable, Any] = {}

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        with self._guard:
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._results:
                self._results[key] = compute()
            return self._results[key]
