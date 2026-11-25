# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop replaying hits once they keep failing the same way.

The same error on several hits in a row comes from the setup (a missing garak interpreter, a dead
detector endpoint), not from the hits, and every further hit would only wait out the same failure.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Enough identical failures in a row to rule out a one-off blip, few enough to skip most of a large replay.
DEFAULT_LIMIT = 5


@dataclass(slots=True)
class ErrorStreak:
    """Identical consecutive hit errors, counted in completion order; trips at ``limit``."""

    limit: int = DEFAULT_LIMIT
    _error: str | None = None
    _count: int = 0
    #: Why the remaining hits are skipped, once the streak has tripped.
    abort_reason: str | None = None

    def record_success(self) -> None:
        if self.abort_reason is None:
            self._error, self._count = None, 0

    def record_error(self, error: str) -> None:
        if self.abort_reason is not None:
            return
        self._count = self._count + 1 if error == self._error else 1
        self._error = error
        if self._count >= self.limit:
            self.abort_reason = f"stopped after {self._count} hits in a row failed with: {error}"
