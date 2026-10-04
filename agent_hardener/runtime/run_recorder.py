# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-round event recorder: the round's logger and event stream.

:class:`RunRecorder` owns the :func:`~agent_hardener.loggers.round_logging` lifecycle (logger + event
sink) and injects ``round_id``/``mission_id`` into every event so callers stop repeating them. Its
sibling :class:`~agent_hardener.runtime.round_store.RoundStore` owns the round's disk writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.loggers import round_logging

if TYPE_CHECKING:
    import logging
    from pathlib import Path
    from types import TracebackType

    from agent_hardener.loggers import JsonlEventLogger


class RunRecorder:
    """Own the round's logger/event sink for one round."""

    def __init__(self, round_id: str, mission_id: str | None, round_dir: Path) -> None:
        self.round_id = round_id
        self.mission_id = mission_id
        self._round_dir = round_dir
        self._cm: Any = None
        self.logger: logging.Logger
        self.events: JsonlEventLogger

    def __enter__(self) -> RunRecorder:
        """Open the round's logger and event sink."""
        self._cm = round_logging(self.round_id, self._round_dir)
        self.logger, self.events = self._cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Close the round's logger and event sink, flushing the run log."""
        return self._cm.__exit__(exc_type, exc, tb)

    def emit(self, event: str, **payload: Any) -> None:
        """Emit an event with ``round_id``/``mission_id`` filled in automatically."""
        self.events.emit(event, round_id=self.round_id, mission_id=self.mission_id, **payload)
