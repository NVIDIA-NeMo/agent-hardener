# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward run events to an HTTP sink (the NeMo plugin's SSE ingest) when configured.

Set ``AGENT_HARDENER_EVENT_SINK_URL`` and every EventBus event is POSTed there as ``{event, payload}`` — the
plugin re-broadcasts to Studio over SSE. Unset (plain CLI runs) → no sink.

The sink must never block the run: ``EventBus.emit`` calls subscribers inline, so the returned callback only
enqueues (instantly) and a daemon worker thread does the POSTs. Best-effort — a full queue drops events and
POST failures are swallowed, so a slow/absent plugin never stalls the war-game.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_ENVVAR = "AGENT_HARDENER_EVENT_SINK_URL"
_MAX_QUEUED = 2000


def make_http_event_sink() -> Callable[[str, dict[str, Any]], None] | None:
    """Return a non-blocking EventBus subscriber that POSTs events to ``$AGENT_HARDENER_EVENT_SINK_URL``, or None."""
    url = os.environ.get(_ENVVAR)
    if not url:
        return None

    events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=_MAX_QUEUED)

    def _worker() -> None:
        client = httpx.Client(timeout=5.0)
        while True:
            event, payload = events.get()
            try:
                client.post(url, json={"event": event, "payload": payload})
            except httpx.HTTPError:
                logger.debug("event sink POST to %s failed", url, exc_info=True)

    threading.Thread(target=_worker, daemon=True, name="agent-hardener-event-sink").start()

    def _sink(event: str, payload: dict[str, Any]) -> None:
        try:
            events.put_nowait((event, payload))
        except queue.Full:
            logger.debug("event sink queue full; dropping %s event", event)

    return _sink
