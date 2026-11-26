# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward run events to an HTTP sink (the NeMo plugin's SSE ingest) when configured.

Set ``AGENT_HARDENER_EVENT_SINK_URL`` and every EventBus event is POSTed there as
``{event, payload, sink_id, seq}`` — the plugin stores it for Studio. Unset (plain CLI runs) → no sink.

The sink must never block the run: ``EventBus.emit`` calls subscribers inline, so the returned callback only
enqueues (instantly) and a daemon worker thread does the POSTs. Best-effort — a full queue drops events and
a POST that fails twice is dropped, so a slow/absent plugin never stalls the war-game.

``seq`` counts this sink's events and ``sink_id`` names the sink: one run has several (``serve`` and ``run``
each make one), so the pair is what lets the plugin recognize a resent event and store it once.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_ENVVAR = "AGENT_HARDENER_EVENT_SINK_URL"
_MAX_QUEUED = 2000
_RESEND_DELAY_S = 1.0


def make_http_event_sink() -> Callable[[str, dict[str, Any]], None] | None:
    """Return a non-blocking EventBus subscriber that POSTs events to ``$AGENT_HARDENER_EVENT_SINK_URL``, or None."""
    url = os.environ.get(_ENVVAR)
    if not url:
        return None

    events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_MAX_QUEUED)
    sink_id = uuid.uuid4().hex

    def _worker() -> None:
        client = httpx.Client(timeout=5.0)
        seq = 0
        while True:
            body = events.get()
            seq += 1
            body.update(sink_id=sink_id, seq=seq)
            if not _deliver(client, url, body):
                time.sleep(_RESEND_DELAY_S)
                if not _deliver(client, url, body):
                    logger.debug("event sink dropped %s event after a resend", body["event"])

    threading.Thread(target=_worker, daemon=True, name="agent-hardener-event-sink").start()

    def _sink(event: str, payload: dict[str, Any]) -> None:
        try:
            events.put_nowait({"event": event, "payload": payload})
        except queue.Full:
            logger.debug("event sink queue full; dropping %s event", event)

    return _sink


def _deliver(client: httpx.Client, url: str, body: dict[str, Any]) -> bool:
    """POST one event; False when it may not have arrived (a transport error or a 5xx), so resending is safe."""
    try:
        return client.post(url, json=body).status_code < 500
    except httpx.HTTPError:
        logger.debug("event sink POST to %s failed", url, exc_info=True)
        return False
