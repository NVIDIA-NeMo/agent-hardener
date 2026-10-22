# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest

from agent_hardener.runtime import event_sink

_URL = "http://plugin.invalid/events"
_WAIT = 5.0


class _FakeClient:
    """Stands in for httpx.Client; the worker thread builds it, so behaviour is configured on the class."""

    posts: list[tuple[str, dict[str, Any]]]
    failures_left: int
    gate: threading.Event | None
    entered: threading.Semaphore
    done: threading.Semaphore

    def __init__(self, *, timeout: float) -> None:
        del timeout

    def post(self, url: str, *, json: dict[str, Any]) -> None:
        cls = type(self)
        cls.entered.release()
        if cls.gate is not None:
            assert cls.gate.wait(_WAIT)
        cls.posts.append((url, json))
        try:
            if cls.failures_left > 0:
                cls.failures_left -= 1
                raise httpx.ConnectError("down")
        finally:
            cls.done.release()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.posts = []
    _FakeClient.failures_left = 0
    _FakeClient.gate = None
    _FakeClient.entered = threading.Semaphore(0)
    _FakeClient.done = threading.Semaphore(0)
    monkeypatch.setattr(event_sink.httpx, "Client", _FakeClient)
    monkeypatch.setenv(event_sink._ENVVAR, _URL)
    return _FakeClient


def _events(client: type[_FakeClient]) -> list[str]:
    return [body["event"] for _, body in client.posts]


@pytest.mark.parametrize("value", [None, ""])
def test_no_sink_without_a_url(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(event_sink._ENVVAR, raising=False)
    else:
        monkeypatch.setenv(event_sink._ENVVAR, value)

    assert event_sink.make_http_event_sink() is None


def test_events_are_posted_in_order_as_event_and_payload(client: type[_FakeClient]) -> None:
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("round_started", {"n": 1})
    sink("round_finished", {"n": 2})
    assert client.done.acquire(timeout=_WAIT)
    assert client.done.acquire(timeout=_WAIT)

    assert client.posts == [
        (_URL, {"event": "round_started", "payload": {"n": 1}}),
        (_URL, {"event": "round_finished", "payload": {"n": 2}}),
    ]


def test_a_failed_post_is_swallowed_and_the_worker_keeps_going(client: type[_FakeClient]) -> None:
    client.failures_left = 1
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("fails", {})
    sink("succeeds", {})
    assert client.done.acquire(timeout=_WAIT)
    assert client.done.acquire(timeout=_WAIT)

    assert _events(client) == ["fails", "succeeds"]


def test_a_full_queue_drops_events_instead_of_blocking_the_caller(
    monkeypatch: pytest.MonkeyPatch, client: type[_FakeClient]
) -> None:
    monkeypatch.setattr(event_sink, "_MAX_QUEUED", 1)
    client.gate = threading.Event()
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("in_flight", {})
    assert client.entered.acquire(timeout=_WAIT)  # worker took it off the queue and is stuck posting
    sink("queued", {})
    sink("dropped", {})
    client.gate.set()
    assert client.done.acquire(timeout=_WAIT)
    assert client.done.acquire(timeout=_WAIT)

    assert _events(client) == ["in_flight", "queued"]
