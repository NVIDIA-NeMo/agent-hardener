# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any

import httpx
import pytest

from agent_hardener.runtime import event_sink

_URL = "http://plugin.invalid/events"
_WAIT = 5.0


class _FakeTransport(httpx.BaseTransport):
    """Stands in for the network under the sink's retry transport; configured on the class (the worker builds it)."""

    posts: list[tuple[str, dict[str, Any]]]
    failures_left: int
    statuses: list[int]
    gate: threading.Event | None
    entered: threading.Semaphore
    done: threading.Semaphore

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        cls = type(self)
        cls.entered.release()
        if cls.gate is not None:
            assert cls.gate.wait(_WAIT)
        cls.posts.append((str(request.url), json.loads(request.content)))
        try:
            if cls.failures_left > 0:
                cls.failures_left -= 1
                raise httpx.ConnectError("down")
            return httpx.Response(cls.statuses.pop(0) if cls.statuses else 204)
        finally:
            cls.done.release()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTransport]:
    _FakeTransport.posts = []
    _FakeTransport.failures_left = 0
    _FakeTransport.statuses = []
    _FakeTransport.gate = None
    _FakeTransport.entered = threading.Semaphore(0)
    _FakeTransport.done = threading.Semaphore(0)
    monkeypatch.setattr(httpx, "HTTPTransport", _FakeTransport)
    monkeypatch.setattr(
        event_sink, "EVENT_SINK_RETRY", replace(event_sink.EVENT_SINK_RETRY, server_error_delays_s=(0.0,))
    )
    monkeypatch.setenv(event_sink._ENVVAR, _URL)
    return _FakeTransport


def _events(client: type[_FakeTransport]) -> list[tuple[str, int]]:
    return [(body["event"], body["seq"]) for _, body in client.posts]


def _wait_for_posts(client: type[_FakeTransport], count: int) -> None:
    for _ in range(count):
        assert client.done.acquire(timeout=_WAIT)


@pytest.mark.parametrize("value", [None, ""])
def test_no_sink_without_a_url(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(event_sink._ENVVAR, raising=False)
    else:
        monkeypatch.setenv(event_sink._ENVVAR, value)

    assert event_sink.make_http_event_sink() is None


def test_events_are_posted_in_order_with_the_sinks_id_and_a_counter(client: type[_FakeTransport]) -> None:
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("round_started", {"n": 1})
    sink("round_finished", {"n": 2})
    _wait_for_posts(client, 2)

    sink_id = client.posts[0][1]["sink_id"]
    assert sink_id
    assert client.posts == [
        (_URL, {"event": "round_started", "payload": {"n": 1}, "sink_id": sink_id, "seq": 1}),
        (_URL, {"event": "round_finished", "payload": {"n": 2}, "sink_id": sink_id, "seq": 2}),
    ]


def test_each_sink_has_its_own_id(client: type[_FakeTransport]) -> None:
    # `serve` and `run` each make a sink for the same run and both count from 1; the id tells them apart.
    for sink in (event_sink.make_http_event_sink(), event_sink.make_http_event_sink()):
        assert sink is not None
        sink("e", {})
    _wait_for_posts(client, 2)

    assert [body["seq"] for _, body in client.posts] == [1, 1]
    assert len({body["sink_id"] for _, body in client.posts}) == 2


@pytest.mark.parametrize("failure", ["transport", "server"])
def test_a_failed_post_is_resent_once_with_the_same_seq(client: type[_FakeTransport], failure: str) -> None:
    if failure == "transport":
        client.failures_left = 1
    else:
        client.statuses = [503]
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("flaky", {})
    sink("next", {})
    _wait_for_posts(client, 3)

    assert _events(client) == [("flaky", 1), ("flaky", 1), ("next", 2)]


def test_an_event_failing_twice_is_dropped_and_the_worker_keeps_going(client: type[_FakeTransport]) -> None:
    client.failures_left = 2
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("fails", {})
    sink("succeeds", {})
    _wait_for_posts(client, 3)

    assert _events(client) == [("fails", 1), ("fails", 1), ("succeeds", 2)]


def test_a_client_error_is_not_resent(client: type[_FakeTransport]) -> None:
    client.statuses = [404]
    sink = event_sink.make_http_event_sink()
    assert sink is not None

    sink("unknown_run", {})
    sink("next", {})
    _wait_for_posts(client, 2)

    assert _events(client) == [("unknown_run", 1), ("next", 2)]


def test_a_full_queue_drops_events_instead_of_blocking_the_caller(
    monkeypatch: pytest.MonkeyPatch, client: type[_FakeTransport]
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
    _wait_for_posts(client, 2)

    assert [event for event, _ in _events(client)] == ["in_flight", "queued"]
