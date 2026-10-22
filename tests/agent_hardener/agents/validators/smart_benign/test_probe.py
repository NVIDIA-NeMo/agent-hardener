# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the api_prober subgraph: bounded meta-prompts against the victim's chat endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.nodes import probe as pb
from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.state import ApiProberState

if TYPE_CHECKING:
    from collections.abc import Callable

_REAL_CLIENT = httpx.AsyncClient


def _state(**overrides: Any) -> ApiProberState:
    base = {
        "target_url": "http://victim.test/v1/chat/completions",
        "target_name": "victim",
        "probe_model": "victim-model",
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return ApiProberState(**{**base, **overrides})


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class _Victim:
    """Installs a MockTransport-backed AsyncClient and records the requests it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        def client(*, timeout: float) -> httpx.AsyncClient:
            return _REAL_CLIENT(timeout=timeout, transport=httpx.MockTransport(respond))

        monkeypatch.setattr(pb.httpx, "AsyncClient", client)

    def questions(self) -> list[str]:
        return [json.loads(r.content)["messages"][0]["content"] for r in self.requests]


def _probe(state: ApiProberState) -> dict[str, object]:
    return asyncio.run(pb.probe(state))


def _responses(delta: dict[str, object]) -> list[tuple[str, str]]:
    return cast("list[tuple[str, str]]", delta["probe_responses"])


def _errors(delta: dict[str, object]) -> list[str]:
    return cast("list[str]", delta["probe_errors"])


@pytest.mark.parametrize(
    ("overrides", "note"),
    [({"target_url": ""}, "no api_endpoint configured"), ({"max_probes": 0}, "probe budget is 0")],
)
def test_probing_is_skipped_without_an_endpoint_or_a_budget(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], note: str
) -> None:
    victim = _Victim(monkeypatch, lambda _r: pytest.fail("must not call the victim"))

    delta = _probe(_state(**overrides))

    assert _responses(delta) == []
    assert _errors(delta) == []
    assert note in str(delta["source_note"])
    assert victim.requests == []


def test_every_probe_is_sent_as_a_chat_completion_and_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    victim = _Victim(monkeypatch, lambda _request: _reply("ok"))

    delta = _probe(_state())

    assert len(victim.requests) == len(pb.PROBES)
    assert {r.url for r in victim.requests} == {httpx.URL("http://victim.test/v1/chat/completions")}
    assert all(json.loads(r.content)["model"] == "victim-model" for r in victim.requests)
    assert sorted(victim.questions()) == sorted(pb.PROBES)
    assert sorted(q for q, _ in _responses(delta)) == sorted(pb.PROBES)
    assert delta["probe_errors"] == []


def test_the_probe_budget_caps_the_number_of_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    victim = _Victim(monkeypatch, lambda _r: _reply("ok"))

    delta = _probe(_state(max_probes=2))

    assert sorted(victim.questions()) == sorted(pb.PROBES[:2])
    assert len(_responses(delta)) == 2


def test_responses_stay_paired_with_their_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["messages"][0]["content"]
        return _reply(f"re: {question}")

    _Victim(monkeypatch, handler)

    delta = _probe(_state())

    assert all(answer == f"re: {question}" for question, answer in _responses(delta))


@pytest.mark.parametrize(
    "failing",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={"unexpected": "shape"}),
    ],
    ids=["http-500", "rate-limited", "malformed-body"],
)
def test_one_failing_probe_is_recorded_and_the_rest_still_count(
    monkeypatch: pytest.MonkeyPatch, failing: httpx.Response
) -> None:
    first = pb.PROBES[0]

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][0]["content"] == first:
            return failing
        return _reply("fine")

    _Victim(monkeypatch, handler)

    delta = _probe(_state())

    assert len(_responses(delta)) == len(pb.PROBES) - 1
    (error,) = _errors(delta)
    assert error.startswith(f"probe failed [{first[:40]}...]")


def test_a_dead_endpoint_fails_every_probe_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _Victim(monkeypatch, refuse)

    delta = _probe(_state())

    assert delta["probe_responses"] == []
    assert len(_errors(delta)) == len(pb.PROBES)
    assert all("connection refused" in e for e in _errors(delta))
