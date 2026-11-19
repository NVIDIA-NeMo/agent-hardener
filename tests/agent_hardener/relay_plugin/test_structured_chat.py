# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``with_structured_output`` on the project's chat model reads replies the OpenAI SDK's ``.parse()`` rejects."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import openai
import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agent_hardener.relay_plugin.json_reply import JsonReplyError
from agent_hardener.relay_plugin.structured_chat import RobustChatOpenAI

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    ok: bool
    reason: str


def _model(replies: list[Any], requests: list[dict[str, Any]]) -> RobustChatOpenAI:
    """A model whose endpoint answers each request with the next reply (a str, or an int status)."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "response_format json_schema is not supported"}})
        reply, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        message = {"role": "assistant", "content": reply}
        body = {
            "id": "1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    return RobustChatOpenAI(
        model="m",
        api_key="k",
        base_url="http://llm.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
    )


def test_a_fenced_reply_with_reasoning_reads_in_a_chain() -> None:
    requests: list[dict[str, Any]] = []
    model = _model(['<think>hm</think>\n```json\n{"ok": true, "reason": "fine"}\n```'], requests)
    chain = ChatPromptTemplate.from_template("{prompt}") | model.with_structured_output(Verdict)
    assert chain.invoke({"prompt": "judge"}) == Verdict(ok=True, reason="fine")
    assert requests[0]["response_format"]["json_schema"]["name"] == "Verdict"


def test_an_unreadable_reply_is_asked_for_again_with_the_error() -> None:
    requests: list[dict[str, Any]] = []
    model = _model([("I think it is fine.", "length"), '{"ok": true, "reason": "fine"}'], requests)
    result = asyncio.run(model.with_structured_output(Verdict).ainvoke("judge"))
    assert result == Verdict(ok=True, reason="fine")
    retry = requests[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": "I think it is fine."}
    assert "no JSON found" in retry[-1]["content"]
    assert "cut off at the token limit" in retry[-1]["content"]


def test_a_reply_cut_off_at_max_tokens_keeps_what_is_complete() -> None:
    class Rows(BaseModel):
        rows: list[Verdict]

    reply = '{"rows": [{"ok": true, "reason": "a"}, {"ok": false, "reas'
    assert _model([(reply, "length")], []).with_structured_output(Rows).invoke("go") == Rows(
        rows=[Verdict(ok=True, reason="a")]
    )


def test_no_retries_means_one_call() -> None:
    requests: list[dict[str, Any]] = []
    model = _model(["nope"], requests)
    model.structured_retries = 0
    with pytest.raises(JsonReplyError, match="nope"):
        model.with_structured_output(Verdict).invoke("judge")
    assert len(requests) == 1


def test_attempts_are_bounded_and_the_last_error_carries_the_reply() -> None:
    model = _model(["nope", "still nope"], [])
    with pytest.raises(JsonReplyError, match="still nope"):
        model.with_structured_output(Verdict).invoke("judge")


@pytest.mark.parametrize("use_async", [False, True])
def test_an_endpoint_rejecting_the_schema_is_asked_without_it(use_async: bool) -> None:
    requests: list[dict[str, Any]] = []
    model = _model([400, 'Sure: {"ok": false, "reason": "no"}'], requests)
    runnable = model.with_structured_output(Verdict)
    result = asyncio.run(runnable.ainvoke("judge")) if use_async else runnable.invoke("judge")
    assert result == Verdict(ok=False, reason="no")
    assert "response_format" not in requests[1]
    assert "JSON schema" in requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("use_async", [False, True])
def test_any_other_bad_request_is_raised(use_async: bool) -> None:
    error = {"error": {"message": "context too long"}}
    transport = httpx.MockTransport(lambda _request: httpx.Response(400, json=error))
    model = RobustChatOpenAI(
        model="m",
        api_key="k",
        base_url="http://llm.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
    )
    runnable = model.with_structured_output(Verdict)
    with pytest.raises(openai.BadRequestError, match="context too long"):
        asyncio.run(runnable.ainvoke("judge")) if use_async else runnable.invoke("judge")


def test_other_structured_output_modes_are_chatopenai_own() -> None:
    model = _model([], [])
    assert type(model.with_structured_output(Verdict, include_raw=True)).__name__ != "RunnableLambda"
    assert (
        type(model.with_structured_output({"title": "x", "type": "object", "properties": {}})).__name__
        != "RunnableLambda"
    )
