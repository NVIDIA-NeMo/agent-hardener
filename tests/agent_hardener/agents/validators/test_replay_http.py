# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``replay_prompt`` sends the attacker's exact request shape and fails loudly on victim errors."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from agent_hardener.agents.validators.replay_http import ReplayResult, contract_from, replay_prompt
from agent_hardener.endpoint import SESSION_ID_HEADER
from agent_hardener.rate_limits import RateLimitError

pytestmark = pytest.mark.unit


@dataclass
class _Config:
    replay_url: str = "http://victim.test/v1/chat/completions"
    replay_mode: str = "openai_chat"
    model: str = "victim-model"
    input_field: str = "input_message"
    response_json_path: str | None = None
    timeout_seconds: float = 5.0


def _replay(config: _Config, handler: Any, prompt: str = "hello", **kwargs: Any) -> ReplayResult:
    async def _go() -> ReplayResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_prompt(
                prompt, config, client, rate_limit_source="victim", error_excerpt=lambda text: text[:20], **kwargs
            )

    return asyncio.run(_go())


def test_openai_chat_replay_posts_chat_payload_and_extracts_assistant_text() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]})

    result = _replay(_Config(), handler, session_id="sess-1")

    assert result.ok is True
    assert result.text == "hi there"
    assert result.status_code == 200
    assert seen["url"] == "http://victim.test/v1/chat/completions"
    assert seen["json"] == {"model": "victim-model", "messages": [{"role": "user", "content": "hello"}]}
    assert seen["headers"][SESSION_ID_HEADER] == "sess-1"


def test_json_mode_uses_input_field_and_response_json_path() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, json={"data": {"answer": "42"}})

    config = _Config(replay_mode="json", response_json_path="$.data.answer")
    result = _replay(config, handler)

    assert seen["json"] == {"input_message": "hello"}
    assert SESSION_ID_HEADER not in seen["headers"]
    assert result.text == "42"


def test_non_json_success_body_is_returned_as_text() -> None:
    result = _replay(_Config(), lambda _request: httpx.Response(200, text="plain victim answer"))

    assert result.text == "plain victim answer"
    assert result.body == "plain victim answer"


def test_empty_replay_url_is_rejected_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    with pytest.raises(ValueError, match="replay_url is not configured"):
        _replay(_Config(replay_url=""), handler)


def test_http_error_status_raises_with_truncated_excerpt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "x" * 200})

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _replay(_Config(), handler)

    message = str(excinfo.value)
    assert "HTTP 500" in message
    assert "x" * 100 not in message  # excerpt callback bounds what leaks into the error


def test_rate_limited_response_raises_rate_limit_error_not_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})

    with pytest.raises(RateLimitError) as excinfo:
        _replay(_Config(), handler)

    assert excinfo.value.source == "victim"
    assert excinfo.value.retry_after == "7"
    assert excinfo.value.status_code == 429


def test_contract_from_maps_unknown_replay_mode_to_field_mode() -> None:
    assert contract_from(_Config(replay_mode="openai_chat")).mode == "openai_chat"
    assert contract_from(_Config(replay_mode="json")).mode == "field"
    assert contract_from(_Config(replay_mode="anything-else")).mode == "field"
