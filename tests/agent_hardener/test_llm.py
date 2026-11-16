# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx

from agent_hardener import llm
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import AsyncRateLimitRetryTransport, RateLimitRetryTransport


def test_nemotron_analysis_model_runs_without_reasoning() -> None:
    model = build_chat_model(model="nvidia/nemotron-3.5-lightning-30b-a3b", api_key="k")
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_other_models_get_no_reasoning_toggle() -> None:
    assert build_chat_model(model="openai/gpt-oss-20b", api_key="k").extra_body is None


def test_caller_extra_body_wins() -> None:
    model = build_chat_model(model="nvidia/nemotron-3.5-lightning-30b-a3b", api_key="k", extra_body={"x": 1})
    assert model.extra_body == {"x": 1}


def test_every_chat_model_waits_out_rate_limits_on_shared_clients() -> None:
    first, second = build_chat_model(api_key="k"), build_chat_model(api_key="k")
    assert isinstance(first.http_client._transport, RateLimitRetryTransport)
    assert isinstance(first.http_async_client._transport, AsyncRateLimitRetryTransport)
    assert first.http_async_client is second.http_async_client  # one connection pool, not one per model


def test_a_caller_supplied_http_client_is_kept() -> None:
    client = httpx.AsyncClient()
    assert build_chat_model(api_key="k", http_async_client=client).http_async_client is client


def test_the_retrying_transport_wraps_the_sdks_own_http_library() -> None:
    http = llm._sdk_http_module()
    sync_client, async_client = llm._rate_limited_http_clients()
    assert isinstance(sync_client._transport._transport, http.HTTPTransport)
    assert isinstance(async_client._transport._transport, http.AsyncHTTPTransport)
