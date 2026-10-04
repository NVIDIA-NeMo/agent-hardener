# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest

from agent_hardener.rate_limits import (
    RateLimitError,
    is_rate_limited_response,
    raise_for_rate_limit_response,
    rate_limit_error_from_exception,
)


def _response(status_code: int, body: object, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body,
        headers=headers,
        request=httpx.Request("POST", "http://victim.local/run"),
    )


def test_raise_for_rate_limit_response_includes_retry_after() -> None:
    response = _response(429, {"error": "too many requests"}, {"retry-after": "12"})

    with pytest.raises(RateLimitError) as exc_info:
        raise_for_rate_limit_response(response, source="victim replay")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == "12"
    assert exc_info.value.source == "victim replay"
    assert "HTTP 429" in str(exc_info.value)


def test_rate_limit_detection_handles_provider_quota_403() -> None:
    response = _response(403, {"error": "quota exceeded for this model"})

    assert is_rate_limited_response(response) is True


def test_rate_limit_detection_ignores_normal_server_error() -> None:
    response = _response(500, {"error": "backend failed"})

    assert is_rate_limited_response(response) is False
    raise_for_rate_limit_response(response, source="victim replay")


def test_rate_limit_error_from_http_exception() -> None:
    response = _response(429, {"error": "rate limit"}, {"x-ratelimit-reset": "123"})
    exc = httpx.HTTPStatusError("failed", request=response.request, response=response)

    rate_limit = rate_limit_error_from_exception(exc, source="agent service")

    assert rate_limit is not None
    assert rate_limit.status_code == 429
    assert rate_limit.retry_after == "123"
    assert "agent service rate limited request" in str(rate_limit)


def test_rate_limit_error_from_text_exception() -> None:
    rate_limit = rate_limit_error_from_exception(RuntimeError("resource exhausted: rate limit"), source="llm")

    assert rate_limit is not None
    assert rate_limit.source == "llm"
