# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for surfacing rate-limit failures clearly."""

from __future__ import annotations

from typing import Any

import httpx


class RateLimitError(RuntimeError):
    """Raised when a victim or provider reports rate limiting."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        retry_after: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.retry_after = retry_after
        self.status_code = status_code


def raise_for_rate_limit_response(response: httpx.Response, *, source: str) -> None:
    """Raise a clear error when an HTTP response is a rate-limit response."""
    if not is_rate_limited_response(response):
        return
    retry_after = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset")
    detail = _response_detail(response)
    suffix = f"; retry_after={retry_after}" if retry_after else ""
    raise RateLimitError(
        f"{source} rate limited request with HTTP {response.status_code}{suffix}: {detail}",
        source=source,
        retry_after=retry_after,
        status_code=response.status_code,
    )


def rate_limit_error_from_exception(exc: BaseException, *, source: str) -> RateLimitError | None:
    """Return a RateLimitError when an exception clearly represents rate limiting."""
    if isinstance(exc, RateLimitError):
        return exc
    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response) and is_rate_limited_response(response):
        retry_after = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset")
        return RateLimitError(
            f"{source} rate limited request with HTTP {response.status_code}: {_response_detail(response)}",
            source=source,
            retry_after=retry_after,
            status_code=response.status_code,
        )
    text = str(exc)
    if _looks_like_rate_limit(text):
        return RateLimitError(f"{source} rate limited request: {text}", source=source)
    return None


def is_rate_limited_response(response: httpx.Response) -> bool:
    """Return whether an HTTP response looks like a rate-limit response."""
    if response.status_code == 429:
        return True
    if response.status_code in {403, 503}:
        return _looks_like_rate_limit(_response_detail(response))
    return False


def _response_detail(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    text = str(body)
    return text[:1000]


def _looks_like_rate_limit(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "rate limit",
            "rate_limit",
            "ratelimit",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
            "try again later",
            "retry-after",
        )
    )
