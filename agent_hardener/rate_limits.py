# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for surfacing rate-limit failures clearly, and for waiting them out.

A provider's rate limit is a reason to wait, not to fail: :class:`RateLimitRetryTransport` and
:class:`AsyncRateLimitRetryTransport` retry a rate-limited request with exponential backoff before the
caller ever sees the response. Only when the total wait runs out does the rate-limited response reach the
caller, where the helpers below turn it into a :class:`RateLimitError` that names the source.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import os
import random
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

#: Longest total time one request waits out rate limits before the rate-limited response is returned.
#: Overridable with ``AGENT_HARDENER_RATE_LIMIT_MAX_WAIT_S``; ``0`` turns retrying off.
DEFAULT_RATE_LIMIT_MAX_WAIT_SECONDS = 600.0
RATE_LIMIT_MAX_WAIT_ENV = "AGENT_HARDENER_RATE_LIMIT_MAX_WAIT_S"
#: The first wait, doubled on each retry up to the cap. A per-minute limit (build.nvidia.com sends no
#: ``Retry-After``) clears within a minute, so the cap keeps a retry inside the next window.
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_CAP_SECONDS = 60.0


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


def rate_limit_max_wait() -> float:
    """The total wait for one request: ``AGENT_HARDENER_RATE_LIMIT_MAX_WAIT_S``, else the default."""
    raw = os.environ.get(RATE_LIMIT_MAX_WAIT_ENV, "").strip()
    if not raw:
        return DEFAULT_RATE_LIMIT_MAX_WAIT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("ignoring %s=%r: not a number of seconds", RATE_LIMIT_MAX_WAIT_ENV, raw)
        return DEFAULT_RATE_LIMIT_MAX_WAIT_SECONDS


def retry_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    The provider's ``Retry-After`` (seconds or an HTTP date) wins when it is usable. Otherwise the wait
    doubles from :data:`_BACKOFF_BASE_SECONDS` up to :data:`_BACKOFF_CAP_SECONDS`, half of it random so
    the many requests a burst rate-limits at once do not all retry in the same instant.
    """
    if retry_after:
        seconds = _parse_retry_after(retry_after)
        if seconds is not None:
            return seconds
    ceiling = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2.0**attempt)
    return ceiling / 2 + random.uniform(0, ceiling / 2)  # noqa: S311 - jitter, not cryptography


def _parse_retry_after(value: str) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class _RetryPolicy:
    """What both transports share: which responses to retry, how long to wait, when to give up."""

    def __init__(self, *, max_wait: float | None) -> None:
        self.max_wait = rate_limit_max_wait() if max_wait is None else max_wait

    @staticmethod
    def may_be_rate_limited(response: httpx.Response) -> bool:
        """Whether the response needs its body read to decide (403/503 are rate limits only by their text)."""
        return response.status_code in {403, 429, 503}

    def next_delay(self, request: httpx.Request, response: httpx.Response, attempt: int, waited: float) -> float | None:
        """The wait before the next try, or ``None`` when the response should go to the caller as is."""
        if not is_rate_limited_response(response):
            return None
        retry_after = response.headers.get("retry-after")
        delay = min(retry_delay(attempt, retry_after), self.max_wait - waited)
        if delay <= 0:
            logger.warning(
                "%s still rate limited (HTTP %s) after waiting %.0fs; giving up",
                request.url.host,
                response.status_code,
                waited,
            )
            return None
        logger.info(
            "%s rate limited the request (HTTP %s); retry %d in %.1fs (waited %.0fs of %.0fs)",
            request.url.host,
            response.status_code,
            attempt + 1,
            delay,
            waited,
            self.max_wait,
        )
        return delay


class RateLimitRetryTransport(httpx.BaseTransport):
    """An httpx transport that waits out rate limits: it retries a rate-limited request with backoff.

    A response that is not a rate limit (including every other error) passes through untouched, as
    does the last rate-limited one once ``max_wait`` seconds are spent, so the caller's own error
    handling still runs. It only calls methods httpx and httpx2 share, so it can wrap a transport of
    either: pass the one from the library the client is built on (see :mod:`agent_hardener.llm`).
    """

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        *,
        max_wait: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport or httpx.HTTPTransport()
        self._policy = _RetryPolicy(max_wait=max_wait)
        self._sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()  # buffer the body so the same request can be sent again
        attempt, waited = 0, 0.0
        while True:
            response = self._transport.handle_request(request)
            if not self._policy.may_be_rate_limited(response):
                return response
            response.read()
            delay = self._policy.next_delay(request, response, attempt, waited)
            if delay is None:
                return response
            response.close()
            self._sleep(delay)
            attempt, waited = attempt + 1, waited + delay

    def close(self) -> None:
        self._transport.close()


class AsyncRateLimitRetryTransport(httpx.AsyncBaseTransport):
    """The async counterpart of :class:`RateLimitRetryTransport`."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_wait: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport or httpx.AsyncHTTPTransport()
        self._policy = _RetryPolicy(max_wait=max_wait)
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()  # buffer the body so the same request can be sent again
        attempt, waited = 0, 0.0
        while True:
            response = await self._transport.handle_async_request(request)
            if not self._policy.may_be_rate_limited(response):
                return response
            await response.aread()
            delay = self._policy.next_delay(request, response, attempt, waited)
            if delay is None:
                return response
            await response.aclose()
            await self._sleep(delay)
            attempt, waited = attempt + 1, waited + delay

    async def aclose(self) -> None:
        await self._transport.aclose()
