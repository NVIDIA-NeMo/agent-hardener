# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for surfacing rate-limit failures clearly, and for waiting them out.

A provider's rate limit is a reason to wait, not to fail: :class:`RetryTransport` and
:class:`AsyncRetryTransport` retry a rate-limited request with exponential backoff before the caller ever
sees the response, and a :class:`RetryPolicy` can have them retry a brief 5xx too. This is Agent
Hardener's one HTTP retry policy. Only when the total wait runs out does the rate-limited response reach
the caller, where the helpers below turn it into a :class:`RateLimitError` that names the source.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, replace
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
# A wait can last up to ``rate_limit_max_wait()`` (10 min by default), so its start is a console warning — or
# the run looks stuck. At most one per host per interval: parallel calls throttled together share one line.
_WAIT_WARNING_INTERVAL_SECONDS = 60.0
_last_wait_warning: dict[str, float] = {}
_wait_warning_lock = threading.Lock()


def _warn_wait_started(host: str, status: int, max_wait: float) -> None:
    """Say once (per host per interval) that calls to ``host`` are waiting out a rate limit."""
    now = time.monotonic()
    with _wait_warning_lock:
        last = _last_wait_warning.get(host)
        if last is not None and now - last < _WAIT_WARNING_INTERVAL_SECONDS:
            return
        _last_wait_warning[host] = now
    logger.warning(
        "%s is rate limiting requests (HTTP %s); waiting up to %.0fs before retrying", host, status, max_wait
    )


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


@dataclass
class RetryState:
    """One request's retries so far: kept by a transport, updated by :meth:`RetryPolicy.next_delay`."""

    rate_limit_retries: int = 0
    rate_limit_waited: float = 0.0
    server_error_retries: int = 0


@dataclass(frozen=True)
class RetryPolicy:
    """Which failed responses a retry transport sends again, and how long it waits before each try.

    A rate limit (a 429, or a 403/503 whose body says so) is waited out with :func:`retry_delay` backoff for
    up to ``rate_limit_budget_s`` seconds in total; ``None`` means :func:`rate_limit_max_wait`. Any other
    5xx is retried once per entry of ``server_error_delays_s``, after that many seconds. Transport errors
    are never retried: after a timeout the server may already have acted on the request.
    """

    rate_limit_budget_s: float | None = None
    server_error_delays_s: tuple[float, ...] = ()

    def resolved(self) -> RetryPolicy:
        """This policy with an unset rate-limit budget read from the environment now, once."""
        return replace(self, rate_limit_budget_s=self._rate_limit_budget())

    def _rate_limit_budget(self) -> float:
        return rate_limit_max_wait() if self.rate_limit_budget_s is None else self.rate_limit_budget_s

    def may_retry(self, response: httpx.Response) -> bool:
        """Whether the response needs its body read to decide (403/503 are rate limits only by their text)."""
        return response.status_code in {403, 429, 503} or (
            bool(self.server_error_delays_s) and response.status_code >= 500
        )

    def next_delay(self, request: httpx.Request, response: httpx.Response, state: RetryState) -> float | None:
        """The wait before the next try (recorded in ``state``), or ``None`` to hand the response to the caller."""
        if is_rate_limited_response(response):
            return self._rate_limit_delay(request, response, state)
        if response.status_code >= 500:
            return self._server_error_delay(request, response, state)
        return None

    def _rate_limit_delay(self, request: httpx.Request, response: httpx.Response, state: RetryState) -> float | None:
        budget = self._rate_limit_budget()
        waited = state.rate_limit_waited
        retry_after = response.headers.get("retry-after")
        delay = min(retry_delay(state.rate_limit_retries, retry_after), budget - waited)
        if delay <= 0:
            logger.warning(
                "%s still rate limited (HTTP %s) after waiting %.0fs; giving up",
                request.url.host,
                response.status_code,
                waited,
            )
            return None
        if waited == 0:
            _warn_wait_started(request.url.host, response.status_code, budget)
        logger.info(
            "%s rate limited the request (HTTP %s); retry %d in %.1fs (waited %.0fs of %.0fs)",
            request.url.host,
            response.status_code,
            state.rate_limit_retries + 1,
            delay,
            waited,
            budget,
        )
        state.rate_limit_retries += 1
        state.rate_limit_waited += delay
        return delay

    def _server_error_delay(self, request: httpx.Request, response: httpx.Response, state: RetryState) -> float | None:
        if state.server_error_retries >= len(self.server_error_delays_s):
            return None
        delay = self.server_error_delays_s[state.server_error_retries]
        state.server_error_retries += 1
        logger.warning(
            "%s returned HTTP %s; retry %d of %d in %.0fs",
            request.url.host,
            response.status_code,
            state.server_error_retries,
            len(self.server_error_delays_s),
            delay,
        )
        return delay


class RetryTransport(httpx.BaseTransport):
    """An httpx transport that retries a failed request as its :class:`RetryPolicy` says.

    The default policy waits out rate limits only. A response the policy does not retry passes through
    untouched, as does the last failed one once the policy gives up, so the caller's own error handling
    still runs. It only calls methods httpx and httpx2 share, so it can wrap a transport of either: pass
    the one from the library the client is built on (see :mod:`agent_hardener.llm`).
    """

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        *,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport or httpx.HTTPTransport()
        self._policy = (policy or RetryPolicy()).resolved()
        self._sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()  # buffer the body so the same request can be sent again
        state = RetryState()
        while True:
            response = self._transport.handle_request(request)
            if not self._policy.may_retry(response):
                return response
            try:
                response.read()
            except BaseException:
                response.close()
                raise
            delay = self._policy.next_delay(request, response, state)
            if delay is None:
                return response
            response.close()
            self._sleep(delay)

    def close(self) -> None:
        self._transport.close()


class AsyncRetryTransport(httpx.AsyncBaseTransport):
    """The async counterpart of :class:`RetryTransport`."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport or httpx.AsyncHTTPTransport()
        self._policy = (policy or RetryPolicy()).resolved()
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()  # buffer the body so the same request can be sent again
        state = RetryState()
        while True:
            response = await self._transport.handle_async_request(request)
            if not self._policy.may_retry(response):
                return response
            try:
                await response.aread()
            except BaseException:
                await response.aclose()
                raise
            delay = self._policy.next_delay(request, response, state)
            if delay is None:
                return response
            await response.aclose()
            await self._sleep(delay)

    async def aclose(self) -> None:
        await self._transport.aclose()
