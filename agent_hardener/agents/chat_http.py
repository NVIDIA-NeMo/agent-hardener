# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-level pieces shared by every victim caller.

Request shaping and text extraction moved to :class:`~agent_hardener.endpoint.EndpointContract`, which the
attacker, the victim probe and validator replay all resolve from one config. This module is what sits in
front of it: the HTTP client those callers use (:func:`victim_client`) and the "did the victim even return
JSON?" step.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from agent_hardener.rate_limits import is_rate_limited_response

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)

#: Waits before each retry of a victim 5xx. One flaky upstream call should not end an hours-long run,
#: but a victim that is really down should end it within seconds.
SERVER_ERROR_RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 3.0)


class AsyncServerErrorRetryTransport(httpx.AsyncBaseTransport):
    """Retries a request that got a 5xx, then hands the last response to the caller as is.

    A rate-limited 503 is left alone: rate limits have their own, much longer wait
    (:mod:`agent_hardener.rate_limits`). Transport errors are not retried either, since after a timeout
    the victim may already have acted on the request.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        delays: Sequence[float] = SERVER_ERROR_RETRY_DELAYS_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._transport = transport or httpx.AsyncHTTPTransport()
        self._delays = tuple(delays)
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()  # buffer the body so the same request can be sent again
        for attempt, delay in enumerate(self._delays, start=1):
            response = await self._transport.handle_async_request(request)
            if not await _is_retryable_server_error(response):
                return response
            logger.warning(
                "%s returned HTTP %s; retry %d of %d in %.0fs",
                request.url.host,
                response.status_code,
                attempt,
                len(self._delays),
                delay,
            )
            await response.aclose()
            await self._sleep(delay)
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


async def _is_retryable_server_error(response: httpx.Response) -> bool:
    if response.status_code < 500:
        return False
    try:
        await response.aread()  # a 503 is a rate limit only by its body text
    except BaseException:
        await response.aclose()
        raise
    return not is_rate_limited_response(response)


def victim_client(*, timeout: float) -> httpx.AsyncClient:
    """The HTTP client for calls to the victim agent."""
    return httpx.AsyncClient(transport=AsyncServerErrorRetryTransport(), timeout=timeout)


def response_body_or_text(response: httpx.Response) -> Any:
    """Return the parsed JSON body, or the raw text when the response isn't JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text
