# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import email.utils
import logging
import time
from collections.abc import Callable

import httpx
import pytest

from agent_hardener import rate_limits
from agent_hardener.rate_limits import (
    AsyncRetryTransport,
    RateLimitError,
    RetryPolicy,
    RetryTransport,
    is_rate_limited_response,
    raise_for_rate_limit_response,
    rate_limit_error_from_exception,
    retry_delay,
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


# --- waiting out rate limits -------------------------------------------------------------------------


class _Provider:
    """A mock provider: answers each request with the next queued response, recording what it was sent."""

    def __init__(self, *responses: tuple[int, dict[str, str] | None, object] | httpx.TransportError) -> None:
        self.queue = list(responses)
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.read())
        answer = self.queue.pop(0)
        if isinstance(answer, httpx.TransportError):
            raise answer
        status, headers, body = answer
        return httpx.Response(status, json=body, headers=headers)


_LIMITED = (429, None, {"status": 429, "title": "Too Many Requests"})
_OK = (200, None, {"ok": True})


def _post_sync(provider: _Provider, *, policy: RetryPolicy | None = None) -> tuple[httpx.Response, list[float]]:
    slept: list[float] = []
    transport = RetryTransport(httpx.MockTransport(provider), policy=policy, sleep=slept.append)
    with httpx.Client(transport=transport) as client:
        return client.post("https://provider.test/v1/chat", json={"q": 1}), slept


def _post_async(provider: _Provider, *, policy: RetryPolicy | None = None) -> tuple[httpx.Response, list[float]]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    async def post() -> httpx.Response:
        transport = AsyncRetryTransport(httpx.MockTransport(provider), policy=policy, sleep=sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://provider.test/v1/chat", json={"q": 1})

    return asyncio.run(post()), slept


_Post = Callable[..., tuple[httpx.Response, list[float]]]


@pytest.mark.parametrize("post", [_post_sync, _post_async], ids=["sync", "async"])
def test_a_rate_limited_request_is_retried_until_it_succeeds(post) -> None:  # type: ignore[no-untyped-def]
    provider = _Provider(_LIMITED, _LIMITED, _OK)

    response, slept = post(provider)

    assert response.status_code == 200
    assert len(slept) == 2
    assert provider.bodies == [b'{"q":1}'] * 3  # the same body is sent again on each retry


@pytest.mark.parametrize("post", [_post_sync, _post_async], ids=["sync", "async"])
def test_other_errors_and_unlimited_503s_pass_through_untouched(post) -> None:  # type: ignore[no-untyped-def]
    for status, body in ((500, {"error": "boom"}), (503, {"error": "model offline"}), (401, {"error": "bad key"})):
        response, slept = post(_Provider((status, None, body)))
        assert (response.status_code, slept) == (status, [])


@pytest.mark.parametrize("post", [_post_sync, _post_async], ids=["sync", "async"])
def test_the_rate_limited_response_reaches_the_caller_once_the_wait_runs_out(post) -> None:  # type: ignore[no-untyped-def]
    response, slept = post(_Provider(*[_LIMITED] * 20), policy=RetryPolicy(rate_limit_budget_s=10))

    assert response.status_code == 429
    assert response.json() == {"status": 429, "title": "Too Many Requests"}  # still readable for the caller
    assert sum(slept) == pytest.approx(10)


def test_a_quota_503_is_waited_out_like_a_429() -> None:
    response, slept = _post_sync(_Provider((503, None, {"error": "quota exceeded, try again later"}), _OK))
    assert (response.status_code, len(slept)) == (200, 1)


@pytest.mark.parametrize("post", [_post_sync, _post_async], ids=["sync", "async"])
def test_a_server_error_policy_retries_a_5xx_with_its_fixed_delays_then_gives_up(post: _Post) -> None:
    policy = RetryPolicy(server_error_delays_s=(1.0, 3.0))
    provider = _Provider(*((500, None, {"error": f"boom {n}"}) for n in range(3)))

    response, slept = post(provider, policy=policy)

    assert (response.status_code, response.json(), slept) == (500, {"error": "boom 2"}, [1.0, 3.0])
    assert provider.bodies == [b'{"q":1}'] * 3


@pytest.mark.parametrize("post", [_post_sync, _post_async], ids=["sync", "async"])
def test_a_transport_error_is_resent_only_when_the_policy_opts_in(post: _Post) -> None:
    with pytest.raises(httpx.ConnectError):
        post(_Provider(httpx.ConnectError("down"), _OK), policy=RetryPolicy(server_error_delays_s=(1.0,)))

    policy = RetryPolicy(server_error_delays_s=(1.0,), retry_transport_errors=True)
    response, slept = post(_Provider(httpx.ConnectError("down"), _OK), policy=policy)
    assert (response.status_code, slept) == (200, [1.0])

    with pytest.raises(httpx.ConnectError):  # it shares the 5xx resends, so a second failure is raised
        post(_Provider((502, None, {}), httpx.ConnectError("down"), _OK), policy=policy)


def test_rate_limit_and_server_error_retries_are_counted_separately() -> None:
    policy = RetryPolicy(rate_limit_budget_s=600, server_error_delays_s=(1.0,))
    provider = _Provider(_LIMITED, (502, None, {"error": "bad gateway"}), _LIMITED, _OK)

    response, slept = _post_sync(provider, policy=policy)

    assert response.status_code == 200
    assert slept[1] == 1.0  # the one 5xx retry, between two rate-limit waits
    assert len(slept) == 3


def test_retry_after_is_honoured_in_seconds_or_as_a_date() -> None:
    assert retry_delay(5, "7") == 7
    in_ten = email.utils.formatdate(time.time() + 10, usegmt=True)
    assert 8 <= retry_delay(0, in_ten) <= 10
    assert retry_delay(0, "not a date") <= 2  # unusable header: back off as if none was sent


def test_backoff_doubles_with_jitter_up_to_the_cap() -> None:
    for attempt, ceiling in ((0, 2), (1, 4), (3, 16), (10, 60)):
        delay = retry_delay(attempt, None)
        assert ceiling / 2 <= delay <= ceiling


@pytest.mark.parametrize(("raw", "expected"), [("", 600.0), ("30", 30.0), ("0", 0.0), ("-5", 0.0), ("soon", 600.0)])
def test_max_wait_reads_the_env(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv(rate_limits.RATE_LIMIT_MAX_WAIT_ENV, raw)
    assert rate_limits.rate_limit_max_wait() == expected


def test_a_wait_starting_is_one_console_warning_per_host_per_minute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(rate_limits, "_last_wait_warning", {})
    clock = [1000.0]
    monkeypatch.setattr(rate_limits.time, "monotonic", lambda: clock[0])

    with caplog.at_level(logging.INFO, logger="agent_hardener.rate_limits"):
        _post_sync(_Provider(_LIMITED, _LIMITED, _OK))  # one wait, two retries
        _post_sync(_Provider(_LIMITED, _OK))  # a parallel call throttled at the same time
        clock[0] += 61
        _post_sync(_Provider(_LIMITED, _OK))  # a minute later: worth saying again

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["provider.test is rate limiting requests (HTTP 429); waiting up to 600s before retrying"] * 2
    assert sum(r.levelno == logging.INFO for r in caplog.records) == 4  # every retry, in the log only
