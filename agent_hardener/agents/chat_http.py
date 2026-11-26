# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-level pieces shared by every victim caller.

Request shaping and text extraction moved to :class:`~agent_hardener.endpoint.EndpointContract`, which the
attacker, the victim probe and validator replay all resolve from one config. This module is what sits in
front of it: the HTTP client those callers use (:func:`victim_client`) and the "did the victim even return
JSON?" step.
"""

from __future__ import annotations

from typing import Any

import httpx

from agent_hardener.rate_limits import AsyncRetryTransport, RetryPolicy

#: A victim's rate limit (often its model's 429, passed through) is waited out like any other. A victim 5xx
#: gets a couple of quick retries: one flaky upstream call should not end an hours-long run, but a victim
#: that is really down should end it within seconds.
VICTIM_RETRY = RetryPolicy(server_error_delays_s=(1.0, 3.0))


def victim_client(*, timeout: float, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """The HTTP client for every call to the victim agent; ``transport`` replaces the network one (tests)."""
    return httpx.AsyncClient(transport=AsyncRetryTransport(transport, policy=VICTIM_RETRY), timeout=timeout)


def response_body_or_text(response: httpx.Response) -> Any:
    """Return the parsed JSON body, or the raw text when the response isn't JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text
