# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP replay helpers for validator agents.

The request shape and response extraction live in :class:`~agent_hardener.endpoint.EndpointContract`
so replay sends byte-for-byte what the attacker sent. When the two drift, a "blocked" verdict stops
being evidence about the guardrail and starts being evidence about the two payload builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from agent_hardener.agents.chat_http import response_body_or_text
from agent_hardener.endpoint import EndpointContract
from agent_hardener.rate_limits import raise_for_rate_limit_response

if TYPE_CHECKING:
    from collections.abc import Callable


class ReplayHTTPConfig(Protocol):
    """Config shape required to replay one prompt against a victim endpoint."""

    replay_url: str
    replay_mode: str
    model: str
    input_field: str
    response_json_path: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class ReplayResult:
    """Normalized victim replay result."""

    ok: bool
    text: str
    status_code: int | None = None
    body: Any = None


async def replay_prompt(
    prompt: str,
    config: ReplayHTTPConfig,
    client: httpx.AsyncClient,
    *,
    rate_limit_source: str,
    error_excerpt: Callable[[str], str],
    session_id: str | None = None,
) -> ReplayResult:
    """Replay a prompt against a victim endpoint and normalize the response."""
    if not config.replay_url:
        msg = "validator replay_url is not configured and target.base_url is empty"
        raise ValueError(msg)

    contract = contract_from(config)
    payload = contract.request_payload(prompt)

    response = await client.post(
        contract.url, json=payload, timeout=contract.timeout_seconds, headers=contract.headers(session_id)
    )
    raise_for_rate_limit_response(response, source=rate_limit_source)
    body = response_body_or_text(response)
    text = contract.extract_text(body)
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"victim replay returned HTTP {response.status_code}: {error_excerpt(str(body))}",
            request=response.request,
            response=response,
        )
    return ReplayResult(ok=True, text=text, status_code=response.status_code, body=body)


def contract_from(config: ReplayHTTPConfig) -> EndpointContract:
    """Adapt a validator's replay config to the shared endpoint contract."""
    return EndpointContract(
        url=config.replay_url,
        mode="openai_chat" if config.replay_mode == "openai_chat" else "field",
        model=config.model,
        input_field=config.input_field,
        response_json_path=config.response_json_path,
        timeout_seconds=config.timeout_seconds,
    )
