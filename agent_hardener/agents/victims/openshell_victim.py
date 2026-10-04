# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell-backed victim implementation."""

from __future__ import annotations

from typing import Any

import httpx

from agent_hardener.agents.chat_http import response_body_or_text
from agent_hardener.endpoint import EndpointContract
from agent_hardener.errors import VictimUnavailableError
from agent_hardener.models import AgentConfig, AgentRunInput, VictimResult
from agent_hardener.rate_limits import raise_for_rate_limit_response, rate_limit_error_from_exception


async def run(request: AgentRunInput, agent: AgentConfig) -> VictimResult:
    """Call the configured OpenShell victim API."""
    url = str(agent.config.get("url") or request.target.base_url or "")
    if not url:
        return VictimResult(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            ok=False,
            summary="OpenShell victim URL is not configured",
            error="missing victim URL",
        )

    contract = contract_for(url, agent)
    payload = contract.request_payload(_input_message(request, agent))
    # One session per round so the victim's ATOF scopes can be traced back to this invocation
    # rather than correlated by arrival order, which concurrency makes meaningless.
    headers = contract.headers(f"{request.round_id}-{agent.agent_id}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=contract.timeout_seconds, headers=headers)
        raise_for_rate_limit_response(response, source=f"victim {agent.name}")
        body = response_body_or_text(response)
        return VictimResult(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            ok=response.status_code < 400,
            summary=f"OpenShell victim returned HTTP {response.status_code}",
            observations={"status_code": response.status_code, "body": body, "request": payload},
            error=None if response.status_code < 400 else str(body),
        )
    except httpx.HTTPError as exc:
        rate_limit = rate_limit_error_from_exception(exc, source=f"victim {agent.name}")
        if rate_limit is not None:
            raise rate_limit from exc
        # A transport-level failure here (connection refused, server disconnected, timeout) means the
        # victim/sandbox is DOWN, not that the agent produced a bad answer. Abort loudly instead of
        # returning a soft failure that lets the run replay against a dead victim and report "0 blocked".
        msg = (
            f"OpenShell victim {agent.name} is unreachable at {url} ({exc}). The sandbox or victim agent "
            "likely crashed — commonly a malformed workflow/policy it failed to load. Check the victim logs."
        )
        raise VictimUnavailableError(msg) from exc


def contract_for(url: str, agent: AgentConfig) -> EndpointContract:
    """Resolve how this victim's endpoint is shaped, defaulting to OpenAI chat completions."""
    return EndpointContract(
        url=url,
        mode="field" if agent.config.get("input_field") else "openai_chat",
        model=str(agent.config.get("model") or "openshell-victim"),
        input_field=str(agent.config.get("input_field") or "input"),
        response_json_path=agent.config.get("response_json_path"),
        timeout_seconds=agent.timeout_seconds,
    )


def _input_message(request: AgentRunInput, agent: AgentConfig) -> str:
    return str(
        agent.config.get("input_message")
        or request.context.get("victim_input_message")
        or "Run the configured victim scenario and report observations."
    )


def build_payload(request: AgentRunInput, agent: AgentConfig) -> dict[str, Any]:
    """Build the victim request body without sending it."""
    return contract_for("", agent).request_payload(_input_message(request, agent))
