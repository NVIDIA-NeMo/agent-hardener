# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded attacks against the agent to obtain their execution traces.

A ``--replay`` run feeds recorded garak hits to the defenders without contacting the agent, so their traces
must come from the recording made with the hitlog (``attack_traces.json``). When an attack has no recorded
trace, the traces guardrails defender cannot write a rail for it. :func:`replay_attacks_for_traces` closes
that gap: it sends each such attack's prompt to the live agent once, before defense, and matches the trace
the agent produced, exactly as a live run collects its attackers' traces (:mod:`.matching`).

Only direct attacks are replayed. An indirect hit (``triggers.trigger_query``) needs its injection planted
where the agent will read it, which a plain resend does not do, so it is skipped and logged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.endpoint import DEFAULT_VICTIM_CONCURRENCY
from agent_hardener.traces.attacks import RecordedAttackTrace, attack_key, record_prompt
from agent_hardener.traces.matching import find_attack_trace, trace_clock, traces_since

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agent_hardener.endpoint import EndpointContract
    from agent_hardener.models import AttackRecord
    from agent_hardener.models.traces import AgentTrace
    from agent_hardener.traces.provider import TraceProvider

logger = logging.getLogger(__name__)


async def replay_attacks_for_traces(
    attacks: list[AttackRecord],
    *,
    recorded: Mapping[str, AgentTrace],
    provider: TraceProvider,
    contract: EndpointContract,
    concurrency: int = DEFAULT_VICTIM_CONCURRENCY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[RecordedAttackTrace]:
    """Send every direct attack with no recorded trace to the agent, and return the traces it produced.

    ``recorded`` is keyed like the defenders' lookup (``attack_key(record_prompt(record))``); an attack found
    there is not sent. Each returned entry is keyed by the attack's own prompt, so the defenders find it.
    An attack whose send fails, or whose trace cannot be found, is left out; nothing here raises for one
    attack. ``transport`` replaces the HTTP transport (tests).
    """
    pending = _attacks_to_replay(attacks, recorded)
    if not pending:
        return []
    anchor = trace_clock(await asyncio.to_thread(provider.get_traces))
    async with httpx.AsyncClient(transport=transport) as client:
        answers = await gather_limited_ordered(pending, concurrency, lambda prompt: _send(client, contract, prompt))
    traces = traces_since(await asyncio.to_thread(provider.get_traces), anchor)
    claimed: set[str] = set()
    replayed: list[RecordedAttackTrace] = []
    for prompt, answer in zip(pending, answers, strict=True):
        if not isinstance(answer, str):
            continue
        trace = find_attack_trace(traces, prompt, answer, claimed=claimed)
        if trace is None:
            logger.warning("no trace found for a replayed attack: %.80s", prompt)
            continue
        replayed.append(RecordedAttackTrace(prompt=prompt, response=answer, trace=trace))
    logger.info("replayed %d attack(s) for their traces; %d traced", len(pending), len(replayed))
    return replayed


def _attacks_to_replay(attacks: list[AttackRecord], recorded: Mapping[str, AgentTrace]) -> list[str]:
    """The prompts to send: one per direct record that has a prompt and no recorded trace."""
    prompts: list[str] = []
    skipped_indirect = 0
    for attack in attacks:
        for record in attack.records:
            if (record.get("triggers") or {}).get("trigger_query"):
                skipped_indirect += 1
                continue
            prompt = record_prompt(record)
            if prompt and attack_key(prompt) not in recorded:
                prompts.append(prompt)
    if skipped_indirect:
        logger.info("not replaying %d indirect attack(s): their injection is not re-planted", skipped_indirect)
    return prompts


async def _send(client: httpx.AsyncClient, contract: EndpointContract, prompt: str) -> str | None:
    """Send one prompt to the agent and return its answer, or ``None`` (logged) when the call fails."""
    try:
        response = await client.post(
            contract.url,
            json=contract.request_payload(prompt),
            headers=contract.headers(),
            timeout=contract.timeout_seconds,
        )
        response.raise_for_status()
        body: Any = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("could not replay an attack to the agent (%s): %.80s", exc, prompt)
        return None
    return contract.extract_text(body)
