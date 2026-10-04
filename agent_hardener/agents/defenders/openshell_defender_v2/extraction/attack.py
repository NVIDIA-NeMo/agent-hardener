# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the attack's request tuples and identify the single one to cut.

Prefers structured records from ``DefenderInput.context`` when present.  Falls back to
prompt-text extraction: the attack prompt always contains the target URL and operation
explicitly, whereas the agent response may be opaque (e.g. an echo-service reply).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import load_config
from ..errors import ExtractionError
from ..llm.client import complete_structured
from ..models import HarmCertificate, RequestTuple
from .prompts import ATTACK_TUPLE_EXTRACTION_PROMPT, HARM_CERTIFICATE_PROMPT, AttackTuplesOutput
from .request_tuple import binary_from_tool, canonicalize, from_log_record

if TYPE_CHECKING:
    from ..config import DefenderConfig
    from ..policy.schema import Policy

# Well-known DefenderInput.context key an upstream attacker/observability layer may populate with
# already-structured request records, bypassing prose extraction entirely.
REQUEST_RECORDS_KEY = "request_records"


def _from_context(defender_input) -> list[RequestTuple] | None:
    """Structured records from ``context[REQUEST_RECORDS_KEY]``, if present — the preferred path."""
    records: Any = defender_input.context.get(REQUEST_RECORDS_KEY)
    if not records or not isinstance(records, list):
        return None
    tuples = [from_log_record(r) for r in records if isinstance(r, dict)]
    return [canonicalize(t) for t in tuples] or None


def _from_prompt_text(defender_input, policy: Policy, config: DefenderConfig) -> list[RequestTuple]:
    """Extract tuples from the attack prompt text itself.

    Reliable even when the agent response is opaque (e.g. just 'Hey ya!' from a beeceptor echo),
    because the attacker's instruction always contains the target URL and operation explicitly.
    """
    prompt = ATTACK_TUPLE_EXTRACTION_PROMPT.format(
        attack_prompt=defender_input.attack_prompt,
        attacked_tool=defender_input.attacked_tool,
    )
    output = complete_structured(prompt, AttackTuplesOutput, config)
    tuples = [canonicalize(t) for t in output.tuples]
    binary = binary_from_tool(defender_input.attacked_tool, policy)
    if binary:
        tuples = [t.model_copy(update={"binary": t.binary or binary}) for t in tuples]
    return tuples


def extract_attack_tuples(defender_input, policy: Policy, config: DefenderConfig | None = None) -> list[RequestTuple]:
    """Structured records first, prompt-text extraction second.

    Extracts from the attack prompt itself rather than the agent response — the prompt always
    contains the target URL and operation explicitly, whereas the response may be opaque
    (e.g. an echo service reply with no URL visible).

    Raises ``ExtractionError`` if neither path recovers anything.
    """
    tuples = _from_context(defender_input)
    if tuples is not None:
        return tuples
    tuples = _from_prompt_text(defender_input, policy, config or load_config())
    if not tuples:
        raise ExtractionError("could not recover any attack request tuples from the given evidence")
    return tuples


def build_harm_certificate(
    defender_input, tuples: list[RequestTuple], config: DefenderConfig | None = None
) -> HarmCertificate:
    """LLM call identifying the harm class, channel, and which tuple to cut."""
    tuple_lines = "\n".join(f"{i}: {t.model_dump(exclude_none=True)}" for i, t in enumerate(tuples))
    prompt = HARM_CERTIFICATE_PROMPT.format(
        attack_prompt=defender_input.attack_prompt,
        agent_response=defender_input.agent_response,
        tuples=tuple_lines or "(none)",
    )
    cert = complete_structured(prompt, HarmCertificate, config or load_config())
    if not (0 <= cert.cut_index < max(len(tuples), 1)):
        cert = cert.model_copy(update={"cut_index": 0})
    return cert


def select_cut(tuples: list[RequestTuple], harm_cert: HarmCertificate) -> RequestTuple:
    """The single tuple whose denial breaks the attack chain, per the harm certificate."""
    if not tuples:
        raise ExtractionError("no attack tuples to select a cut from")
    index = harm_cert.cut_index if 0 <= harm_cert.cut_index < len(tuples) else 0
    return tuples[index]
