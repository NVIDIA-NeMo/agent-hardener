# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legitimate calls to the guarded tool, written by an LLM from the benign suite's requests.

The benign suite holds user prompts, not tool calls, and a pre-tool rail is judged on tool calls. So
the requests are turned into the calls the agent would plausibly make for them, and the suggestor and
grader see what legitimate use of the guarded tool looks like. Calls are cached per tool and suite:
every attack on the same tool in a run is contrasted with the same benign calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

from langchain_core.language_models import BaseChatModel  # noqa: TC002 - LangGraph reads node annotations at runtime

from agent_hardener.models import DefenderInput  # noqa: TC001 - pydantic model used at runtime below

from ..prompting import messages
from ..schemas import BenignArgsResponse, BenignToolCall
from ..state import DefenderState  # noqa: TC001 - LangGraph reads node annotations at runtime

logger = logging.getLogger("agent_hardener.session")

#: Benign calls per tool: enough to catch an over-broad rail, few enough to keep validation cheap.
BENIGN_CALLS = 8

_cache: dict[str, list[BenignToolCall]] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def benign_args_node(state: DefenderState, *, llm: BaseChatModel) -> dict[str, Any]:
    """Benign calls to the analyzed tool, from the cache or freshly written."""
    analysis = state.vulnerability_analysis
    if analysis is None or analysis.tool_name is None:  # unreachable: routing only sends tool analyses here
        msg = "benign_args_node needs a tool analysis"
        raise ValueError(msg)
    requests = pick_requests(state.input, analysis.tool_name)
    if not requests:
        return {"benign_calls": []}
    key = hashlib.sha1(json.dumps([analysis.tool_name, requests]).encode(), usedforsecurity=False).hexdigest()
    with _lock_for(key):
        if key not in _cache:
            _cache[key] = _synthesize(llm, analysis.tool_name, analysis.call_args, requests)
        return {"benign_calls": _cache[key]}


def pick_requests(defender_input: DefenderInput, tool_name: str) -> list[str]:
    """Up to :data:`BENIGN_CALLS` requests, the ones the suite says exercise this tool first.

    Only ``benign`` rows (see :func:`benign_requests`).
    """
    bare = tool_name.removeprefix("functions.")
    suite = [r for r in defender_input.benign_suite if r.label in (None, "benign")]
    if suite:
        own = [r.payload for r in suite if (r.tool or "").removeprefix("functions.") == bare]
        others = [r.payload for r in suite if r.payload not in own]
        ordered = own + others
    else:
        ordered = list(defender_input.benign_requests)
    return list(dict.fromkeys(ordered))[:BENIGN_CALLS]


def benign_requests(defender_input: DefenderInput) -> list[str]:
    """The suite's must-allow requests.

    ``borderline_benign`` rows are left out. They graze attack patterns on purpose ("view the contents
    of /etc/shadow" is one), so a rail that refuses them may be right, and presenting them as
    must-allow pushes the rail away from exactly the attacks it should block.
    """
    suite = [r.payload for r in defender_input.benign_suite if r.label in (None, "benign")]
    return list(dict.fromkeys(suite or defender_input.benign_requests))


def _synthesize(llm: BaseChatModel, tool_name: str, attack_args: Any, requests: list[str]) -> list[BenignToolCall]:
    arg_names = list(attack_args) if isinstance(attack_args, dict) else []
    shape = {name: type(value).__name__ for name, value in attack_args.items()} if isinstance(attack_args, dict) else {}
    prompt = messages("benign_args", {"tool_name": tool_name, "arg_shape": json.dumps(shape), "requests": requests})
    try:
        response = llm.with_structured_output(BenignArgsResponse).invoke(prompt)
    except Exception:
        logger.exception("defender v3 could not write benign calls for %s", tool_name)
        return []
    if not isinstance(response, BenignArgsResponse):
        response = BenignArgsResponse.model_validate(response)
    calls = []
    for call in response.calls:
        args = _parse(call.args_json)
        if args is None or (arg_names and not set(args) <= set(arg_names)):
            logger.info("defender v3 dropped a benign call for %s with unusable args: %r", tool_name, call.args_json)
            continue
        calls.append(BenignToolCall(tool=tool_name, args=args, request=call.request))
    return calls


def _parse(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) and value else None


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())
