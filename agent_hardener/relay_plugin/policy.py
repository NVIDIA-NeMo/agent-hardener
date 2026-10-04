# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a tool call should be refused — the decision, with no Relay in sight.

Kept import-free of ``nemo_relay`` on purpose. The decision is the part worth testing hard (an
over-broad guardrail breaks the agent; an under-broad one lets the attack through), and it should be
testable without a Relay runtime or a live judge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import Guardrail

logger = logging.getLogger("agent_hardener.relay_plugin")


class SafetyJudge(Protocol):
    """Scores how strongly a tool call matches a guardrail's description of an attack."""

    def score(self, instructions: str, tool_name: str, args: str) -> float:
        """Return 0.0 (clearly benign) to 1.0 (clearly the attack)."""


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome for one tool call."""

    refuse: bool
    reason: str = ""


def decide(rail: Guardrail, tool_name: str, args: Any, judge: SafetyJudge) -> Decision:
    """Decide whether ``rail`` refuses this call.

    A guardrail is scoped to one tool, so a call to any other tool is allowed without consulting the
    judge — that check is what keeps the judge off the hot path for the agent's ordinary work, and
    the judge is an LLM call inside the tool path.

    The judge scores the **tool call** — the same input NAT's ``pre_tool_verifier`` had, where
    ``target_location: input`` selected the *function's* input rather than its output
    (``nat.middleware.common.TargetLocation``). Scoping to one tool is handled here by
    ``target_tool``, mirroring NAT's ``target_function_or_group``.

    Known limit, inherited rather than introduced: a benign and a hostile call can be nearly
    identical as arguments — ``transfer_funds`` with an amount and two accounts either way — so a
    rule keyed on how the request was *phrased* has little to work with here.

    A judge that raises **allows** the call. Failing closed would turn a transient inference error
    into an agent that refuses everything, which reads in the report as a guardrail so effective it
    broke the agent. Failing open is visible instead: the attack lands and the round says so.
    """
    if not _targets(rail.target_tool, tool_name):
        return Decision(refuse=False)
    try:
        score = judge.score(rail.system_instructions, tool_name, _render_args(args))
    except Exception:
        logger.exception(
            "agent-hardener guardrail %s could not score a %s call; allowing it rather than refusing "
            "every call on a transient judge failure",
            rail.name,
            tool_name,
        )
        return Decision(refuse=False)
    if score < rail.threshold:
        return Decision(refuse=False)
    return Decision(refuse=True, reason=_refusal(rail, tool_name, score))


#: OpenAI-style tool-calling namespace. Models emit calls as ``functions.<name>``, and the defender
#: writes ``target_tool`` by reading the attack transcript, so it copies the namespace through often
#: enough to matter — the same run can produce ``bash_executor`` one round and
#: ``functions.bash_executor`` the next.
_TOOL_NAMESPACE = "functions."


def _targets(target_tool: str, tool_name: str) -> bool:
    """Whether ``tool_name`` is the tool ``target_tool`` means to guard.

    Relay reports the bare name (``bash_executor``); ``target_tool`` is written by an LLM and may
    carry the caller's namespace (``functions.bash_executor``). Compared exactly, those differ, and
    :func:`decide` reads a mismatch as "not my tool" and allows the call — so a one-word difference
    silently disables the guardrail, with no judge call and nothing in the log to show for it. That
    is indistinguishable in a report from a guardrail that ran and found nothing wrong.

    Only the known namespace is stripped, not any dotted prefix: a tool genuinely named
    ``a.b`` should not be guarded by a rail aimed at ``b``.
    """
    return tool_name == target_tool or tool_name == target_tool.removeprefix(_TOOL_NAMESPACE)


def _refusal(rail: Guardrail, tool_name: str, score: float) -> str:
    """The message the agent sees. Names the guardrail so a run report can trace it back."""
    return f"agent-hardener[{rail.name}]: refused {tool_name} (safety score {score:.2f} >= {rail.threshold:.2f})"


def _render_args(args: Any) -> str:
    """Arguments as text for the judge, tolerating values that do not serialise.

    Relay hands the callback any JSON value, not necessarily an object — a tool taking a bare string
    is legal — so this must not assume a mapping.
    """
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)
