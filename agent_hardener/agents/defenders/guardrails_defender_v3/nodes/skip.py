# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip an attack no tool rail can separate from ordinary use: the defender abstains, saying why."""

from __future__ import annotations

from typing import Any

from agent_hardener.models import DefenderOutput

from ..schemas import Outcome
from ..state import DefenderState  # noqa: TC001 - LangGraph reads node annotations at runtime


def skip_node(state: DefenderState) -> dict[str, Any]:
    """Write no rail: no tool call carried the harm, so a tool rail could not tell the attack apart.

    The defender abstains (``ok=False`` with the reason), as a defender does for an attack outside its
    scope, so the manager records why no guardrail was added.
    """
    analysis = state.vulnerability_analysis
    why = (analysis.fallback_reason if analysis else None) or "no tool call to guard"
    reason = f"no tool rail: {why}"
    outcome = Outcome(attack=state.attack, status="skipped", reason=reason, fallback_reason=why)
    return {"outcome": outcome, "output": DefenderOutput(ok=False, error_message=f"skipped: {reason}")}
