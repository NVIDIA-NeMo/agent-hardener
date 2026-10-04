# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every feasibility predicate and pick exactly one node to synthesize.

Order is specificity corrected for bypass resistance: path/method levers can't be routed around
without changing what the agent is asking for; binary can (swap ``curl`` for ``python -c``), so
it sits low despite touching the least policy surface.
"""

from __future__ import annotations

from ..models import DefenderContext, Selection, Witness
from .feasibility import FEASIBILITY_CHECKS, infeasibility_reason

NODE_PRIORITY: list[str] = [
    "remove_endpoint",
    "add_deny_rule",
]


def _evaluate_all(ctx: DefenderContext, excluded: set[str]) -> tuple[dict[str, Witness], dict[str, str]]:
    witnesses: dict[str, Witness] = {}
    infeasible: dict[str, str] = {}
    for name in NODE_PRIORITY:
        if name in excluded:
            infeasible[name] = "excluded by validator feedback from a previous round"
            continue
        check = FEASIBILITY_CHECKS[name]
        witness = check(ctx)
        if witness is not None:
            witnesses[name] = witness
        else:
            infeasible[name] = infeasibility_reason(name, ctx)
    return witnesses, infeasible


def select(ctx: DefenderContext, excluded: set[str] | None = None) -> Selection:
    """Evaluate the whole priority chain and return the head to synthesize plus the fallbacks."""
    witnesses, infeasible = _evaluate_all(ctx, excluded or set())
    ordered = [name for name in NODE_PRIORITY if name in witnesses]
    if not ordered:
        return Selection(chosen=None, witness=None, fallbacks=[], infeasible=infeasible)
    chosen = ordered[0]
    fallbacks = [(name, witnesses[name]) for name in ordered[1:]]
    return Selection(chosen=chosen, witness=witnesses[chosen], fallbacks=fallbacks, infeasible=infeasible)
