# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured logging for the selection/synthesis pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Candidate, DefenderContext, Selection

logger = logging.getLogger(__name__)


def log_selection(ctx: DefenderContext, selection: Selection) -> None:
    """Log the outcome of ``analysis.selector.select`` for one defender run."""
    logger.info(
        "openshell_defender_v2 selection",
        extra={
            "chosen": selection.chosen,
            "fallback_nodes": [name for name, _ in selection.fallbacks],
            "infeasible": selection.infeasible,
            "iteration": ctx.iteration,
        },
    )


def log_candidate(ctx: DefenderContext, candidate: Candidate) -> None:
    """Log the synthesized candidate before it's linted and dumped to YAML."""
    logger.info(
        "openshell_defender_v2 candidate",
        extra={
            "node": candidate.node,
            "resource_type": candidate.resource_type,
            "confidence": candidate.confidence,
            "guardrail_name": candidate.guardrail_name,
            "iteration": ctx.iteration,
        },
    )
