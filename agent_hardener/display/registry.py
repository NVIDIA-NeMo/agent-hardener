# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderer registry and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.display.helpers import all_artifacts, output_role
from agent_hardener.display.renderers import (
    GarakAttackerRenderer,
    GarakReplayValidatorRenderer,
    GenericRenderer,
    PolicyDefenderRenderer,
    SmartBenignValidatorRenderer,
)

if TYPE_CHECKING:
    from agent_hardener.display.protocol import AgentDisplayRenderer

_GENERIC = GenericRenderer()

_ARTIFACT_TYPE_RENDERERS: dict[str, AgentDisplayRenderer] = {
    "garak_report": GarakAttackerRenderer(),
    "garak_hitlog": GarakAttackerRenderer(),
    "defender.structured": PolicyDefenderRenderer(),
    "benign.requests_csv": SmartBenignValidatorRenderer(),
}


def resolve_renderer(output: dict[str, Any]) -> AgentDisplayRenderer:
    """Pick the best renderer for one serialized agent output.

    Resolution order:
    1. Artifact type  — stable data contract declared by the agent.
    2. Role + shape   — role inferred from the presence of records/policy_patches/kind.
    3. Generic        — fallback.
    """
    for artifact in all_artifacts(output):
        renderer = _ARTIFACT_TYPE_RENDERERS.get(artifact.type)
        if renderer is not None:
            return renderer

    role = output_role(output)
    if role == "attacker":
        return GarakAttackerRenderer()
    if role == "defender":
        return PolicyDefenderRenderer()
    if role == "validator":
        return SmartBenignValidatorRenderer() if output.get("kind") == "benign" else GarakReplayValidatorRenderer()
    return _GENERIC
