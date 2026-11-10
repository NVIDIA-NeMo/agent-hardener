# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas sent as structured output must stay within what grammar-constrained decoders accept."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes.data_augmenter import AugmentedData
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.prompts import (
    AttackTuplesOutput,
    EnvironmentEndpointOutput,
)
from agent_hardener.agents.defenders.openshell_defender_v2.models import HarmCertificate
from agent_hardener.agents.defenders.openshell_defender_v2.policy.merge import MergeConflictOutput
from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.schemas import ProberOutput
from agent_hardener.agents.validators.smart_benign.subgraphs.critic.schemas import CriticResult
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.schemas import BoundaryVerdict
from agent_hardener.agents.validators.smart_benign.subgraphs.github_analyzer.schemas import SummarizerOutput
from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.schemas import InterviewBatch
from agent_hardener.agents.validators.smart_benign.subgraphs.nl_parser.schemas import ParserOutput
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.schemas import (
    NormalizedProfileGlobals,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.schemas import ToolGenerationOutput
from agent_hardener.relay_plugin.judge import JudgeScore

if TYPE_CHECKING:
    from pydantic import BaseModel

# Every model passed to ``with_structured_output``.
STRUCTURED_OUTPUT_SCHEMAS: list[type[BaseModel]] = [
    AttackTuplesOutput,
    AugmentedData,
    BoundaryVerdict,
    CriticResult,
    EnvironmentEndpointOutput,
    HarmCertificate,
    InterviewBatch,
    JudgeScore,
    MergeConflictOutput,
    NormalizedProfileGlobals,
    ParserOutput,
    ProberOutput,
    SummarizerOutput,
    ToolGenerationOutput,
]


def _pattern_with_length(node: Any, path: str = "#") -> list[str]:
    """JSON-schema paths of string nodes that set ``pattern`` together with a length bound."""
    if isinstance(node, list):
        return [hit for i, item in enumerate(node) for hit in _pattern_with_length(item, f"{path}/{i}")]
    if not isinstance(node, dict):
        return []
    hits = [path] if "pattern" in node and ({"minLength", "maxLength"} & node.keys()) else []
    return hits + [hit for key, value in node.items() for hit in _pattern_with_length(value, f"{path}/{key}")]


@pytest.mark.parametrize("schema", STRUCTURED_OUTPUT_SCHEMAS, ids=lambda s: s.__name__)
def test_no_pattern_combined_with_length_bounds(schema: type[BaseModel]) -> None:
    # xgrammar (behind several OpenAI-compatible providers) rejects the whole request with a 400.
    assert _pattern_with_length(schema.model_json_schema()) == []
