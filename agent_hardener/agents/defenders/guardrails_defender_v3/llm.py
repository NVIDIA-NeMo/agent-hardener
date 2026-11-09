# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA-hosted chat model for the guardrails defender's nodes (:func:`agent_hardener.llm.build_chat_model`).

The model is :data:`agent_hardener.llm.DEFAULT_MODEL` (``AGENT_HARDENER_MODEL``) for every role. v3 builds it
once per attack for the analyzer, the benign-args writer and the suggestor, and once more at temperature 0 for
the grader, and hands both to the graph, so a test can swap in a scripted model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.llm import build_chat_model

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

#: Low but not zero: the analysis should be repeatable, the rail wording may vary.
TEMPERATURE = 0.2
#: The grader scores drafts against a rubric, so it should give the same draft the same score.
GRADER_TEMPERATURE = 0.0


def get_llm(temperature: float = TEMPERATURE) -> ChatOpenAI:
    """Return the shared NVIDIA-hosted chat model for this defender's nodes."""
    return build_chat_model(temperature=temperature)
