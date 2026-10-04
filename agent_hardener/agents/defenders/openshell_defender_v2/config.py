# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for openshell_defender_v2, parsed once per ``run()`` call."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DefenderConfig:
    """Tunables for the deterministic pipeline and its LLM boundary.

    Attributes:
        model: Chat model identifier passed to ``llm.client``.
        api_key_env: Env var name to read the inference API key from.
        max_iterations: Retry-loop cap references (kept for parity with the manager's own
            per-attack retry budget; this agent itself is single-shot per call).
        min_corpus_coverage: Predicted-benign-request count below which ``confidence_for``
            degrades a candidate to ``"low"``.
        node_priority_override: Optional override of ``analysis.selector.NODE_PRIORITY`` for tests.
    """

    model: str | None = None
    api_key_env: str = "INFERENCE_API_KEY"
    max_iterations: int = 3
    min_corpus_coverage: int = 3
    node_priority_override: tuple[str, ...] | None = None


def load_config(overrides: dict | None = None) -> DefenderConfig:
    """Build a :class:`DefenderConfig`, applying environment overrides then explicit ``overrides``."""
    kwargs: dict = {}
    if os.environ.get("OPENSHELL_DEFENDER_V2_MODEL"):
        kwargs["model"] = os.environ["OPENSHELL_DEFENDER_V2_MODEL"]
    if overrides:
        kwargs.update(overrides)
    return DefenderConfig(**kwargs)
