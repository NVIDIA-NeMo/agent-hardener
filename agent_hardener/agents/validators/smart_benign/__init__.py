# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smart benign validator: infers victim capabilities and replays a tiered benign suite.

Replaces the static-CSV benign validator. Given heterogeneous descriptions of a victim
(NL text, GitHub repo, API endpoint, interactive Q&A) this package synthesizes a
``VictimCapabilityProfile``, generates a tiered benign request suite, and replays it
against the post-mitigation victim to detect over-blocking guardrails.

The orchestrator entry points are :func:`synthesize` (pre-flight, run once per
target) and :func:`run` (per-attempt replay of the synthesized suite).
"""

from __future__ import annotations

from .validator import cached_suite_count, run, synthesize

__all__ = ["cached_suite_count", "run", "synthesize"]
