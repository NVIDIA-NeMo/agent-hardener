# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subgraphs that compose the smart benign validator's synth DAG.

Each subgraph is a compiled LangGraph ``CompiledStateGraph`` with its own private
typed state and entry/exit adapters that bridge to the parent ``SynthState``.
See the package-level ``graph.py`` for how they are wired into the parent graph.
"""

from __future__ import annotations
