# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA-hosted chat model for the guardrails defender's LangGraph nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.llm import build_chat_model

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return the shared NVIDIA-hosted chat model for this defender's nodes."""
    return build_chat_model(temperature=temperature)
