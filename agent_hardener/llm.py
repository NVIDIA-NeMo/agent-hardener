# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single factory for the project's chat-model client.

Every LLM call in Agent Hardener targets NVIDIA's OpenAI-compatible inference endpoint, so all
model construction funnels through :func:`build_chat_model`. Keeping the client class, the
default endpoint, and credential lookup in one place means callers pass only what varies
(model name, and an optional temperature) and there is exactly one place to change the
provider, base URL, or auth scheme.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from agent_hardener.env import inference_api_key
from agent_hardener.llm_telemetry import LlmTelemetryCallback
from agent_hardener.relay_plugin.config import no_reasoning_body

# NVIDIA's OpenAI-compatible inference endpoint that every agent talks to by default. Overridable via
# AGENT_HARDENER_BASE_URL so an operator can retarget every default-model caller (defenders, benign
# validator) at a different endpoint without touching code or per-component config.
DEFAULT_BASE_URL = os.environ.get("AGENT_HARDENER_BASE_URL") or "https://integrate.api.nvidia.com/v1"
# Default model for callers that don't thread one through their own config (e.g. defenders). Overridable
# via AGENT_HARDENER_MODEL — the single lever for the "analysis" model across defenders + benign validator.
DEFAULT_MODEL = os.environ.get("AGENT_HARDENER_MODEL") or "nvidia/nemotron-3-super-120b-a12b"


def build_chat_model(
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Build the shared chat-model client for NVIDIA's inference endpoint.

    Args:
        model: Model identifier; defaults to :data:`DEFAULT_MODEL`.
        base_url: Endpoint override; falls back to :data:`DEFAULT_BASE_URL`.
        api_key: Credential override; falls back to the ``INFERENCE_API_KEY`` env var.
        temperature: Sampling temperature; omitted from the client when ``None`` so the
            server/client default applies.
        **kwargs: Extra keyword arguments forwarded verbatim to :class:`ChatOpenAI`.

    Returns:
        A configured :class:`~langchain_openai.ChatOpenAI` instance.
    """
    if temperature is not None:
        kwargs["temperature"] = temperature
    if (body := no_reasoning_body(model)) is not None:
        kwargs.setdefault("extra_body", body)
    # Attach per-LLM-call telemetry alongside any caller-provided callbacks (e.g. langfuse) so every
    # agent's model traffic is captured and attributed to the running agent.
    callbacks = list(kwargs.pop("callbacks", None) or [])
    callbacks.append(LlmTelemetryCallback())
    return ChatOpenAI(
        model=model,
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key or inference_api_key(),
        callbacks=callbacks,
        **kwargs,
    )
