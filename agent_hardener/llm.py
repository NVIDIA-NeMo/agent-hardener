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

import functools
import importlib
import os
from typing import TYPE_CHECKING, Any

import openai
from langchain_openai import ChatOpenAI

from agent_hardener.env import inference_api_key
from agent_hardener.llm_telemetry import LlmTelemetryCallback
from agent_hardener.rate_limits import AsyncRetryTransport, RetryTransport
from agent_hardener.relay_plugin.config import no_reasoning_body

if TYPE_CHECKING:
    from types import ModuleType

_FALLBACK_BASE_URL = "https://integrate.api.nvidia.com/v1"
_FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def default_base_url() -> str:
    """NVIDIA's OpenAI-compatible endpoint every agent talks to, unless ``AGENT_HARDENER_BASE_URL`` retargets it.

    Read live, not at import: the CLI loads the ``--env-file`` after this module is imported.
    """
    return os.environ.get("AGENT_HARDENER_BASE_URL") or _FALLBACK_BASE_URL


def default_model() -> str:
    """The "analysis" model for callers that don't thread one through their own config (defenders, benign validator).

    ``AGENT_HARDENER_MODEL`` overrides it. Read live, not at import, for the same reason as :func:`default_base_url`.
    """
    return os.environ.get("AGENT_HARDENER_MODEL") or _FALLBACK_MODEL


def build_chat_model(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Build the shared chat-model client for NVIDIA's inference endpoint.

    Args:
        model: Model identifier; defaults to :func:`default_model`.
        base_url: Endpoint override; falls back to :func:`default_base_url`.
        api_key: Credential override; falls back to the ``INFERENCE_API_KEY`` env var.
        temperature: Sampling temperature; omitted from the client when ``None`` so the
            server/client default applies.
        **kwargs: Extra keyword arguments forwarded verbatim to :class:`ChatOpenAI`.

    Returns:
        A configured :class:`~langchain_openai.ChatOpenAI` instance.
    """
    model = model or default_model()
    if temperature is not None:
        kwargs["temperature"] = temperature
    if (body := no_reasoning_body(model)) is not None:
        kwargs.setdefault("extra_body", body)
    # Attach per-LLM-call telemetry alongside any caller-provided callbacks (e.g. langfuse) so every
    # agent's model traffic is captured and attributed to the running agent.
    callbacks = list(kwargs.pop("callbacks", None) or [])
    callbacks.append(LlmTelemetryCallback())
    # A rate limit is waited out (backoff, then retry) below the SDK, so every caller gets it without
    # handling 429 itself; a caller that passes its own HTTP clients keeps them. The SDK's own retries
    # are turned off with it: they sit on top of the transport, so each re-send would get a fresh wait
    # budget (3 x 600s in the worst case). The cost is that a 5xx or timeout is no longer retried.
    if "http_client" not in kwargs and "http_async_client" not in kwargs:
        kwargs["http_client"], kwargs["http_async_client"] = _rate_limited_http_clients()
        kwargs.setdefault("max_retries", 0)
    return ChatOpenAI(
        model=model,
        base_url=base_url or default_base_url(),
        api_key=api_key or inference_api_key(),
        callbacks=callbacks,
        **kwargs,
    )


@functools.cache
def _rate_limited_http_clients() -> tuple[openai.DefaultHttpxClient, openai.DefaultAsyncHttpxClient]:
    """The HTTP clients every chat model shares: the SDK's defaults, over a transport that waits out rate limits.

    Shared rather than built per model, like the SDK's own default clients, so each ``build_chat_model``
    call does not open a new connection pool.
    """
    http = _sdk_http_module()
    return (
        openai.DefaultHttpxClient(transport=RetryTransport(http.HTTPTransport())),
        openai.DefaultAsyncHttpxClient(transport=AsyncRetryTransport(http.AsyncHTTPTransport())),
    )


def _sdk_http_module() -> ModuleType:
    """The HTTP library the installed openai SDK is built on: ``httpx`` (openai 1.x/2.x) or ``httpx2`` (3.x).

    The retrying transport only wraps the real one, so the real one must come from the same library as
    the SDK's client: a request of one library cannot be sent by the other's transport.
    """
    for cls in openai.DefaultAsyncHttpxClient.__mro__:
        if cls.__name__ == "AsyncClient":
            return importlib.import_module(cls.__module__.split(".", 1)[0])
    raise RuntimeError(f"cannot tell which HTTP library openai {openai.__version__} uses")  # pragma: no cover
