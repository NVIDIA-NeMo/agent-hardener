# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Langfuse tracing for the smart benign validator synth DAG.

Set LANGFUSE_ENABLED=1 plus LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and
optionally LANGFUSE_HOST to enable. All three env vars must be present; missing
any one silently disables tracing with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_hardener.env import langfuse_enabled

logger = logging.getLogger(__name__)


@dataclass
class TracingContext:
    """Holds Langfuse state for one validator run."""

    callbacks: list[Any] = field(default_factory=list)
    client: Any | None = None  # langfuse.Langfuse instance; None when disabled
    round_id: str | None = None
    target_name: str | None = None
    enabled: bool = False


def make_langfuse_callbacks(*, round_id: str | None = None, target_name: str | None = None) -> TracingContext:
    if not langfuse_enabled():
        return TracingContext()
    try:
        from langfuse import Langfuse  # noqa: PLC0415
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415

        # v4: credentials are read from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars
        # Silence the OTel OTLP span-exporter timeout noise — Langfuse can be slow but traces still land.
        # Set OTEL_EXPORTER_OTLP_TRACES_TIMEOUT=30000 in .env for a proper fix.
        logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)
        handler = CallbackHandler()
        client = Langfuse()
        logger.info("langfuse tracing enabled", extra={"target": target_name})
        return TracingContext(
            callbacks=[handler],
            client=client,
            round_id=round_id,
            target_name=target_name,
            enabled=True,
        )
    except ImportError:
        logger.warning("LANGFUSE_ENABLED set but langfuse not installed; run: uv add langfuse")
        return TracingContext()
