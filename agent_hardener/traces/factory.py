# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the configured trace provider, and the one flag check every component goes through.

Adding a backend means one ``case`` here and one name in ``TracesConfig.provider``; nothing outside
this package changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_hardener.traces.relay import RelayTraceProvider

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agent_hardener.models import AgentConfig, RelayVictimSpec, TracesConfig
    from agent_hardener.traces.provider import TraceProvider

logger = logging.getLogger(__name__)


def build_trace_provider(
    cfg: TracesConfig,
    *,
    relay_spec: RelayVictimSpec | None,
    atof_path: Path | None,
    sync: Callable[[], None],
    raw_dir: Path,
) -> TraceProvider | None:
    """The provider ``cfg`` asks for, or ``None`` when traces are off or the backend is unavailable."""
    match cfg.provider:
        case "relay":
            if relay_spec is None or atof_path is None:
                logger.warning("traces.provider is 'relay' but the victim has no relay-victim block; traces are off")
                return None
            return RelayTraceProvider(atof_path, sync, raw_dir)
        case _:
            return None


def uses_traces(agent: AgentConfig, provider: TraceProvider | None) -> bool:
    """Whether ``agent`` consumes traces: the global provider exists and its own ``use_traces`` is on.

    The only flag check. Components never read the flags themselves: the orchestrator hands a trace or
    a provider only to the agents this returns ``True`` for.
    """
    return provider is not None and bool(agent.config.get("use_traces", False))
