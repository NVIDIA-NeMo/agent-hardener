# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one interface every trace backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hardener.traces.models import AgentTrace


class TraceProvider(ABC):
    """Returns the victim's traces. Nothing else.

    A provider does no matching, filtering or querying: consumers pick what they need with
    :mod:`agent_hardener.traces.matching`, which only reads :class:`AgentTrace` fields and so works for
    every backend.

    Every provider must:

    * return ``[]`` and never raise when there is no data, because an empty result is always safe;
    * return fresh data on every call, so callers call it after the traffic they care about;
    * fill ``ToolCall.ok`` / ``outcome`` / ``rail`` and ``AgentTrace.guardrail_latched`` with
      :mod:`agent_hardener.traces.signals`, never with its own patterns;
    * never lose history: if the source can be reset (a sandbox recreate), keep what it already saw.
    """

    @abstractmethod
    def get_traces(self) -> list[AgentTrace]:
        """All traces currently available from the victim, oldest first. Empty list if none.

        Synchronous because the sandbox fetch is; callers on the event loop wrap it in
        ``asyncio.to_thread``.
        """
