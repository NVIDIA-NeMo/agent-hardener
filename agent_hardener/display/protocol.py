# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderer protocol for agent-specific CLI presentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_hardener.display.context import DisplayContext


class AgentDisplayRenderer(Protocol):
    """Render the verbose view for one agent family."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Any | None:
        """Verbose detail for a single agent output."""
