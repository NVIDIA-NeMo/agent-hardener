# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default renderer for agents without a bespoke display implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text

from agent_hardener.display.helpers import all_artifacts

if TYPE_CHECKING:
    from agent_hardener.display.context import DisplayContext


class GenericRenderer:
    """Print summary plus artifact paths when no specialized renderer matches."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Text | None:
        artifacts = all_artifacts(output)
        if not artifacts:
            return None
        lines = [f"  artifacts ({output.get('agent_name', '<unknown>')}):"]
        for artifact in artifacts:
            label = artifact.label or artifact.path.name
            lines.append(f"    - {artifact.type}: {label} → {artifact.path}")
        return Text("\n".join(lines), style="dim")
