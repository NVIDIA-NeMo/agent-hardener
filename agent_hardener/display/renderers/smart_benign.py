# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smart benign validator display renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text

from agent_hardener.display.helpers import all_artifacts, verbose_result_lines

if TYPE_CHECKING:
    from agent_hardener.display.context import DisplayContext


def _benign_line(result: dict[str, Any]) -> str:
    verdict = result.get("verdict")
    status = verdict.get("status") if isinstance(verdict, dict) else "?"
    return f"{result.get('tool', '?')}: {status}"


class SmartBenignValidatorRenderer:
    """Render smart benign replay metadata and artifact pointers."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Text | None:
        lines = verbose_result_lines(output, label="benign replay", line_for=_benign_line)
        for artifact in all_artifacts(output):
            lines.append(f"    artifact: {artifact.type} → {artifact.path}")
        return Text("\n".join(lines))
