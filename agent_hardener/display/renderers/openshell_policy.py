# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell policy defender display renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text

from agent_hardener.display.policy_detail import render_policy_detail

if TYPE_CHECKING:
    from agent_hardener.display.context import DisplayContext


def render_defender_detail(defenders: list[dict[str, Any]], policy_patches: list[dict[str, Any]]) -> Text:
    """Defender reasoning (statuses + per-finding decisions + candidate policy/guardrails) as text."""
    return Text("\n".join(render_policy_detail(defenders, policy_patches, [])).rstrip())


class PolicyDefenderRenderer:
    """Render defender reasoning and policy detail (OpenShell and guardrails alike)."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Any | None:
        defenders = [output]
        policy_patches = output.get("policy_patches") if isinstance(output.get("policy_patches"), list) else []
        return render_defender_detail(defenders, policy_patches)
