# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading display artifacts from agent outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.models import Artifact

if TYPE_CHECKING:
    from collections.abc import Callable


def all_artifacts(output: dict[str, Any]) -> list[Artifact]:
    """Return artifacts declared by the agent on this output."""
    raw = output.get("artifacts")
    return [Artifact.model_validate(a) for a in raw] if isinstance(raw, list) else []


def verbose_result_lines(output: dict[str, Any], *, label: str, line_for: Callable[[dict[str, Any]], str]) -> list[str]:
    """Header + up to 20 per-result lines (``line_for`` each) with a ``… +N more`` tail, else the summary.

    Shared by the replay-validator renderers, which differ only in the per-result line and (for benign)
    a trailing artifact list the caller appends.
    """
    metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
    results = metadata.get("results") if isinstance(metadata.get("results"), list) else []
    lines = [f"  {output.get('agent_name', '<unknown>')} {label}:"]
    if not results:
        lines.append(f"    {output.get('summary', '')}")
        return lines
    for index, result in enumerate(results[:20], start=1):
        if isinstance(result, dict):
            lines.append(f"    #{index} {line_for(result)}")
    if len(results) > 20:
        lines.append(f"    … +{len(results) - 20} more")
    return lines


def output_role(output: dict[str, Any]) -> str | None:
    """Infer agent role from a serialized output dict."""
    if "kind" in output:
        return "validator"
    if "policy_patches" in output:
        return "defender"
    if "records" in output:
        return "attacker"
    if "observations" in output:
        return "victim"
    return None
