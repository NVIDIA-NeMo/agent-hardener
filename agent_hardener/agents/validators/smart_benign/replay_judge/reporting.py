# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for formatting per-row results into ``ValidatorReport`` fields."""

from __future__ import annotations

from typing import Any, Literal

VerdictStatus = Literal["complied", "refused", "error"]

_STATS_KEY: dict[VerdictStatus, str] = {
    "complied": "complied",
    "refused": "refused",
    "error": "errors",
}


def stats_key(status: VerdictStatus) -> str:
    """Map a verdict status to its key in the per-tool stats dict."""
    return _STATS_KEY[status]


def format_finding(result: dict[str, Any]) -> str:
    """Format one result dict as a single ``ValidatorReport.findings`` entry."""
    verdict = result["verdict"]
    label = {"complied": "COMPLIED", "refused": "REFUSED", "error": "ERROR"}[verdict["status"]]
    confidence = verdict.get("confidence") or 0.0
    return (
        f"{label} index={result['index']} label={result.get('label', '?')} tool={result['tool']} "
        f"confidence={confidence:.2f} payload={result['payload_excerpt']!r} "
        f"reason={verdict.get('reasoning', '')}"
    )


def excerpt(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` with a visible ellipsis marker."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"
