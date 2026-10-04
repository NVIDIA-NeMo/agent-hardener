# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonicalization and bridging helpers for :class:`RequestTuple`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import RequestTuple

if TYPE_CHECKING:
    from ..policy.schema import Policy


def canonicalize(t: RequestTuple) -> RequestTuple:
    """Normalize a tuple so identical requests compare equal (lowercase host/method, strip

    trailing slash from path, drop an explicit default port).
    """
    return t.model_copy(
        update={
            "host": t.host.lower() if t.host else t.host,
            "method": t.method.upper() if t.method else t.method,
            "path": t.path.rstrip("/") or "/" if t.path else t.path,
        }
    )


def tuple_id(t: RequestTuple) -> str:
    """A stable string key for deduping/logging one tuple."""
    parts = [
        t.binary or "-",
        t.host or "-",
        str(t.port) if t.port else "-",
        t.protocol or "-",
        t.method or "-",
        t.path or "-",
    ]
    return "|".join(parts)


def from_log_record(record: dict[str, Any]) -> RequestTuple:
    """Build a :class:`RequestTuple` from a structured log/trace record, when one is available

    in ``DefenderInput.context`` (the preferred, non-LLM extraction path).
    """
    return RequestTuple(
        binary=record.get("binary"),
        host=record.get("host"),
        port=record.get("port"),
        protocol=record.get("protocol"),
        method=record.get("method"),
        path=record.get("path"),
        query=record.get("query"),
        gql_field=record.get("gql_field"),
        mcp_tool=record.get("mcp_tool"),
    )


def binary_from_tool(attacked_tool: str, policy: Policy) -> str | None:
    """Best-effort bridge from the swarm's ``attacked_tool`` namespace (e.g. ``bash_executor``) to

    a ``binaries[].path`` entry in the policy (e.g. ``/bin/bash``) — the two never match literally.

    Uses substring matching against the tool name's normalized form; returns ``None`` (not a
    guess) when nothing plausible is found so callers fall back to host/port-only matching.
    """
    normalized = attacked_tool.lower().replace("_executor", "").replace("_tool", "")
    for entry in policy.network_policies.values():
        if not entry.binaries:
            continue
        for binary in entry.binaries:
            stem = binary.path.rsplit("/", 1)[-1].lower()
            if normalized and (normalized in stem or stem in normalized):
                return binary.path
    return None
