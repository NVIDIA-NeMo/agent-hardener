# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable identity helpers for configured agents and runs."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_hardener.models import AgentConfig


def normalize_agent_name(name: str) -> str:
    """Normalize a display name into a stable id prefix."""
    normalized = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized or "agent"


def stable_agent_config(agent: AgentConfig) -> dict[str, Any]:
    """Return only identity-affecting config fields."""
    return {
        "name": agent.name,
        "role": agent.role,
        "service_url": agent.service_url,
        "timeout_seconds": agent.timeout_seconds,
        "implementation": agent.implementation,
        "config": agent.config,
    }


def stable_config_json(agent: AgentConfig) -> str:
    """Return canonical JSON used for agent-id hashing."""
    return json.dumps(stable_agent_config(agent), sort_keys=True, separators=(",", ":"), default=str)


def build_agent_id(agent: AgentConfig) -> str:
    """Build normalized-name plus eight-character config hash."""
    digest = hashlib.sha256(stable_config_json(agent).encode("utf-8")).hexdigest()[:8]
    return f"{normalize_agent_name(agent.name)}-{digest}"


def generate_round_id(now: datetime | None = None, random_uuid: uuid.UUID | None = None) -> str:
    """Build a UTC timestamp plus short UUID id (used for run/mission/round ids)."""
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.astimezone(UTC)
    suffix = (random_uuid or uuid.uuid4()).hex[:8]
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{suffix}"


def format_round_id(round_number: int) -> str:
    """Build a human-readable round id scoped to one mission."""
    if round_number < 1:
        msg = "round_number must be at least 1"
        raise ValueError(msg)
    return f"round-{round_number:04d}"
