# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swappable, environment-specific dependencies (the two-worlds seam).

Each provider is selected from config via a ``build_*`` factory and conforms to a small base class /
Protocol, so standalone-OSS and NeMo-platform deployments can supply different implementations
without touching the run engine. The garak agent_breaker attacker now spawns the garak CLI directly,
so there is no garak provider here; backends remain.
"""

from __future__ import annotations

from agent_hardener.providers.backends import BackendManager, resolve_compose_command

__all__ = [
    "BackendManager",
    "resolve_compose_command",
]
