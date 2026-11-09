# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guardrails defender's stable entry point: ``agent_hardener.agents.defenders.guardrails_defender:run``.

Manifests and integrations name this path, so it stays put while the implementation lives in
:mod:`agent_hardener.agents.defenders.guardrails_defender_v3`: it reads the attack's execution trace and
writes a rail on the exact tool call that did the harm. The graph is imported on the first call, so the
manifest can import this module without loading it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hardener.models import DefenderInput, DefenderOutput


def run(defender_input: DefenderInput) -> DefenderOutput:
    """Write a trace-grounded guardrail for one attack."""
    from agent_hardener.agents.defenders.guardrails_defender_v3.guardrails_defender_agent import (  # noqa: PLC0415
        run as run_v3,
    )

    return run_v3(defender_input)
