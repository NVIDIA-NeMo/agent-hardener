# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guardrails defender: runs v2 or v3, as the run's feature flag says.

- ``v2`` (:mod:`agent_hardener.agents.defenders.guardrails_defender_v2`) works from the attack prompt
  and the agent's response, so it has to infer which tool was attacked.
- ``v3`` (:mod:`agent_hardener.agents.defenders.guardrails_defender_v3`) reads the attack's execution
  trace and rails the exact call that did the harm, falling back to the user prompt.

The flag is ``defenders.guardrails_version`` in the manifest, which lands in this defender's config
(and so in ``DefenderInput.context``) as :data:`GUARDRAILS_VERSION_KEY`. Both versions are imported
only when chosen, so the manifest can import this module without pulling in either graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hardener.models import DefenderInput, DefenderOutput

#: ``DefenderInput.context`` key carrying the feature flag.
GUARDRAILS_VERSION_KEY = "guardrails_version"
#: The version a config without the flag runs: the behaviour before v3 existed.
DEFAULT_VERSION = "v2"


def run(defender_input: DefenderInput) -> DefenderOutput:
    """Run the guardrails defender version the flag selects."""
    version = defender_input.context.get(GUARDRAILS_VERSION_KEY) or DEFAULT_VERSION
    if version == "v3":
        from agent_hardener.agents.defenders.guardrails_defender_v3.guardrails_defender_agent import (  # noqa: PLC0415
            run as run_v3,
        )

        return run_v3(defender_input)
    if version == "v2":
        from agent_hardener.agents.defenders.guardrails_defender_v2.guardrails_defender_agent import (  # noqa: PLC0415
            run as run_v2,
        )

        return run_v2(defender_input)
    msg = f"unknown {GUARDRAILS_VERSION_KEY} {version!r}: expected 'v2' or 'v3'"
    raise ValueError(msg)
