# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Before the sandbox is built, every model a run will call must answer with the key it will be called with.

Each model is resolved by the code the run itself uses to pick it, so a model, endpoint or key set by env,
by a manifest or by Helix's flags is checked as the run will use it: the analysis model from the env, the
guardrail judge as the guardrails defender will write it, and garak's models from the attacker's resolved
garak config.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_hardener.agents.attackers.agent_breaker.preflight import model_uses
from agent_hardener.agents.attackers.agent_breaker.run_agent_breaker import DEFAULT_REPORT_DIR, ScanOptions
from agent_hardener.agents.defenders.traces_guardrails_defender.nodes.component_writer import planned_judge
from agent_hardener.manifest import AGENT_BREAKER_IMPLEMENTATION, GUARDRAILS_DEFENDER_IMPLEMENTATION
from agent_hardener.model_check import ModelUse, analysis_model_use, require_answers, require_inference_key
from agent_hardener.openshell.lifecycle import sandbox_credentials
from agent_hardener.relay_plugin.config import judge_credentials, judge_endpoint, no_reasoning_body
from agent_hardener.tools.openshell import openshell_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agent_hardener.models import SessionConfig


def preflight_run_models(session: SessionConfig) -> None:
    """Raise :class:`~agent_hardener.errors.ModelUnavailableError` for the first model of the run that won't answer."""
    require_inference_key()
    require_answers([analysis_model_use(), *_judge_uses(session), *_attacker_uses(session)])


def _judge_uses(session: SessionConfig) -> list[ModelUse]:
    defender = next((d for d in session.defenders if d.implementation == GUARDRAILS_DEFENDER_IMPLEMENTATION), None)
    if defender is None:
        return []
    judge = planned_judge(session.target.agent_relay_plugins, defender.config.get("safety_llm"))
    return [
        ModelUse(
            "guardrail judge",
            judge.model,
            judge.base_url or judge_endpoint(),
            _victim_env(session).get(judge.api_key_env),
            judge.api_key_env,
            extra_body=no_reasoning_body(judge.model),
        )
    ]


def _victim_env(session: SessionConfig) -> Mapping[str, str]:
    """The variables the judge reads its key from: in a sandbox, only the credentials it was given."""
    control = session.victim_control
    if control.type == "openshell":
        return sandbox_credentials(openshell_config(control))
    return {**os.environ, **judge_credentials()}


def _attacker_uses(session: SessionConfig) -> list[ModelUse]:
    return [
        use
        for attacker in session.attackers
        if attacker.implementation == AGENT_BREAKER_IMPLEMENTATION
        for use in model_uses(
            ScanOptions.from_agent(attacker.config, session.target.base_url).garak_config(DEFAULT_REPORT_DIR)
        )
    ]
