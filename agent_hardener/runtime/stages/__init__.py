# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stages the orchestrator runs each round.

``agents/{role}/`` holds agent *implementations*; this package holds how the orchestrator *runs*
each pipeline step. Each stage takes the per-round :class:`~agent_hardener.runtime.run_recorder.RunRecorder`
and owns its own events + artifact writes; the agent stages additionally share one single-agent
invoker (:class:`AgentInvoker`) by composition — there is no shared base class, because the stages
have different shapes (attack = fan-out, defense = LLM routing, validation = typed per-agent fan-out,
victim = single call) and are never used polymorphically. ``DefenseStage`` *uses*
:class:`~agent_hardener.agents.defenders.defenders_manager.DefendersManager` (which stays in
``agents/defenders/``); deployment is the one non-agent step (:class:`DeployStage`).
"""

from agent_hardener.runtime.stages.agent_invoker import AgentInvoker
from agent_hardener.runtime.stages.attack import AttackStage
from agent_hardener.runtime.stages.context import DefenseResult, IterationContext
from agent_hardener.runtime.stages.defense import DefenseStage
from agent_hardener.runtime.stages.deploy import DeployStage
from agent_hardener.runtime.stages.validation import ValidationStage
from agent_hardener.runtime.stages.victim import VictimStage

__all__ = [
    "AgentInvoker",
    "AttackStage",
    "DefenseResult",
    "DefenseStage",
    "DeployStage",
    "IterationContext",
    "ValidationStage",
    "VictimStage",
]
