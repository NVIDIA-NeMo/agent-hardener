# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The component config the guardrails defender emits and the plugin consumes.

This is the wire format between a defender running on the host and the plugin running inside the
victim, so it is a schema rather than a convention: a malformed guardrail must be rejected at
``validate()`` with a diagnostic, not discovered when an attack is already in flight.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Nemotron reasons by default and can spend its whole token budget thinking, which leaves callers an
#: empty or very slow reply. Other providers may reject this field, so only Nemotron gets it.
_NEMOTRON_NO_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def no_reasoning_body(model: str) -> dict[str, Any] | None:
    """Request-body fields that turn reasoning off for ``model``, or ``None`` when not applicable."""
    return dict(_NEMOTRON_NO_THINKING) if "nemotron" in model.lower() else None


#: The kind the victim registers with Relay, and the ``kind`` of the ``[[components]]`` entry
#: the guardrails defender writes. The two must match or the component is inert.
PLUGIN_KIND = "agent_hardener.pre_tool_verifier"


#: The safety judge's own endpoint and credential; unset falls back to the analysis endpoint and key.
SAFETY_BASE_URL_ENV = "AGENT_HARDENER_SAFETY_BASE_URL"
SAFETY_API_KEY_ENV = "AGENT_HARDENER_SAFETY_API_KEY"  # pragma: allowlist secret
DEFAULT_JUDGE_API_KEY_ENV = "INFERENCE_API_KEY"  # pragma: allowlist secret


def judge_endpoint() -> str:
    """Where the in-victim safety judge sends its scoring calls.

    ``AGENT_HARDENER_SAFETY_BASE_URL`` when set, else the endpoint the rest of the run's analysis
    models use. It is resolved here rather than left unset because an unset ``base_url`` sends the
    judge to OpenAI's default host — unreachable from the sandbox and wrong for the model it is given.

    The sandbox allow-lists this host explicitly (``agent_hardener.tools.openshell``): the judge runs
    inside the victim, and egress discovery cannot find an endpoint Agent Hardener itself injected.
    """
    return (
        os.environ.get(SAFETY_BASE_URL_ENV)
        or os.environ.get("AGENT_HARDENER_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    )


def judge_api_key_env() -> str:
    """Keep the judge's resolved key separate from the victim's own inference credential."""
    return SAFETY_API_KEY_ENV if judge_credentials() else DEFAULT_JUDGE_API_KEY_ENV


def judge_credentials() -> dict[str, str]:
    """Forward the explicit or inherited judge key under its own sandbox credential name."""
    value = os.environ.get(SAFETY_API_KEY_ENV) or os.environ.get(DEFAULT_JUDGE_API_KEY_ENV)
    return {SAFETY_API_KEY_ENV: value} if value else {}


#: Judge scores at or above this refuse. One value for the rail the defender deploys and for the
#: scoring contract its prompts describe.
DEFAULT_GUARDRAIL_THRESHOLD = 0.7


class JudgeModel(BaseModel):
    """The LLM that decides whether a tool call is the attack or ordinary use."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    provider: str = "nvidia"
    base_url: str | None = None
    #: Name of the env var holding the key — never the key itself. This file is uploaded into the
    #: victim image and rendered in run reports.
    api_key_env: str = DEFAULT_JUDGE_API_KEY_ENV


class Guardrail(BaseModel):
    """One tool the war-game hardened, and the instructions that decide when to refuse it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    target_tool: str = Field(min_length=1)
    system_instructions: str = Field(min_length=1)
    action: Literal["refusal"] = "refusal"
    #: Judge scores at or above this refuse. Higher is more permissive.
    threshold: float = Field(default=DEFAULT_GUARDRAIL_THRESHOLD, ge=0.0, le=1.0)
    #: The attack that motivated this guardrail, so the run report and Studio can pair them.
    attack_id: str | None = None


#: What the agent answers once a guardrail has refused. Deliberately generic: it is the only thing
#: an attacker sees, so it must not leak which rail fired or what the judge scored. The detailed
#: refusal still reaches telemetry through :func:`agent_hardener.relay_plugin.policy._refusal`.
DEFAULT_BLOCKED_MESSAGE = "The request was blocked by security policy."


class GuardrailsComponentConfig(BaseModel):
    """The whole ``[components.config]`` table."""

    model_config = ConfigDict(extra="forbid")

    model: JudgeModel
    guardrails: list[Guardrail] = Field(default_factory=list)
    #: The agent's final answer after a refusal. Safe to add under ``extra="forbid"`` because the
    #: plugin is copied into the victim image from this same tree
    #: (``agent_hardener.openshell.relay_victim.GUARDRAIL_PACKAGE``), so the defender that writes
    #: this key and the plugin that reads it are always the same version.
    blocked_message: str = Field(default=DEFAULT_BLOCKED_MESSAGE, min_length=1)


def configured_guardrails(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every guardrail across the component entries of a Relay plugins.toml document.

    Tolerant by design: it reads files the defenders wrote *and* files an operator hand-edited, and
    both the run report and the console renderer call it while a run is in flight. A malformed entry
    is skipped rather than raised on, because the schema check that matters happens where the
    guardrail is written (:mod:`agent_hardener.agents.defenders.guardrails_defender_v2.nodes.component_writer`)
    and again where it is loaded, inside the victim.
    """
    components = document.get("components")
    if not isinstance(components, list):
        return []
    return [
        rail
        for entry in components
        if isinstance(entry, dict) and isinstance(entry.get("config"), dict)
        for rail in entry["config"].get("guardrails", [])
        if isinstance(rail, dict)
    ]
