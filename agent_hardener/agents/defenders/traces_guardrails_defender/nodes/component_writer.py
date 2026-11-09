# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append the rail to the run's NeMo Relay plugin config.

One component entry of kind :data:`~agent_hardener.relay_plugin.config.PLUGIN_KIND` holds every rail
(``custom_guardrail_N``), each guarding the tool the grounded analysis chose, judged by the defender's
safety model at the shared threshold.
"""

from __future__ import annotations

import re
import threading
import tomllib
from typing import TYPE_CHECKING, Any

import tomli_w

from agent_hardener.llm import DEFAULT_MODEL
from agent_hardener.models import DefenderOutput
from agent_hardener.relay_plugin.config import (
    DEFAULT_GUARDRAIL_THRESHOLD,
    PLUGIN_KIND,
    GuardrailsComponentConfig,
    judge_api_key_env,
    judge_endpoint,
)

from ..schemas import Outcome
from ..state import DefenderState  # noqa: TC001 - LangGraph reads node annotations at runtime

if TYPE_CHECKING:
    from pathlib import Path


def component_writer_node(state: DefenderState) -> dict[str, Any]:
    """Append the chosen rail to the run's plugin config."""
    rail, analysis = state.rail, state.vulnerability_analysis
    if rail is None or analysis is None:  # unreachable: routing only sends generated rails here
        msg = "component_writer_node needs a rail"
        raise ValueError(msg)
    name, rendered = append_guardrail(
        state.input.relay_plugins_path,
        tool=rail.target_tool,
        instructions=rail.system_instructions,
        attack_id=state.attack,
        judge_model=state.input.context.get("safety_llm"),
    )
    grade = next((g for g in state.grades if g.iteration == rail.iteration), None)
    outcome = Outcome(
        attack=state.attack,
        status="applied",
        route="tool",
        fallback_reason=analysis.fallback_reason,
        tool_name=rail.target_tool,
        guardrail_name=name,
        grader_score=grade.score if grade else None,
        grader_iterations=len(state.drafts),
        lint_warnings=rail.lint_warnings,
    )
    output = DefenderOutput(
        ok=True,
        new_policy_yaml=rendered,
        iteration_count=len(state.drafts),
        resource_type="relay_guardrail_component",
        guardrail_name=name,
    )
    return {"outcome": outcome, "output": output}


#: The manager runs one defender graph per attack concurrently, all extending the same file.
_PLUGINS_LOCK = threading.Lock()

_GUARDRAIL_NAME_RE = re.compile(r"^custom_guardrail_(\d+)$")


def append_guardrail(
    target: Path | None,
    *,
    tool: str,
    instructions: str,
    attack_id: str | None = None,
    judge_model: str | None = None,
) -> tuple[str, str]:
    """Add one ``custom_guardrail_N`` entry guarding ``tool`` and return its name and the rendered file.

    ``judge_model`` (the defender's ``safety_llm``) is the in-victim judge, written only when the file has
    no judge model yet; ``None`` means :data:`agent_hardener.llm.DEFAULT_MODEL`.
    """
    with _PLUGINS_LOCK:
        document = _load(target)
        config = _component_config(document, judge_model)
        guardrails = config.setdefault("guardrails", [])
        name = f"custom_guardrail_{_next_index(guardrails)}"
        guardrail: dict[str, Any] = {
            "name": name,
            "target_tool": tool,
            "action": "refusal",
            "threshold": DEFAULT_GUARDRAIL_THRESHOLD,
            "system_instructions": instructions,
        }
        if attack_id:
            guardrail["attack_id"] = attack_id
        guardrails.append(guardrail)
        rendered = tomli_w.dumps(document)
        GuardrailsComponentConfig.model_validate(config)
        if target is not None:
            target.write_text(rendered, encoding="utf-8")
    return name, rendered


def _load(target: Path | None) -> dict[str, Any]:
    if target is not None and target.exists():
        with target.open("rb") as handle:
            return tomllib.load(handle)
    return {"version": 1}


def _component_config(document: dict[str, Any], judge_model: str | None) -> dict[str, Any]:
    components = document.setdefault("components", [])
    for entry in components:
        if isinstance(entry, dict) and entry.get("kind") == PLUGIN_KIND:
            config: dict[str, Any] = entry.setdefault("config", {})
            config.setdefault("model", _judge_model(judge_model))
            return config
    fresh: dict[str, Any] = {"model": _judge_model(judge_model)}
    components.append({"kind": PLUGIN_KIND, "enabled": True, "config": fresh})
    return fresh


def _judge_model(model: str | None) -> dict[str, Any]:
    # Only the variable name of the key is written: this file is uploaded into the victim image.
    return {
        "provider": "nvidia",
        "api_key_env": judge_api_key_env(),
        # The sandbox allow-list and the judge both read judge_endpoint(), so they cannot disagree.
        "model": model or DEFAULT_MODEL,
        "base_url": judge_endpoint(),
    }


def _next_index(guardrails: list[Any]) -> int:
    highest = 0
    for entry in guardrails:
        match = _GUARDRAIL_NAME_RE.match(str(entry.get("name", ""))) if isinstance(entry, dict) else None
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1
