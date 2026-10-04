# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the generated guardrail into the run's NeMo Relay plugin configuration.

Replaces the NAT-era workflow patcher. That one did a read-modify-write on the victim's *own*
config, injecting a ``pre_tool_verifier`` middleware block plus a reference in the attacked tool's
``middleware:`` list — two places that had to stay in sync, where a dangling reference failed the
victim's config validation and stopped it serving.

Here each guardrail is one self-contained table in a file Agent Hardener owns, so there is nothing to
cross-reference and nothing of the user's to corrupt.
"""

from __future__ import annotations

import re
import threading
import tomllib
from typing import TYPE_CHECKING, Any

import tomli_w

from agent_hardener.llm import DEFAULT_MODEL
from agent_hardener.relay_plugin.config import PLUGIN_KIND, GuardrailsComponentConfig, judge_endpoint

from ..schemas import DefenderOutput, GraphState

if TYPE_CHECKING:
    from pathlib import Path

# The manager fans out one guardrails-defender graph per attack, each in its own thread (the defense
# stage dispatches the sync run via ``asyncio.to_thread``). Every graph ends here doing a
# read-modify-write on the SAME plugins.toml, so serialize the whole read→append→save: concurrent
# writes would otherwise interleave and lose a guardrail.
_PLUGINS_LOCK = threading.Lock()

#: The plugin kind the victim registers before ``nemo_relay.plugin.initialize()``. A component
#: naming a kind nothing registered is inert, so this string is the whole contract between the file
#: the defender writes here and the code running inside the victim.
COMPONENT_KIND = PLUGIN_KIND

_GUARDRAIL_NAME_RE = re.compile(r"^custom_guardrail_(\d+)$")


def component_writer_node(state: GraphState) -> dict[str, Any]:
    """Append this attack's guardrail to the run's plugin config.

    Writes nothing when the attack could not be attributed to a tool. A guardrail with no target
    guards nothing, but would still appear in the run report as a mitigation — a false claim about
    what the round achieved. Abstaining says the attack was not attributable, which is true.
    """
    if not state.original_input.attacked_tool:
        return {
            "final_output": DefenderOutput(
                ok=False,
                error_message=(
                    "the attack prompt names no tool and the attack record carries none, so there is nothing to guard"
                ),
                iteration_count=state.iteration_count,
            )
        }

    target = state.relay_plugins_path
    with _PLUGINS_LOCK:
        document = _load(target, state.current_config)
        name = _append_guardrail(document, state)
        rendered = _save(document, target)

    output = DefenderOutput(
        ok=True,
        new_policy_yaml=rendered,
        iteration_count=state.iteration_count,
        resource_type="relay_guardrail_component",
        guardrail_name=name,
    )
    return {"current_config": rendered, "final_output": output}


def _load(target: Path | None, current: str) -> dict[str, Any]:
    """The plugin document to extend: the run's file, the in-memory copy, or a fresh one."""
    if target is not None and target.exists():
        with target.open("rb") as handle:
            return tomllib.load(handle)
    if current:
        return tomllib.loads(current)
    return {"version": 1}


def _append_guardrail(document: dict[str, Any], state: GraphState) -> str:
    """Add one ``custom_guardrail_N`` entry and return the name it was given."""
    config = _component_config(document, state)
    guardrails = config.setdefault("guardrails", [])
    name = f"custom_guardrail_{_next_index(guardrails)}"
    guardrail: dict[str, Any] = {
        "name": name,
        "target_tool": state.original_input.attacked_tool,
        "action": "refusal",
        "threshold": 0.7,
        "system_instructions": state.draft_instruction,
    }
    attack_id = state.original_input.context.get("attack_id")
    if attack_id:
        guardrail["attack_id"] = str(attack_id)
    guardrails.append(guardrail)
    return name


def _component_config(document: dict[str, Any], state: GraphState) -> dict[str, Any]:
    """The ``[components.config]`` table for our plugin kind, created on first use.

    A top-level ``[[components]]`` entry rather than a ``[[plugins.dynamic]]`` one: dynamic plugins
    are activated from CLI-managed lifecycle state (``nemo-relay plugins add`` provisions an
    environment and records an ``environment_ref``), which a file Agent Hardener uploads cannot supply.
    A statically-registered kind reads its config straight from the discovered file, which is the
    only delivery this harness controls.
    """
    components = document.setdefault("components", [])
    for entry in components:
        if isinstance(entry, dict) and entry.get("kind") == COMPONENT_KIND:
            config: dict[str, Any] = entry.setdefault("config", {})
            return config
    fresh: dict[str, Any] = {"model": _judge_model(state)}
    components.append({"kind": COMPONENT_KIND, "enabled": True, "config": fresh})
    return fresh


def _judge_model(state: GraphState) -> dict[str, Any]:
    """The model that scores tool calls.

    Only the model name is the operator's to choose; the endpoint and the *name* of the key env var
    are pinned. The key itself is never written here — this file is uploaded into the victim image
    and rendered in run reports.
    """
    return {
        "provider": "nvidia",
        "api_key_env": "INFERENCE_API_KEY",  # pragma: allowlist secret
        # The operator picks the model; Agent Hardener pins the endpoint. That is the existing contract
        # for the safety model group, and it is what keeps the sandbox allow-list and the judge from
        # disagreeing: both read judge_endpoint(). Move it with AGENT_HARDENER_BASE_URL, which moves the
        # run's other analysis models too.
        "model": state.safety_llm or DEFAULT_MODEL,
        # Never left unset: that sends the judge to OpenAI's default host, which the sandbox does not
        # allow out and which cannot serve the model it is given — and the guardrail fails *open*, so
        # the run would report the attack landing rather than an error.
        "base_url": judge_endpoint(),
    }


def _save(document: dict[str, Any], target: Path | None) -> str:
    """Persist the document and return its text, validating what we just wrote.

    Validating here rather than at the victim turns a malformed guardrail into a failed defender —
    visible in the round — instead of a victim that refuses to start two stages later.
    """
    rendered = tomli_w.dumps(document)
    for entry in document.get("components", []):
        if isinstance(entry, dict) and entry.get("kind") == COMPONENT_KIND:
            GuardrailsComponentConfig.model_validate(entry.get("config", {}))
    if target is not None:
        target.write_text(rendered, encoding="utf-8")
    return rendered


def _next_index(guardrails: list[Any]) -> int:
    """The next free index, so successive rounds append rather than overwrite."""
    highest = 0
    for entry in guardrails:
        match = _GUARDRAIL_NAME_RE.match(str(entry.get("name", ""))) if isinstance(entry, dict) else None
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1
