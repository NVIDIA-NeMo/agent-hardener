# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the garak ``agent_breaker`` run-config dynamically.

Replaces the static ``scan_agent_breaker.yaml`` that the retired REST app shipped. The probe
is driven straight from the garak CLI (``garak --config <file>.yaml``); only the target
``uri`` and the reporting destination are truly dynamic, so :func:`build_agent_breaker_config`
bakes the historical model defaults (overridable via ``GARAK_*`` env vars or explicit
``overrides``) and :func:`apply_runtime_fields` overlays the resolved per-run values onto a
(possibly user-edited) base config.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from agent_hardener.endpoint import EndpointContract
from agent_hardener.relay_plugin.config import no_reasoning_body

#: ``probes.<module>.<Class>`` — naming the class explicitly runs it even though the probe is
#: ``active = False`` (only *modules* in ``probe_spec`` honour the active flag).
PROBE_SPEC = "agent_breaker.AgentBreaker"
DETECTOR_SPEC = "agent_breaker.AgentBreakerResult"

# Baked-in defaults mirror the historical scan_agent_breaker.yaml; each is env-overridable.
DEFAULT_RED_TEAM_MODEL_TYPE = "nim.NVOpenAIChat"
DEFAULT_RED_TEAM_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_DETECTOR_MODEL_TYPE = "nim"
DEFAULT_DETECTOR_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_MODEL_URI = "https://integrate.api.nvidia.com/v1/"
DEFAULT_MAX_ATTEMPTS_PER_TOOL = 5
DEFAULT_GENERATIONS = 1
#: Cap on a single attack turn. Override per-run via the manifest's ``garak.request_timeout``.
#: High because DeepAgents compiles with ``recursion_limit = 9_999``; it absorbs the loop rather
#: than bounding it. Once a *released* nemo-fabric carries #268, bound turns with
#: ``runtime.max_turns`` in the victim's ``agent.yaml`` and leave this as a backstop.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
# Empty by default: see the skip_codes comment in build_agent_breaker_config.
DEFAULT_SKIP_CODES: list[int] = []

_SUPPRESSED_PARAMS = ["stop", "top_p", "frequency_penalty", "presence_penalty", "seed"]


def _model_config(uri: str, model_name: str, max_tokens: int) -> dict[str, Any]:
    config: dict[str, Any] = {"uri": uri, "max_tokens": max_tokens, "suppressed_params": list(_SUPPRESSED_PARAMS)}
    if (body := no_reasoning_body(model_name)) is not None:
        config["extra_params"] = {"extra_body": body}
    return config


#: Override keys accepted (from ``overrides`` or, upper-cased with a ``GARAK_`` prefix, env).
OVERRIDE_KEYS = (
    "red_team_model_type",
    "red_team_model_name",
    "red_team_model_uri",
    "detector_model_type",
    "detector_model_name",
    "detector_model_uri",
    "max_attempts_per_tool",
    "generations",
    "request_timeout",
)


def _resolve(key: str, default: Any, overrides: dict[str, Any]) -> Any:
    """Resolve one setting: explicit override wins, then ``GARAK_<KEY>`` env, then default."""
    value = overrides.get(key)
    if value is not None:
        return value
    env_value = os.environ.get(f"GARAK_{key.upper()}")
    return env_value if env_value else default


def build_agent_breaker_config(
    *,
    target_uri: str,
    report_dir: str,
    report_prefix: str,
    overrides: dict[str, Any] | None = None,
    contract: EndpointContract | None = None,
) -> dict[str, Any]:
    """Build the full garak run-config dict for the agent_breaker probe.

    Args:
        target_uri: Victim chat-completions endpoint the RestGenerator attacks.
        report_dir: Directory garak writes ``*.report.jsonl`` / ``*.hitlog.jsonl`` into.
        report_prefix: Filename prefix garak stamps on this run's artifacts.
        overrides: Optional per-setting overrides (see :data:`OVERRIDE_KEYS`); each also
            falls back to ``GARAK_<KEY>`` then a baked-in default.
        contract: How this victim's endpoint is shaped. Defaults to OpenAI chat-completions at
            ``target_uri``. Passing the run's own contract is what keeps the attacker and the
            validators' replay sending the same request — otherwise a "blocked" verdict could just
            mean the two payload builders disagreed.

    Returns:
        A config mapping ready to ``yaml.safe_dump`` and pass to ``garak --config``.
    """
    overrides = overrides or {}
    contract = contract or EndpointContract(url=target_uri)
    max_attempts = int(_resolve("max_attempts_per_tool", DEFAULT_MAX_ATTEMPTS_PER_TOOL, overrides))
    red_team_model = _resolve("red_team_model_name", DEFAULT_RED_TEAM_MODEL_NAME, overrides)
    detector_model = _resolve("detector_model_name", DEFAULT_DETECTOR_MODEL_NAME, overrides)
    generations = int(_resolve("generations", DEFAULT_GENERATIONS, overrides))
    return {
        "run": {"generations": generations},
        "plugins": {
            "target_type": "rest.RestGenerator",
            "target_name": "personal-assistant",
            "probe_spec": PROBE_SPEC,
            "extended_detectors": [DETECTOR_SPEC],
            "generators": {
                "rest": {
                    "RestGenerator": {
                        "uri": target_uri,
                        "headers": {"Content-Type": "application/json"},
                        **contract.garak_rest_fields(),
                        # One attack turn, not the whole scan (that cap is `garak.timeout_s`). Sized
                        # for an agent that can call a sub-agent: measured on a healthy DeepAgents
                        # victim, a `read_customer_record` turn takes 12s, `write_file` 34s, and a
                        # `task` turn — which spawns a sub-agent — 141s. The validators replay at
                        # concurrency 2, so two such turns share the victim and its model endpoint.
                        # At 240s a single `task` attack timed out and failed the whole attacker,
                        # scoring every remaining attack as unrun.
                        "request_timeout": _resolve("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS, overrides),
                        "max_tokens": 4096,
                        # Which status codes mean "the victim broke" is the victim's business, not
                        # ours: Agent Hardener no longer owns the server. Default to skipping nothing so a
                        # real failure is visible, and let a manifest declare its own codes. (A
                        # guardrail refusal is not one of these — it comes back 200 with the refusal
                        # in the response text, because the tool error is handled inside the agent.)
                        "skip_codes": list(_resolve("skip_codes", DEFAULT_SKIP_CODES, overrides)),
                    }
                }
            },
            "probes": {
                "agent_breaker": {
                    "AgentBreaker": {
                        "max_attempts_per_tool": max_attempts,
                        "red_team_model_type": _resolve("red_team_model_type", DEFAULT_RED_TEAM_MODEL_TYPE, overrides),
                        "red_team_model_name": red_team_model,
                        "red_team_model_config": _model_config(
                            _resolve("red_team_model_uri", DEFAULT_MODEL_URI, overrides), red_team_model, 8192
                        ),
                    }
                }
            },
            "detectors": {
                "agent_breaker": {
                    "AgentBreakerResult": {
                        "detector_model_type": _resolve("detector_model_type", DEFAULT_DETECTOR_MODEL_TYPE, overrides),
                        "detector_model_name": detector_model,
                        "detector_model_config": _model_config(
                            _resolve("detector_model_uri", DEFAULT_MODEL_URI, overrides), detector_model, 1024
                        ),
                    }
                }
            },
        },
        "reporting": {"report_dir": report_dir, "report_prefix": report_prefix},
    }


def apply_runtime_fields(
    config: dict[str, Any],
    *,
    target_uri: str,
    report_dir: str,
    report_prefix: str,
) -> dict[str, Any]:
    """Return a copy of ``config`` with the resolved per-run fields forced.

    Lets a user-edited scaffold (model settings, probe params) survive while guaranteeing the
    target ``uri`` and reporting destination always match the current run.
    """
    resolved = copy.deepcopy(config)
    plugins = resolved.setdefault("plugins", {})
    generator = plugins.setdefault("generators", {}).setdefault("rest", {}).setdefault("RestGenerator", {})
    generator["uri"] = target_uri
    # A scaffold authored before this setting gets the default rather than an inherited NAT-ism.
    generator.setdefault("skip_codes", list(DEFAULT_SKIP_CODES))
    resolved["reporting"] = {"report_dir": report_dir, "report_prefix": report_prefix}
    return resolved
