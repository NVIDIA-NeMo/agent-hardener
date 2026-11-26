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

from agent_hardener.agents.attackers.garak_probes import AGENT_BREAKER, GarakProbe, split_plugin
from agent_hardener.endpoint import EndpointContract
from agent_hardener.relay_plugin.config import no_reasoning_body

#: ``probes.<module>.<Class>`` — naming the class explicitly runs it even though the probe is
#: ``active = False`` (only *modules* in ``probe_spec`` honour the active flag).
PROBE_SPEC = AGENT_BREAKER.probe
DETECTOR_SPEC = AGENT_BREAKER.detector

# Baked-in defaults mirror the historical scan_agent_breaker.yaml; each is env-overridable.
DEFAULT_RED_TEAM_MODEL_TYPE = "nim.NVOpenAIChat"
DEFAULT_RED_TEAM_MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_DETECTOR_MODEL_TYPE = "nim"
DEFAULT_DETECTOR_MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_MODEL_URI = "https://integrate.api.nvidia.com/v1/"
DEFAULT_MAX_ATTEMPTS_PER_TOOL = 5
DEFAULT_GENERATIONS = 1
#: Cap on a single attack turn. Override per-run via the manifest's ``garak.request_timeout``.
#: High because DeepAgents compiles with ``recursion_limit = 9_999``; it absorbs the loop rather
#: than bounding it. Once a *released* nemo-fabric carries #268, bound turns with
#: ``runtime.max_turns`` in the victim's ``agent.yaml`` and leave this as a backstop.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
#: Cap on one red-team or detector model call. Replaces garak's 60s, which reasoning models outrun.
DEFAULT_MODEL_TIMEOUT_SECONDS = 600
# Empty by default: see the skip_codes comment in build_agent_breaker_config.
DEFAULT_SKIP_CODES: list[int] = []
#: The probe parameter naming the victim profile it attacks, relative to garak's data directories.
AGENT_CONFIG_PARAM = "agent_config_file"

_SUPPRESSED_PARAMS = ["stop", "top_p", "frequency_penalty", "presence_penalty", "seed"]
#: Match garak's AgentBreakerResult default so its structured verdict is not truncated.
_MODEL_MAX_TOKENS = 8192
#: Every AgentBreaker prompt (discovery parse, tool analysis, exploit, verify) asks for a raw JSON object,
#: and the attack payloads it embeds are full of quotes and backslashes the model fails to escape. Left
#: free-form, a tool analysis that does not parse leaves garak with nothing to attack and it exits 0
#: having sent no attack. JSON mode makes the endpoint return valid JSON instead.
_JSON_RESPONSE_FORMAT = {"type": "json_object"}
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def _model_config(uri: str, model_name: str, timeout: int, *, json_mode: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "uri": uri,
        "max_tokens": _MODEL_MAX_TOKENS,
        "timeout": timeout,
        "suppressed_params": list(_SUPPRESSED_PARAMS),
    }
    extra_params: dict[str, Any] = {}
    if json_mode:
        extra_params["response_format"] = dict(_JSON_RESPONSE_FORMAT)
    if (body := no_reasoning_body(model_name)) is not None:
        extra_params["extra_body"] = body
    if extra_params:
        config["extra_params"] = extra_params
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
    "model_timeout",
    "json_mode",
)


def _resolve(key: str, default: Any, overrides: dict[str, Any]) -> Any:
    """Resolve one setting: explicit override wins, then ``GARAK_<KEY>`` env, then default."""
    value = overrides.get(key)
    if value is not None:
        return value
    env_value = os.environ.get(f"GARAK_{key.upper()}")
    return env_value if env_value else default


def resolve_json_mode(overrides: dict[str, Any] | None = None) -> bool:
    """Whether garak's models are asked for JSON mode; on unless ``json_mode``/``GARAK_JSON_MODE`` turns it off.

    The escape hatch is for an endpoint that rejects ``response_format``.
    """
    value = _resolve("json_mode", True, overrides or {})
    return str(value).strip().lower() not in _FALSE_STRINGS


def resolve_model_timeout(overrides: dict[str, Any] | None = None) -> int | None:
    """Return the explicitly configured model timeout (override, then env), or ``None`` when unset."""
    value = _resolve("model_timeout", None, overrides or {})
    return None if value is None else int(value)


def build_agent_breaker_config(
    *,
    target_uri: str,
    report_dir: str,
    report_prefix: str,
    overrides: dict[str, Any] | None = None,
    contract: EndpointContract | None = None,
    probe: GarakProbe = AGENT_BREAKER,
) -> dict[str, Any]:
    """Build the full garak run-config dict for ``probe`` (the agent_breaker probe by default).

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
        probe: The garak probe (and its detector) the config runs.

    Returns:
        A config mapping ready to ``yaml.safe_dump`` and pass to ``garak --config``.
    """
    overrides = overrides or {}
    contract = contract or EndpointContract(url=target_uri)
    max_attempts = int(_resolve("max_attempts_per_tool", DEFAULT_MAX_ATTEMPTS_PER_TOOL, overrides))
    red_team_model = _resolve("red_team_model_name", DEFAULT_RED_TEAM_MODEL_NAME, overrides)
    detector_model = _resolve("detector_model_name", DEFAULT_DETECTOR_MODEL_NAME, overrides)
    generations = int(_resolve("generations", DEFAULT_GENERATIONS, overrides))
    model_timeout = resolve_model_timeout(overrides) or DEFAULT_MODEL_TIMEOUT_SECONDS
    json_mode = resolve_json_mode(overrides)
    probe_module, probe_class = split_plugin(probe.probe)
    detector_module, detector_class = split_plugin(probe.detector)
    return {
        "run": {"generations": generations},
        "plugins": {
            "target_type": "rest.RestGenerator",
            "target_name": "personal-assistant",
            "probe_spec": probe.probe,
            "extended_detectors": [probe.detector],
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
                probe_module: {
                    probe_class: {
                        "max_attempts_per_tool": max_attempts,
                        "red_team_model_type": _resolve("red_team_model_type", DEFAULT_RED_TEAM_MODEL_TYPE, overrides),
                        "red_team_model_name": red_team_model,
                        "red_team_model_config": _model_config(
                            _resolve("red_team_model_uri", DEFAULT_MODEL_URI, overrides),
                            red_team_model,
                            model_timeout,
                            json_mode=json_mode,
                        ),
                    }
                }
            },
            "detectors": {
                detector_module: {
                    detector_class: {
                        "detector_model_type": _resolve("detector_model_type", DEFAULT_DETECTOR_MODEL_TYPE, overrides),
                        "detector_model_name": detector_model,
                        "detector_model_config": _model_config(
                            _resolve("detector_model_uri", DEFAULT_MODEL_URI, overrides),
                            detector_model,
                            model_timeout,
                            json_mode=json_mode,
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
    model_timeout: int | None = None,
    probe: GarakProbe = AGENT_BREAKER,
    agent_config_file: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``config`` with the resolved per-run fields forced.

    Lets a user-edited scaffold (model settings, probe params) survive while guaranteeing the
    target ``uri`` and reporting destination always match the current run. An explicit
    ``model_timeout`` replaces the scaffold's model timeouts; otherwise a scaffold without one
    gets the default rather than garak's 60s. ``agent_config_file`` (a name relative to garak's
    data directory) points the probe at the run's recon profile; unset, the probe keeps its own.
    """
    resolved = copy.deepcopy(config)
    plugins = resolved.setdefault("plugins", {})
    if agent_config_file is not None:
        module, class_name = split_plugin(probe.probe)
        probe_block = plugins.setdefault("probes", {}).setdefault(module, {}).setdefault(class_name, {})
        probe_block[AGENT_CONFIG_PARAM] = agent_config_file
    generator = plugins.setdefault("generators", {}).setdefault("rest", {}).setdefault("RestGenerator", {})
    generator["uri"] = target_uri
    # A scaffold authored before this setting gets the default rather than an inherited NAT-ism.
    generator.setdefault("skip_codes", list(DEFAULT_SKIP_CODES))
    for model_config in _model_configs(plugins, probe):
        if model_timeout is not None:
            model_config["timeout"] = model_timeout
        else:
            model_config.setdefault("timeout", DEFAULT_MODEL_TIMEOUT_SECONDS)
    resolved["reporting"] = {"report_dir": report_dir, "report_prefix": report_prefix}
    return resolved


def _model_configs(plugins: dict[str, Any], probe: GarakProbe) -> list[dict[str, Any]]:
    candidates = (
        probe.plugin_config(plugins, "probes").get("red_team_model_config"),
        probe.plugin_config(plugins, "detectors").get("detector_model_config"),
    )
    return [config for config in candidates if isinstance(config, dict)]
