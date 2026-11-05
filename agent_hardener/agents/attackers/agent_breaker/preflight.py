# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check garak's red-team and detector models answer before the scan starts.

garak retries a failed model call with no limit on attempts, so a dead or hung model endpoint runs the
scan into ``garak.timeout_s`` and reports 0 attempts. One request per model, with the same endpoint,
credential, request body and timeout garak will use, turns that into a clear error up front.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from agent_hardener.agents.attackers.agent_breaker.config import DEFAULT_MODEL_TIMEOUT_SECONDS
from agent_hardener.env import NIM_API_KEY
from agent_hardener.garak_venv import garak_subprocess_env
from agent_hardener.model_check import probe_generation, raise_if_unavailable


@dataclass(frozen=True)
class _ModelCall:
    label: str
    model: str
    uri: str
    timeout: float
    extra_body: dict[str, Any] | None


async def check_model_endpoints(config: dict[str, Any]) -> None:
    """Raise :class:`~agent_hardener.errors.ModelUnavailableError` when a garak NIM model won't answer."""
    calls = _model_calls(config)
    api_key = garak_subprocess_env().get(NIM_API_KEY)
    verdicts = await asyncio.gather(
        *(
            asyncio.to_thread(
                probe_generation, call.model, call.uri, api_key, timeout=call.timeout, extra_body=call.extra_body
            )
            for call in calls
        )
    )
    for call, verdict in zip(calls, verdicts, strict=True):
        raise_if_unavailable(call.label, call.model, call.uri, verdict)


def _model_calls(config: dict[str, Any]) -> list[_ModelCall]:
    plugins = config.get("plugins") or {}
    probe = ((plugins.get("probes") or {}).get("agent_breaker") or {}).get("AgentBreaker") or {}
    detector = ((plugins.get("detectors") or {}).get("agent_breaker") or {}).get("AgentBreakerResult") or {}
    roles = (
        (
            "red-team",
            probe.get("red_team_model_type"),
            probe.get("red_team_model_name"),
            probe.get("red_team_model_config"),
        ),
        (
            "detector",
            detector.get("detector_model_type"),
            detector.get("detector_model_name"),
            detector.get("detector_model_config"),
        ),
    )
    calls: dict[tuple[str, str], _ModelCall] = {}
    for label, model_type, model, model_config in roles:
        if not str(model_type or "").startswith("nim") or not model or not isinstance(model_config, dict):
            continue
        uri = model_config.get("uri")
        if not uri:
            continue
        key = (str(uri), str(model))
        if key in calls:
            calls[key] = replace(calls[key], label=f"{calls[key].label} and {label}")
            continue
        calls[key] = _ModelCall(
            label=label,
            model=str(model),
            uri=str(uri),
            timeout=float(model_config.get("timeout") or DEFAULT_MODEL_TIMEOUT_SECONDS),
            extra_body=(model_config.get("extra_params") or {}).get("extra_body"),
        )
    return list(calls.values())
