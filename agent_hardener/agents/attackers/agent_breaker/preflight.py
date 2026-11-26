# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check garak's red-team and detector models answer before the scan starts.

garak retries a failed model call with no limit on attempts, so a dead or hung model endpoint runs the
scan into ``garak.timeout_s`` and reports 0 attempts. One request per model, with the same endpoint,
credential, request body and timeout garak will use, turns that into a clear error up front.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_hardener.agents.attackers.agent_breaker.config import DEFAULT_MODEL_TIMEOUT_SECONDS
from agent_hardener.env import NIM_API_KEY
from agent_hardener.garak_venv import garak_subprocess_env
from agent_hardener.model_check import ModelUse, require_answers


async def check_model_endpoints(config: dict[str, Any]) -> None:
    """Raise :class:`~agent_hardener.errors.ModelUnavailableError` when a garak NIM model won't answer."""
    await asyncio.to_thread(require_answers, model_uses(config))


def model_uses(config: dict[str, Any]) -> list[ModelUse]:
    """The NIM models a resolved garak config calls, each as garak will call it."""
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
    api_key = garak_subprocess_env().get(NIM_API_KEY)
    uses = []
    for label, model_type, model, model_config in roles:
        if not str(model_type or "").startswith("nim") or not model or not isinstance(model_config, dict):
            continue
        uri = model_config.get("uri")
        if not uri:
            continue
        uses.append(
            ModelUse(
                label,
                str(model),
                str(uri),
                api_key,
                NIM_API_KEY,
                timeout=float(model_config.get("timeout") or DEFAULT_MODEL_TIMEOUT_SECONDS),
                extra_body=_request_fields(model_config.get("extra_params")),
                timeout_setting="garak.model_timeout",
            )
        )
    return uses


def _request_fields(extra_params: Any) -> dict[str, Any] | None:
    """The request-body fields garak's ``extra_params`` add, flattened the way the OpenAI client sends them.

    ``extra_body`` is merged into the body; every other key (``response_format`` in JSON mode) is a body field
    of its own. Probing with all of them is what makes the check exercise the request garak will send: an
    endpoint that rejects JSON mode fails here, with its own message, instead of on every call in the scan.
    """
    if not isinstance(extra_params, dict):
        return None
    fields = {key: value for key, value in extra_params.items() if key != "extra_body"}
    fields.update(extra_params.get("extra_body") or {})
    return fields or None
