# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-run model connectivity preflight.

A war-game builds a Docker sandbox and runs for minutes; a mistyped model name or a wrong
endpoint/key should fail in seconds, not surface later as an opaque attacker/validator error. This
module lists the models a credential can reach (``GET {base_url}/models``) and, for the models the
operator has *explicitly* configured via env (the shared analysis model ``AGENT_HARDENER_MODEL`` and the
garak attacker model ``GARAK_RED_TEAM_MODEL_NAME``), verifies the name is served — raising
:class:`~agent_hardener.errors.ModelUnavailableError` with the reachable list when it isn't.

Only operator-overridden models are checked; the built-in defaults are known-good and left alone, so
a default run pays no preflight cost. Endpoints without an OpenAI-compatible ``/models`` list are a
soft pass (reachability + auth still confirmed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from agent_hardener.env import INFERENCE_API_KEY, NIM_API_KEY, inference_api_key
from agent_hardener.errors import ModelUnavailableError
from agent_hardener.llm import DEFAULT_BASE_URL, DEFAULT_MODEL

_PROBE_TIMEOUT_S = 10.0
# garak's attacker/detector default endpoint (mirrors agent_breaker.config.DEFAULT_MODEL_URI).
_ATTACK_DEFAULT_URI = "https://integrate.api.nvidia.com/v1/"


@dataclass(frozen=True)
class Validation:
    """A model choice's verdict; ``available`` lists what the credentials can reach for a helpful error."""

    ok: bool
    reason: str = ""  # "", "auth", "unreachable", "unknown_model"
    available: list[str] = field(default_factory=list)
    detail: str = ""


def validate_choice(model: str | None, base_url: str, api_key: str | None) -> Validation:
    """Probe ``{base_url}/models`` and verify *model* is served (best-effort; never raises)."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return Validation(ok=False, reason="unreachable", detail=str(exc) or exc.__class__.__name__)
    if resp.status_code in (401, 403):
        return Validation(ok=False, reason="auth", detail=f"HTTP {resp.status_code}")
    if resp.status_code == 404 or resp.status_code >= 400:
        # No OpenAI-compatible model list (or another error we can't enumerate from) — soft pass.
        return Validation(ok=True, detail=f"HTTP {resp.status_code}")
    try:
        ids = sorted(str(m["id"]) for m in resp.json().get("data", []) if isinstance(m, dict) and m.get("id"))
    except (ValueError, KeyError, TypeError):
        return Validation(ok=True, detail="unparseable /models response")
    if model and model not in ids:
        return Validation(ok=False, reason="unknown_model", available=ids)
    return Validation(ok=True, available=ids)


def _raise_if_bad(label: str, model: str | None, base_url: str, verdict: Validation) -> None:
    """Turn a failed :class:`Validation` into a :class:`ModelUnavailableError` (no-op when ok)."""
    if verdict.ok:
        return
    if verdict.reason == "auth":
        raise ModelUnavailableError(f"The {label} model credentials were rejected by {base_url} ({verdict.detail}).")
    if verdict.reason == "unreachable":
        raise ModelUnavailableError(f"Could not reach the {label} model endpoint {base_url} ({verdict.detail}).")
    available = ", ".join(verdict.available[:20]) or "none"
    raise ModelUnavailableError(
        f"The {label} model {model!r} is not available at {base_url}. Reachable models: {available}."
    )


def preflight_configured_models() -> None:
    """Validate credentials and operator-overridden models before the sandbox spins up.

    A missing inference key would otherwise surface mid-run as an opaque per-agent LLM auth error, so
    the default (NVIDIA-hosted) endpoint's credential is required up front. A custom ``AGENT_HARDENER_BASE_URL``
    may point at a keyless local model, so the presence check is skipped there (auth is still probed
    below for any overridden model name). Built-in default model *names* are known-good and left alone.
    """
    if not inference_api_key() and not os.environ.get("AGENT_HARDENER_BASE_URL"):
        raise ModelUnavailableError(
            f"{INFERENCE_API_KEY} is not set — Agent Hardener has no credential for its default models.",
            remediation=(
                f"Set {INFERENCE_API_KEY} in your --env-file or environment "
                "(or point AGENT_HARDENER_BASE_URL at a keyless endpoint), then retry."
            ),
        )
    if os.environ.get("AGENT_HARDENER_MODEL"):
        _raise_if_bad(
            "analysis",
            DEFAULT_MODEL,
            DEFAULT_BASE_URL,
            validate_choice(DEFAULT_MODEL, DEFAULT_BASE_URL, inference_api_key()),
        )
    attack_model = os.environ.get("GARAK_RED_TEAM_MODEL_NAME")
    if attack_model:
        attack_uri = os.environ.get("GARAK_RED_TEAM_MODEL_URI") or _ATTACK_DEFAULT_URI
        attack_key = os.environ.get(NIM_API_KEY) or inference_api_key()
        _raise_if_bad("attack", attack_model, attack_uri, validate_choice(attack_model, attack_uri, attack_key))
