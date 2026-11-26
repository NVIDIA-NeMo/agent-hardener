# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-run model checks: every model the run calls must answer with the key it will be called with.

A war-game builds a Docker sandbox and runs for minutes; a mistyped model name or a wrong
endpoint/key should fail in seconds, not surface later as an opaque attacker/validator error.

Each model gets a one-token ``/chat/completions`` request (:func:`require_answers`): ``/models`` is often
public, so listing it cannot tell a good key from a bad one. Callers describe what they will call as
:class:`ModelUse`; :mod:`agent_hardener.preflight.models` collects them for a whole run. Each failure raises
:class:`~agent_hardener.errors.ModelUnavailableError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from agent_hardener.env import INFERENCE_API_KEY, inference_api_key
from agent_hardener.errors import ModelUnavailableError
from agent_hardener.llm import default_base_url, default_model
from agent_hardener.rate_limits import RetryPolicy, RetryTransport, is_rate_limited_response
from agent_hardener.relay_plugin.config import no_reasoning_body

_PROBE_TIMEOUT_S = 10.0
#: Generous for one token, so a slow but working endpoint is not mistaken for a dead one.
GENERATION_PROBE_TIMEOUT_S = 60.0
#: garak retries a 429 or 5xx with no limit. The one-token probe retries them briefly, so a short overload
#: doesn't abort a scan garak would have waited out, while a backend that stays down still fails in ~20s.
PROBE_RETRY = RetryPolicy(rate_limit_budget_s=20.0, server_error_delays_s=(5.0, 15.0))


@dataclass(frozen=True)
class Validation:
    """A model choice's verdict; ``available`` lists what the credentials can reach for a helpful error."""

    ok: bool
    reason: str = ""  # "", "auth", "unreachable", "unknown_model", "timeout", "error"
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


def probe_generation(
    model: str, base_url: str, api_key: str | None, *, timeout: float, extra_body: dict[str, Any] | None = None
) -> Validation:
    """Ask *model* for one token at ``{base_url}/chat/completions`` (best-effort; never raises).

    Catches what ``/models`` can't: an endpoint that lists the model but never answers a request.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, **(extra_body or {})}
    try:
        with _probe_client(timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        # Not retried: the wait was already a full model timeout, as long as garak's own would be.
        return Validation(ok=False, reason="timeout", detail=f"no response within {timeout:g}s")
    except httpx.HTTPError as exc:
        return Validation(ok=False, reason="unreachable", detail=str(exc) or exc.__class__.__name__)
    if resp.status_code in (401, 403, 404):
        # A gateway can refuse an unknown model with the status it gives a bad key (inference-api.nvidia.com
        # answers both with 403 key_model_access_denied), so ask its model list which one this is.
        listed = validate_choice(model, base_url, api_key)
        if listed.reason == "unknown_model":
            return listed
        reason = "error" if resp.status_code == 404 else "auth"
        return Validation(ok=False, reason=reason, detail=f"HTTP {resp.status_code}: {_server_message(resp)}")
    if resp.status_code >= 400:
        return Validation(
            ok=False, reason="error", detail=f"HTTP {resp.status_code}{_retried(resp)}: {_server_message(resp)}"
        )
    return Validation(ok=True)


def _probe_client(timeout: float) -> httpx.Client:
    return httpx.Client(transport=RetryTransport(policy=PROBE_RETRY), timeout=timeout)


def _retried(resp: httpx.Response) -> str:
    """How long the probe kept trying before it accepted *resp*, for the error detail."""
    if is_rate_limited_response(resp):
        return f" after retrying for up to {PROBE_RETRY.rate_limit_budget_s:g}s"
    if resp.status_code >= 500:
        return f" after {len(PROBE_RETRY.server_error_delays_s) + 1} attempts"
    return ""


def _server_message(resp: httpx.Response) -> str:
    """The ``error.message`` of an OpenAI-style error body, else the start of the raw body."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    return str(message or resp.text)[:200]


def raise_if_unavailable(
    label: str,
    model: str | None,
    base_url: str,
    verdict: Validation,
    *,
    key_env: str | None = None,
    timeout_setting: str | None = None,
) -> None:
    """Turn a failed :class:`Validation` into a :class:`ModelUnavailableError` (no-op when ok).

    ``key_env`` names the variable holding the key and ``timeout_setting`` the setting that gives the model
    more time, so the error can say which one to change.
    """
    if verdict.ok:
        return
    if verdict.reason == "auth":
        raise ModelUnavailableError(
            f"The {label} model credentials were rejected by {base_url} ({verdict.detail}).",
            remediation=f"Set {key_env} to a key {base_url} accepts, then retry." if key_env else None,
        )
    if verdict.reason == "unreachable":
        raise ModelUnavailableError(f"Could not reach the {label} model endpoint {base_url} ({verdict.detail}).")
    if verdict.reason in ("timeout", "error"):
        raise ModelUnavailableError(
            f"The {label} model {model!r} at {base_url} failed a one-token test request ({verdict.detail}).",
            remediation=f"Check that {base_url} is serving {model!r}."
            + (
                f" A slow reasoning model that needs longer to answer can be given more time with {timeout_setting}."
                if timeout_setting
                else ""
            ),
        )
    available = ", ".join(verdict.available[:20]) or "none"
    raise ModelUnavailableError(
        f"The {label} model {model!r} is not available at {base_url}. Reachable models: {available}."
    )


@dataclass(frozen=True)
class ModelUse:
    """One model the run will call, exactly as it will call it."""

    label: str
    model: str
    base_url: str
    api_key: str | None
    #: The variable that holds ``api_key``, named in the error when the key is rejected.
    key_env: str
    timeout: float = GENERATION_PROBE_TIMEOUT_S
    extra_body: dict[str, Any] | None = None
    #: The setting that gives this model more time, named in the error when it does not answer in time.
    timeout_setting: str | None = None


def require_answers(uses: list[ModelUse]) -> None:
    """Ask each distinct model, endpoint and key for one token; raise on the first that does not answer.

    Uses that share all three are asked once, under their joined labels.
    """
    distinct: dict[tuple[str, str, str | None], ModelUse] = {}
    for use in uses:
        target = (use.model, use.base_url.rstrip("/"), use.api_key)
        seen = distinct.get(target)
        distinct[target] = replace(seen, label=f"{seen.label} and {use.label}") if seen else use
    for use in distinct.values():
        verdict = probe_generation(use.model, use.base_url, use.api_key, timeout=use.timeout, extra_body=use.extra_body)
        raise_if_unavailable(
            use.label, use.model, use.base_url, verdict, key_env=use.key_env, timeout_setting=use.timeout_setting
        )


def require_inference_key() -> None:
    """Fail up front when the default (NVIDIA-hosted) endpoint has no key.

    A custom ``AGENT_HARDENER_BASE_URL`` may point at a keyless local model, so it is exempt; the one-token
    request still proves whether that endpoint wants a key.
    """
    if not inference_api_key() and not os.environ.get("AGENT_HARDENER_BASE_URL"):
        raise ModelUnavailableError(
            f"{INFERENCE_API_KEY} is not set — Agent Hardener has no credential for its default models.",
            remediation=(
                f"Set {INFERENCE_API_KEY} in your --env-file or environment "
                "(or point AGENT_HARDENER_BASE_URL at a keyless endpoint), then retry."
            ),
        )


def analysis_model_use() -> ModelUse:
    """The analysis model every LLM agent of the run shares."""
    model = default_model()
    return ModelUse(
        "analysis",
        model,
        default_base_url(),
        inference_api_key(),
        INFERENCE_API_KEY,
        extra_body=no_reasoning_body(model),
    )


def preflight_configured_models() -> None:
    """Check the analysis model, for commands that call no other (``serve``, ``synth-benign``)."""
    require_inference_key()
    require_answers([analysis_model_use()])
