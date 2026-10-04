# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How Agent Hardener talks to a victim: one resolved contract shared by every caller.

Three places drive the victim and each used to hardcode its own request/response shape: the garak
attacker (``agents/attackers/agent_breaker/config.py``), the victim probe
(``agents/victims/openshell_victim.py``) and validator replay
(``agents/validators/replay_http.py``). When they drift, an attack and its replay stop being the same
request and a "blocked" verdict stops meaning anything.

An OpenAI-compatible ``/v1/chat/completions`` is the zero-config default, but it is *not* required.
Demanding it would push work onto the user to adapt their agent to us — the same trade Agent Hardener
declines elsewhere — so an agent exposing another shape declares a mapping instead.

The contract also carries the per-invocation **session id**. Agent Hardener invokes the victim
concurrently, and a Fabric-served victim starts a fresh runtime per request when no session header is
sent (``fabric/session_manager.py:108-111``). Sending our own id keeps concurrent attacks in separate
victim sessions rather than sharing one by accident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# The victim's own default invocation semaphore (``fabric/session_manager.py:34``). Matching it keeps
# Agent Hardener from queueing requests behind a cap it cannot see, where the wait reads as victim latency.
DEFAULT_VICTIM_CONCURRENCY = 8

# Platform agent session header (``nemo_agents_plugin.session_protocol``). Duplicated deliberately:
# agent-hardener must not import nemo_agents_plugin, which pins six nvidia-nat-* distributions.
SESSION_ID_HEADER = "X-Nemo-Session-Id"

RequestMode = Literal["openai_chat", "field"]

# Response keys tried in order when no explicit path is configured, before falling back to raw JSON.
_TEXT_FALLBACK_KEYS = ("output", "response", "text", "summary", "result")


@dataclass(frozen=True, slots=True)
class EndpointContract:
    """Everything needed to send one prompt to a victim and read its answer back."""

    url: str
    mode: RequestMode = "openai_chat"
    model: str = "agent-hardener"
    input_field: str = "input"
    response_json_path: str | None = None
    timeout_seconds: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def request_payload(self, prompt: str) -> dict[str, Any]:
        """Build the request body for ``prompt`` in this victim's shape."""
        if self.mode == "openai_chat":
            return {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        return {self.input_field: prompt}

    def headers(self, session_id: str | None = None) -> dict[str, str]:
        """Headers for one invocation, tagging it with ``session_id`` when supplied."""
        headers = dict(self.extra_headers)
        if session_id:
            headers[SESSION_ID_HEADER] = session_id
        return headers

    def extract_text(self, body: Any) -> str:
        """Pull the assistant's answer out of a response body.

        Falls through explicit path → OpenAI chat shape → common single-field shapes → raw JSON, so a
        victim that answers in an unexpected shape still yields something an attacker can be scored
        against instead of an empty string that looks like a refusal.
        """
        if self.response_json_path:
            value = extract_json_path(body, self.response_json_path)
            if value is not None:
                return str(value)
        if not isinstance(body, dict):
            return str(body)
        text = _openai_chat_text(body)
        if text is not None:
            return text
        for key in _TEXT_FALLBACK_KEYS:
            if body.get(key) is not None:
                return str(body[key])
        return json.dumps(body, sort_keys=True)

    def garak_rest_fields(self) -> dict[str, Any]:
        """The same mapping expressed for garak's ``rest.RestGenerator``.

        The attacker runs as a subprocess and cannot share Python objects, so it gets the mapping as
        config. Deriving it here is what keeps attacker and replay speaking one shape.
        """
        return {
            "req_template_json_object": self.request_payload("$INPUT"),
            "response_json": True,
            "response_json_field": self.response_json_path or "$.choices[0].message.content",
        }


def _openai_chat_text(body: dict[str, Any]) -> str | None:
    """Read ``choices[0].message.content``, tolerating the legacy ``choices[0].text`` completion shape."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if first.get("text") is not None:
        return str(first["text"])
    return None


def extract_json_path(body: Any, path: str) -> Any:
    """Resolve a restricted ``$.a.b[0].c`` path, returning ``None`` when it does not apply.

    Deliberately not full JSONPath: the paths come from a victim manifest, and a small grammar that
    fails closed beats a dependency that can evaluate arbitrary expressions against victim output.
    """
    normalized = path[2:] if path.startswith("$.") else path
    if not normalized:
        return body
    current = body
    for part in normalized.split("."):
        match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\[(\d+)])?", part)
        if match is None:
            return None
        key, index = match.groups()
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
        if index is not None:
            if not isinstance(current, list) or int(index) >= len(current):
                return None
            current = current[int(index)]
    return current
