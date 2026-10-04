# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction of the victim agent's own required endpoints from its ``relay_victim`` run spec.

Unlike ``extraction.benign``, nothing here is inferred from request text — ``agent_env``,
``backends``, and ``egress`` are the run's own configuration, so the agent needs these endpoints by
construction. Benign extraction alone can never surface them (they never appear in a benign request
string), yet a defender patch that blocks one breaks the agent outright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from agent_hardener.openshell.egress.sources import ManualEgressSource

from ..config import load_config
from ..llm.client import complete_structured
from ..models import RequestTuple
from .prompts import ENVIRONMENT_LLM_ENDPOINT_PROMPT, EnvironmentEndpointOutput
from .request_tuple import canonicalize

if TYPE_CHECKING:
    from agent_hardener.models.contracts import DefenderInput
    from agent_hardener.models.infra import RelayVictimSpec

    from ...extraction_cache import ExtractionCache
    from ..config import DefenderConfig

EndpointKey = tuple[str | None, int | None]


def _agent_env_tuples(agent_env: dict[str, str]) -> list[RequestTuple]:
    """Backend URLs injected into the agent's own environment (e.g. ``INTEGRATION_BACKEND_URL``).

    Parsed directly rather than via :class:`ManualEgressSource` — that helper's URL parser skips
    ``host.docker.internal`` (it expects that alias to be covered by ``backends[].allowlist``
    instead), but ``agent_env`` legitimately points there, so dropping it here would silently lose
    the most common case.
    """
    tuples: list[RequestTuple] = []
    for value in agent_env.values():
        if "://" not in value:
            continue
        parsed = urlparse(value)
        if not parsed.hostname:
            continue
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        tuples.append(RequestTuple(host=parsed.hostname, port=port, protocol="rest"))
    return tuples


def _llm_endpoint_tuple_uncached(relay_victim: RelayVictimSpec, config: DefenderConfig) -> RequestTuple | None:
    if not relay_victim.egress:
        return None
    prompt = ENVIRONMENT_LLM_ENDPOINT_PROMPT.format(candidates="\n".join(f"- {e}" for e in relay_victim.egress))
    output = complete_structured(prompt, EnvironmentEndpointOutput, config)
    if not output.host:
        return None
    matched = next((e for e in relay_victim.egress if output.host.lower() in e.lower()), None)
    if matched is None:
        return None
    endpoint = next(iter(ManualEgressSource([matched]).discover()), None)
    return RequestTuple(host=endpoint.host, port=endpoint.port, protocol="rest") if endpoint else None


def _llm_endpoint_tuple(
    relay_victim: RelayVictimSpec, config: DefenderConfig, cache: ExtractionCache | None
) -> RequestTuple | None:
    """Which single ``egress`` entry is the agent's own LLM communication endpoint, per an LLM call.

    ``egress``/``discover_egress`` mixes arbitrary discovered hosts (a search cluster, a queue, the
    LLM endpoint, ...) — unlike ``agent_env``/``backends`` this isn't safe to blanket-trust, so only
    the entry the model identifies as the agent's own inference endpoint is pulled in.

    ``relay_victim.egress`` is the same list for every attack within one ``DefendersManager.run()``
    call — when ``cache`` is given, this result is computed once per run and reused.
    """
    if not relay_victim.egress:
        return None
    if cache is None:
        return _llm_endpoint_tuple_uncached(relay_victim, config)
    key = ("environment_endpoint", tuple(relay_victim.egress))
    return cache.get_or_compute(key, lambda: _llm_endpoint_tuple_uncached(relay_victim, config))


def extract_environment(
    defender_input: DefenderInput,
    config: DefenderConfig | None = None,
    cache: ExtractionCache | None = None,
) -> list[RequestTuple]:
    """Endpoints the victim agent needs by construction — agent_env backend URLs,

    backends[].allowlist entries, and its own LLM inference endpoint. None of these are ever
    inferred from request text, and none of them may ever be modified by any mitigation node —
    see ``endpoint_is_protected`` and ``DefenderContext.protected_endpoints``.
    """
    relay_victim = defender_input.relay_victim_spec
    if relay_victim is None:
        return []

    agent_env_tuples = _agent_env_tuples(relay_victim.agent_env)
    backend_tuples = [
        RequestTuple(host=e.host, port=e.port, protocol=e.protocol)
        for backend in relay_victim.backends
        for e in backend.allowlist
    ]
    llm_endpoint = _llm_endpoint_tuple(relay_victim, config or load_config(), cache)

    tuples = [*agent_env_tuples, *backend_tuples, *([llm_endpoint] if llm_endpoint else [])]
    return [canonicalize(t) for t in tuples]


def endpoint_is_protected(protected: set[EndpointKey], host: str | None, port: int | None) -> bool:
    """Same host+port wildcard semantics as ``policy.query.find_entry_for``/``find_endpoint``:

    a ``None`` port on either side matches any port. Without this, a cut resolved through that
    same wildcard (e.g. an attack whose extracted port is ``None``) could reach a protected
    endpoint's concrete port while an exact-tuple membership check silently missed it.
    """
    if host is None:
        return False
    return any(p_host == host and (port is None or p_port is None or port == p_port) for p_host, p_port in protected)
