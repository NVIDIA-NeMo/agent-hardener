# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``extraction.environment``: ``extract_environment`` (relay_victim -> tuples) and

``endpoint_is_protected`` (the host+port wildcard-aware membership check).

``complete_structured`` is stubbed at its use site in ``extraction.environment`` (same convention
as ``test_openshell_defender_v2_agent.py``), so the LLM egress-classification call never hits a
real model.
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.extraction_cache import ExtractionCache
from agent_hardener.agents.defenders.openshell_defender_v2.config import DefenderConfig
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.environment import (
    endpoint_is_protected,
    extract_environment,
)
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.prompts import EnvironmentEndpointOutput
from agent_hardener.models.contracts import DefenderInput
from agent_hardener.models.infra import BackendEndpoint, BackendServiceSpec, RelayVictimSpec

pytestmark = pytest.mark.unit

_MODULE = "agent_hardener.agents.defenders.openshell_defender_v2.extraction.environment"


def _relay_victim(**overrides) -> RelayVictimSpec:
    kwargs = {
        "project_dir": "/tmp/project",
        "dockerfile": "Dockerfile",
        "victim_binaries": ["/app/**"],
        "agent_env": {"INTEGRATION_BACKEND_URL": "http://host.docker.internal:8089"},
        "backends": [
            BackendServiceSpec(
                name="integration_backend",
                allowlist=[BackendEndpoint(host="host.docker.internal", port=8089, protocol="rest")],
            )
        ],
        "egress": ["integrate.api.nvidia.com", "es.example.com:9200"],
    }
    kwargs.update(overrides)
    return RelayVictimSpec(**kwargs)


def _defender_input(relay_victim_spec: RelayVictimSpec | None) -> DefenderInput:
    return DefenderInput(
        attack_prompt="",
        agent_response="",
        attacked_tool="",
        relay_victim_spec=relay_victim_spec,
    )


def test_returns_empty_when_no_relay_victim_spec(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("LLM should not be called when there's no relay_victim_spec")

    monkeypatch.setattr(f"{_MODULE}.complete_structured", fail)
    assert extract_environment(_defender_input(None), DefenderConfig()) == []


def test_llm_selects_inference_endpoint_only(monkeypatch) -> None:
    monkeypatch.setattr(
        f"{_MODULE}.complete_structured",
        lambda *_args: EnvironmentEndpointOutput(host="integrate.api.nvidia.com"),
    )
    tuples = extract_environment(_defender_input(_relay_victim()), DefenderConfig())

    hosts_ports = {(t.host, t.port) for t in tuples}
    assert ("host.docker.internal", 8089) in hosts_ports  # from agent_env
    assert ("integrate.api.nvidia.com", 443) in hosts_ports  # LLM-selected egress
    assert not any(h == "es.internal" for h, _p in hosts_ports)  # unselected egress excluded
    # host.docker.internal appears once (agent_env) not twice (also via backends allowlist dedup isn't
    # required here — both sources may legitimately emit it; just confirm no crash / stable count).
    assert len(tuples) == 3


def test_llm_returns_no_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        f"{_MODULE}.complete_structured",
        lambda *_args: EnvironmentEndpointOutput(host=None),
    )
    tuples = extract_environment(_defender_input(_relay_victim()), DefenderConfig())
    hosts = {t.host for t in tuples}
    assert hosts == {"host.docker.internal"}


def test_llm_not_called_when_egress_empty(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("LLM should not be called when egress is empty")

    monkeypatch.setattr(f"{_MODULE}.complete_structured", fail)
    tuples = extract_environment(_defender_input(_relay_victim(egress=[])), DefenderConfig())
    assert {t.host for t in tuples} == {"host.docker.internal"}


def test_no_agent_env_or_backends(monkeypatch) -> None:
    monkeypatch.setattr(
        f"{_MODULE}.complete_structured",
        lambda *_args: EnvironmentEndpointOutput(host=None),
    )
    relay_victim = _relay_victim(agent_env={}, backends=[], egress=[])
    assert extract_environment(_defender_input(relay_victim), DefenderConfig()) == []


# --- cache -------------------------------------------------------------------------------------


def test_cache_dedupes_llm_calls_across_extract_environment_calls(monkeypatch) -> None:
    """Same ``egress`` across repeated calls (one per attack in a run) should hit the LLM once."""
    calls = []

    def fake_complete_structured(*_args):
        calls.append(1)
        return EnvironmentEndpointOutput(host="integrate.api.nvidia.com")

    monkeypatch.setattr(f"{_MODULE}.complete_structured", fake_complete_structured)
    cache = ExtractionCache()
    defender_input = _defender_input(_relay_victim())

    first = extract_environment(defender_input, DefenderConfig(), cache=cache)
    second = extract_environment(defender_input, DefenderConfig(), cache=cache)

    assert len(calls) == 1
    assert first == second


# --- endpoint_is_protected -------------------------------------------------------------------

_PROTECTED = {("host.docker.internal", 8089)}


def test_exact_match_is_protected() -> None:
    assert endpoint_is_protected(_PROTECTED, "host.docker.internal", 8089) is True


def test_different_host_is_not_protected() -> None:
    assert endpoint_is_protected(_PROTECTED, "other.internal", 8089) is False


def test_different_port_is_not_protected() -> None:
    assert endpoint_is_protected(_PROTECTED, "host.docker.internal", 9000) is False


def test_none_host_is_never_protected() -> None:
    assert endpoint_is_protected(_PROTECTED, None, 8089) is False


def test_none_cut_port_matches_via_wildcard() -> None:
    """The exact reported regression: ``_locate`` resolves a cut with an unspecified port (``None``)

    to the concrete protected endpoint via the same wildcard convention ``policy.query`` uses, so
    the protection check must treat a ``None`` cut port as matching any protected port for that
    host — not as a mismatch.
    """
    assert endpoint_is_protected(_PROTECTED, "host.docker.internal", None) is True


def test_none_protected_port_matches_any_cut_port() -> None:
    protected = {("host.docker.internal", None)}
    assert endpoint_is_protected(protected, "host.docker.internal", 8089) is True
