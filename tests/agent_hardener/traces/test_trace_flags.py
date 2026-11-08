# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The global ``traces`` switch, the per-component ``use_traces`` flag, and the provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.manifest import AgentManifest, expand, render_manifest_yaml
from agent_hardener.models import AgentConfig, RelayVictimSpec, TracesConfig
from agent_hardener.traces import build_trace_provider, uses_traces
from agent_hardener.traces.relay import RelayTraceProvider
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST = {
    "agent": {
        "name": "bank",
        "project_dir": ".",
        "dockerfile": "Dockerfile",
        "start_command": "python server.py",
        "binaries": ["/usr/bin/python3"],
    }
}


def _agent(**config: object) -> AgentConfig:
    return AgentConfig(name="garak-attack-replay-validator", role="validator", config=dict(config))


def test_traces_default_off_and_existing_configs_load_unchanged() -> None:
    assert TracesConfig().provider == "none"
    assert expand(AgentManifest.model_validate(MANIFEST)).traces.provider == "none"


def test_the_manifest_turns_traces_on() -> None:
    session = expand(AgentManifest.model_validate({**MANIFEST, "traces": {"provider": "relay"}}))
    assert session.traces.provider == "relay"


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider"):
        TracesConfig.model_validate({"provider": "otel"})


def test_the_manifest_template_documents_traces() -> None:
    assert "# traces:" in render_manifest_yaml({"agent": {}, "backends": []})


@pytest.mark.parametrize(
    ("provider", "flag", "expected"),
    [
        (None, True, False),  # global off: the component flag is ignored
        (FakeTraceProvider(), False, False),  # recording only
        (FakeTraceProvider(), True, True),
    ],
)
def test_a_component_uses_traces_only_when_both_flags_are_on(provider, flag, expected) -> None:
    assert uses_traces(_agent(use_traces=flag), provider) is expected
    assert uses_traces(_agent(), provider) is False


def test_factory_builds_relay_only_for_a_relay_victim(tmp_path: Path) -> None:
    spec = RelayVictimSpec(project_dir=".", dockerfile="Dockerfile", victim_binaries=["/usr/bin/python3"])

    def build(cfg: TracesConfig, relay_spec: RelayVictimSpec | None) -> object:
        return build_trace_provider(
            cfg, relay_spec=relay_spec, atof_path=tmp_path / "a.jsonl", sync=lambda: None, raw_dir=tmp_path
        )

    assert build(TracesConfig(), spec) is None
    assert isinstance(build(TracesConfig(provider="relay"), spec), RelayTraceProvider)
    assert build(TracesConfig(provider="relay"), None) is None
