# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traces are always on: the ``traces`` setting and the provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.manifest import AgentManifest, expand, render_manifest_yaml
from agent_hardener.models import RelayVictimSpec, TracesConfig
from agent_hardener.traces import build_trace_provider
from agent_hardener.traces.relay import RelayTraceProvider

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


def test_traces_are_on_by_default() -> None:
    assert TracesConfig().provider == "relay"
    assert expand(AgentManifest.model_validate(MANIFEST)).traces.provider == "relay"


@pytest.mark.parametrize("provider", ["none", "otel"])
def test_turning_traces_off_or_an_unknown_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        TracesConfig.model_validate({"provider": provider})


def test_the_manifest_template_has_no_traces_switch() -> None:
    assert "traces:" not in render_manifest_yaml({"agent": {}, "backends": []})


def test_factory_builds_relay_only_for_a_relay_victim(tmp_path: Path) -> None:
    spec = RelayVictimSpec(project_dir=".", dockerfile="Dockerfile", victim_binaries=["/usr/bin/python3"])

    def build(cfg: TracesConfig, relay_spec: RelayVictimSpec | None) -> object:
        return build_trace_provider(
            cfg, relay_spec=relay_spec, atof_path=tmp_path / "a.jsonl", sync=lambda: None, raw_dir=tmp_path
        )

    assert isinstance(build(TracesConfig(), spec), RelayTraceProvider)
    assert build(TracesConfig(), None) is None  # a plain HTTP target: nothing to read
