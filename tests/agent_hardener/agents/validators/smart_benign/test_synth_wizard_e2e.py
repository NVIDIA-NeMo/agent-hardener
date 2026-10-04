# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System test: the ``agent-hardener synth-benign`` wizard end-to-end (synth graph faked).

Runs the *real* ``run_synth_wizard`` — config load, validator resolution, ``SynthState.from_inputs``,
summary, and exit-code logic — with only the compiled synth graph faked (so no LLM/network). This
covers the wizard orchestration, which the unit suite otherwise only ever stubs out. Asserting the
DAG's own artifact writing needs a schema-dispatching fake model (tracked with the subgraph tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.smart_benign import driver, wizard
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.models import SessionConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_ENDPOINT = "http://victim.test/v1/chat/completions"


async def _no_answers(_questions: Any) -> list[dict[str, Any]]:
    return []


def _session(root: Path) -> SessionConfig:
    return SessionConfig(
        storage={"root_dir": root},
        target={"name": "victim", "base_url": _ENDPOINT},
        victim={"name": "victim", "role": "victim", "implementation": "pkg:run"},
        benign_validators=[
            {
                "name": "smart-benign-validator",
                "role": "validator",
                "implementation": wizard.SMART_BENIGN_IMPL,
                "config": {"kind": "benign", "skip_nl_parser": True, "skip_github_analysis": True},
            }
        ],
    )


def _install_fake_graph(monkeypatch: pytest.MonkeyPatch, final_state: SynthState) -> None:
    class _FakeGraph:
        async def astream(self, _state: Any, _config: Any, stream_mode: Any) -> Any:
            yield ("updates", {"request_generator": {}})  # a mapped progress phase
            yield ("values", final_state)

    monkeypatch.setattr(driver, "build_synth_graph", lambda **_kwargs: _FakeGraph())


def _run(monkeypatch: pytest.MonkeyPatch, root: Path, final_state: SynthState) -> int:
    monkeypatch.setattr(wizard, "load_config", lambda _config: _session(root))
    _install_fake_graph(monkeypatch, final_state)
    return wizard.run_synth_wizard(root / "agent-hardener.yaml", answer_provider=_no_answers, no_interactive=True)


def test_synth_wizard_completes_with_a_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    final = SynthState(
        inputs=SynthInputs(target_name="victim"),
        requests=[
            {"tool": "bash_executor", "payload": "echo ok", "label": "benign", "rationale": "", "persona": None},
            {"tool": "python_executor", "payload": "print(1)", "label": "benign", "rationale": "", "persona": None},
        ],
    )

    code = _run(monkeypatch, tmp_path, final)

    assert code == 0
    summary = capsys.readouterr().out
    assert "Requests:      2" in summary
    assert "victim" in summary


def test_synth_wizard_reports_nonzero_on_synth_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    final = SynthState(inputs=SynthInputs(target_name="victim"), errors=["nl_parser failed"])

    code = _run(monkeypatch, tmp_path, final)

    assert code == 1  # exit code propagates synth errors
    assert "Errors:        1" in capsys.readouterr().out
