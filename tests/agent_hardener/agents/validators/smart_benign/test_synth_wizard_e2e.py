# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System test: the ``agent-hardener synth-benign`` wizard end-to-end (synth graph faked).

Runs the *real* ``run_synth_wizard`` — config load, validator resolution, ``SynthState.from_inputs``,
summary, and exit-code logic — with only the compiled synth graph faked (so no LLM/network). This
covers the wizard orchestration, which the unit suite otherwise only ever stubs out. Asserting the
DAG's own artifact writing needs a schema-dispatching fake model (tracked with the subgraph tests).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign import driver, wizard
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.tool_verification import (
    DropDecision,
    PassRecord,
    ToolVerificationOutcome,
)
from agent_hardener.agents.validators.smart_benign.validator import load_requests
from agent_hardener.agents.validators.smart_benign.verify import VerificationResult
from agent_hardener.models import SessionConfig

if TYPE_CHECKING:
    from pathlib import Path

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
    assert "✓ Benign suite: 2 request(s) for victim" in summary
    assert "victim" in summary


def test_synth_wizard_reports_nonzero_on_synth_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    final = SynthState(inputs=SynthInputs(target_name="victim"), errors=["nl_parser failed"])

    code = _run(monkeypatch, tmp_path, final)

    assert code == 1  # exit code propagates synth errors
    assert "✗ Benign suite: 0 request(s) for victim, 1 error(s)" in capsys.readouterr().out


def _suite_state(root: Path) -> SynthState:
    profile = VictimCapabilityProfile(
        target_name="victim",
        tools=[ToolSpec(name="bash_executor", description="d", source="nl", confidence=1.0)],
        input_hash="h",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return SynthState(
        inputs=SynthInputs(target_name="victim"),
        profile=profile,
        requests=[GeneratedRequest(tool="bash_executor", payload="echo ok", label="benign")],
        artifact_dir=root / "benign_profiles" / "victim",
    )


def test_without_traces_tool_verification_is_skipped_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(wizard, "run_tool_verification", lambda *_a, **_k: pytest.fail("must not verify"))

    assert _run(monkeypatch, tmp_path, _suite_state(tmp_path)) == 0
    assert "tool verification skipped" in capsys.readouterr().out


def test_with_traces_the_verified_suite_replaces_the_synthesized_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "k")
    final = _suite_state(tmp_path)
    seen: dict[str, Any] = {}
    verified_tool = final.profile.tools[0].model_copy(update={"verification": "verified"})  # type: ignore[union-attr]
    added = ToolSpec(name="lookup", description="d", source="trace", confidence=1.0, verification="verified")
    rows = [*final.requests, GeneratedRequest(tool="lookup", payload="find Ann", label="benign")]
    record = PassRecord(
        number=1,
        result=VerificationResult(traced=True, rows=1, traced_rows=1, tools=[], baseline_refused=[0]),
        decision=DropDecision(decided_by="nothing_to_decide"),
        added_tools=["lookup"],
        added_rows=1,
    )

    async def fake_verify(state: SynthState, **kw: Any) -> ToolVerificationOutcome:
        seen.update(kw, state=state)
        profile = state.profile.model_copy(update={"tools": [verified_tool, added]})  # type: ignore[union-attr]
        return ToolVerificationOutcome(requests=rows, profile=profile, passes=[record], unhandled_tools=["later"])

    monkeypatch.setattr(wizard, "run_tool_verification", fake_verify)
    monkeypatch.setattr(wizard, "load_config", lambda _config: _session(tmp_path))
    _install_fake_graph(monkeypatch, final)

    code = wizard.run_synth_wizard(
        tmp_path / "agent-hardener.yaml",
        answer_provider=_no_answers,
        no_interactive=True,
        trace_provider="provider",  # type: ignore[arg-type]
    )

    assert code == 0
    assert seen["provider"] == "provider"
    assert seen["decide"] is wizard.default_drop_decider  # no TTY prompt when not interactive
    assert seen["cfg"].replay_url == _ENDPOINT
    out = capsys.readouterr().out
    assert "✓ Benign suite: 2 request(s) for victim" in out
    assert "2 verified of 2 tool(s); 1/1 row(s) traced" in out
    assert "1 row(s) refused by the unhardened victim" in out
    assert "lookup (1 row(s))" in out
    assert "later (pass cap reached)" in out
    artifact_dir = tmp_path / "benign_profiles" / "victim"
    assert (artifact_dir / "tool_verification.json").is_file()
    assert [r.payload for r in load_requests(artifact_dir / "requests.csv")] == ["echo ok", "find Ann"]


def test_a_skipped_verification_keeps_the_suite_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "k")

    async def fake_verify(state: SynthState, **_kw: Any) -> ToolVerificationOutcome:
        return ToolVerificationOutcome(requests=state.requests, profile=state.profile, skipped="no traces")

    monkeypatch.setattr(wizard, "run_tool_verification", fake_verify)
    monkeypatch.setattr(wizard, "load_config", lambda _config: _session(tmp_path))
    _install_fake_graph(monkeypatch, _suite_state(tmp_path))

    wizard.run_synth_wizard(
        tmp_path / "x.yaml",
        answer_provider=_no_answers,
        no_interactive=True,
        trace_provider="provider",  # type: ignore[arg-type]
    )

    assert "tool verification: no traces" in capsys.readouterr().out
