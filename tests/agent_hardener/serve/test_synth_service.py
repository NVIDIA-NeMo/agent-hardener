# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the HTTP synth service: interview interrupt -> answers -> review -> write suite.

The synth graph, config loading, and CSV writer are stubbed so the test exercises the request/response flow
(and the interrupt/resume mapping) without a live victim or LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from starlette.testclient import TestClient

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.serve import app as serve_app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _interrupt(gap: str) -> dict[str, Any]:
    question = {"gap": gap, "question": f"{gap}?", "options": []}
    return {"__interrupt__": (SimpleNamespace(value={"questions": [question]}),)}


def _final_state() -> dict[str, Any]:
    return SynthState(
        inputs=SynthInputs(target_name="t"),
        requests=[GeneratedRequest(tool="clock", payload="what time is it?", label="benign")],
    ).model_dump()


class _FakeGraph:
    """Interrupts on the first invoke (one interview round), completes on the resume."""

    def __init__(self, interview_rounds: int = 1) -> None:
        self.calls = 0
        self.interview_rounds = interview_rounds

    async def ainvoke(self, _input: Any, _config: Any) -> Any:
        self.calls += 1
        if self.calls <= self.interview_rounds:
            return _interrupt(f"gap-{self.calls}")
        return _final_state()


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, interview_rounds: int = 1) -> TestClient:
    session = SimpleNamespace(
        target=SimpleNamespace(name="t", base_url="http://victim.test/v1", agent_relay_plugins=None),
        storage=SimpleNamespace(root_dir=tmp_path),
    )
    monkeypatch.setattr(serve_app, "build_synth_graph", lambda **_kw: _FakeGraph(interview_rounds))
    monkeypatch.setattr(serve_app, "load_config", lambda _config: session)
    monkeypatch.setattr(serve_app, "_resolve_validator", lambda _s, _v: SimpleNamespace(config={}))
    monkeypatch.setattr(serve_app, "build_synth_inputs", lambda *_a, **_k: SynthInputs(target_name="t"))
    monkeypatch.setattr(serve_app, "_write_requests_csv", lambda target_dir, _reqs: target_dir / "requests.csv")
    return TestClient(serve_app.create_app())


def test_synth_interview_then_review_then_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)

    start = client.post("/synth", json={"config": "agent-hardener.yaml"}).json()
    assert start["status"] == "interview"
    assert start["questions"][0]["gap"] == "gap-1"
    thread_id = start["thread_id"]

    review = client.post(
        f"/synth/{thread_id}/answers", json={"answers": [{"gap": "gap-1", "answer": "support"}]}
    ).json()
    assert review["status"] == "review"
    assert [r["tool"] for r in review["suite"]] == ["clock"]

    done = client.post(f"/synth/{thread_id}/suite", json={"suite": review["suite"]}).json()
    assert done["status"] == "done"
    assert done["benign_csv"].endswith("requests.csv")


def test_synth_multi_round_interview_resumes_across_two_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two interview rounds: each answers post that hits another interrupt returns the next interview."""
    client = _client(tmp_path, monkeypatch, interview_rounds=2)

    start = client.post("/synth", json={"config": "agent-hardener.yaml"}).json()
    assert start["status"] == "interview"
    assert start["questions"][0]["gap"] == "gap-1"
    thread_id = start["thread_id"]

    # First answers round resumes the graph into a *second* interview round, not the review yet.
    second = client.post(f"/synth/{thread_id}/answers", json={"answers": [{"gap": "gap-1", "answer": "a1"}]}).json()
    assert second["status"] == "interview"
    assert second["questions"][0]["gap"] == "gap-2"
    assert second["thread_id"] == thread_id

    # Second answers round completes the interview and yields the review suite.
    review = client.post(f"/synth/{thread_id}/answers", json={"answers": [{"gap": "gap-2", "answer": "a2"}]}).json()
    assert review["status"] == "review"
    assert [r["tool"] for r in review["suite"]] == ["clock"]

    done = client.post(f"/synth/{thread_id}/suite", json={"suite": review["suite"]}).json()
    assert done["status"] == "done"


def test_answers_for_unknown_thread_is_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    assert client.post("/synth/ghost/answers", json={"answers": []}).status_code == 404
