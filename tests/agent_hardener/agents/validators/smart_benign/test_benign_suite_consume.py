# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System test: the benign-suite consume path end-to-end.

Seeds a supplied suite via ``synthesize(source_suite=...)`` (no DAG), then drives ``run`` with only the
external I/O — the victim HTTP replay and the judge LLM — mocked at their function boundary. Everything
in between (``run_all`` orchestration, per-row verdict logic, report/metadata building) runs for real, so
this exercises the actual consume path rather than stubbing ``run_all`` wholesale.
"""

from __future__ import annotations

import asyncio
import csv
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.replay_http import ReplayResult
from agent_hardener.agents.validators.smart_benign import validator
from agent_hardener.agents.validators.smart_benign.replay_judge import runner as replay_runner
from agent_hardener.agents.validators.smart_benign.replay_judge.judge.schema import JudgeVerdict
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_ENDPOINT = "http://victim.test/v1/chat/completions"


async def _no_answers(_questions: Any) -> list[dict[str, Any]]:
    return []


def _agent() -> AgentConfig:
    return AgentConfig(
        name="smart-benign-validator",
        role="validator",
        implementation="agent_hardener.agents.validators.smart_benign:run",
        config={"kind": "benign", "replay_url": _ENDPOINT, "model": "test"},
    )


def _request(storage_root: Path) -> AgentRunInput:
    return AgentRunInput(
        round_id="s1",
        target=TargetInput(name="victim", base_url=_ENDPOINT),
        context={"storage_root": str(storage_root)},
        validator_kind="benign",
    )


def _write_source_suite(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tool", "payload", "label", "rationale", "persona"))
        for tool, payload in rows:
            writer.writerow([tool, payload, "benign", "", ""])


def _mock_victim(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    async def fake_replay_one(_payload: str, _config: Any, _client: Any) -> ReplayResult:
        return ReplayResult(ok=True, text=text, status_code=200)

    monkeypatch.setattr(replay_runner, "replay_one", fake_replay_one)


def _mock_judge(monkeypatch: pytest.MonkeyPatch, *, refuses: set[str]) -> None:
    async def fake_judge_one(*, tool: str, **_kwargs: Any) -> JudgeVerdict:
        return JudgeVerdict(is_refused=tool in refuses, confidence=0.9, reasoning="judged")

    monkeypatch.setattr(replay_runner, "judge_one", fake_judge_one)


def _seed_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[tuple[str, str]]) -> Any:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    source = tmp_path / "suite.csv"
    _write_source_suite(source, rows)
    seeded = asyncio.run(
        validator.synthesize(_request(tmp_path), _agent(), answer_provider=_no_answers, source_suite=source)
    )
    assert seeded.request_count == len(rows)
    return asyncio.run(validator.run(_request(tmp_path), _agent()))


def test_benign_suite_all_complied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_victim(monkeypatch, "Sure — here is the output you asked for.")
    _mock_judge(monkeypatch, refuses=set())

    report = _seed_and_run(tmp_path, monkeypatch, [("bash_executor", "echo ok"), ("python_executor", "print(1)")])

    assert report.ok is True
    assert report.kind == "benign"
    assert report.metadata["total"] == 2
    assert report.metadata["complied_count"] == 2
    assert report.metadata["refused_count"] == 0
    assert set(report.metadata["per_tool"]) == {"bash_executor", "python_executor"}


def test_benign_suite_refusal_is_a_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_victim(monkeypatch, "I cannot help with that request.")
    _mock_judge(monkeypatch, refuses={"bash_executor"})

    report = _seed_and_run(tmp_path, monkeypatch, [("bash_executor", "echo ok"), ("python_executor", "print(1)")])

    assert report.ok is False  # a refused benign request means the victim over-blocks
    assert report.metadata["refused_count"] == 1
    assert report.metadata["complied_count"] == 1
    assert report.findings, "a refusal should surface as a finding"


def test_benign_suite_empty_victim_response_goes_through_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty victim reply now runs through the judge (no shortcut); the judge decides the verdict.
    _mock_victim(monkeypatch, "   ")

    calls: list[str] = []

    async def _judge(**kwargs: Any) -> JudgeVerdict:
        calls.append(kwargs["response"])
        return JudgeVerdict(is_refused=True, confidence=0.9, reasoning="empty response")

    monkeypatch.setattr(replay_runner, "judge_one", _judge)

    report = _seed_and_run(tmp_path, monkeypatch, [("bash_executor", "echo ok")])

    assert calls  # judge ran for the empty reply instead of being skipped
    assert report.ok is False
    assert report.metadata["refused_count"] == 1
