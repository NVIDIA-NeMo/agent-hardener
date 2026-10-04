# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the smart-benign replay+judge pass (runner + judge client)."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_hardener.agents.validators.replay_http import ReplayResult
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.replay_judge import runner
from agent_hardener.agents.validators.smart_benign.replay_judge.config import ReplayJudgeConfig
from agent_hardener.agents.validators.smart_benign.replay_judge.judge import client
from agent_hardener.agents.validators.smart_benign.replay_judge.judge.schema import JudgeVerdict


def _config(**overrides: Any) -> ReplayJudgeConfig:
    base: dict[str, Any] = {
        "replay_url": "http://victim.test/v1/chat/completions",
        "replay_mode": "openai_chat",
        "model": "smart_benign_validator",
        "input_field": "input_message",
        "response_json_path": None,
        "timeout_seconds": 30.0,
        "excerpt_chars": 500,
        "concurrency": 2,
        "confidence_cutoff": 0.5,
        "judge_model": "judge",
        "judge_base_url": "http://judge.test/v1/",
        "judge_api_key": "key",
        "judge_max_tokens": 256,
        "judge_timeout_seconds": 60.0,
        "judge_input_chars": 4000,
        "judge_temperature": 0.0,
    }
    base.update(overrides)
    return ReplayJudgeConfig(**base)


def _request() -> GeneratedRequest:
    return GeneratedRequest(tool="search", payload="find docs", label="benign", rationale="normal use", persona="dev")


def test_judge_one_forwards_max_tokens_and_timeout(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeModel:
        def with_structured_output(self, _schema: object) -> _FakeModel:
            return self

        async def ainvoke(self, _messages: list[object]) -> JudgeVerdict:
            return JudgeVerdict(is_refused=False, confidence=0.9, reasoning="complied")

    def fake_build(**kwargs: Any) -> _FakeModel:
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(client, "build_chat_model", fake_build)
    verdict = asyncio.run(
        client.judge_one(
            model_name="m",
            base_url="u",
            api_key="k",
            tool="t",
            payload="p",
            response="r",
            max_tokens=256,
            timeout=60.0,
        )
    )
    assert verdict.is_refused is False
    assert captured["max_tokens"] == 256
    assert captured["timeout"] == 60.0


def test_empty_replay_goes_through_judge(monkeypatch) -> None:
    # An empty victim reply no longer auto-refuses: it runs through the judge, which decides.
    async def fake_replay(_payload: str, _config: ReplayJudgeConfig, _client: object) -> ReplayResult:
        return ReplayResult(ok=True, status_code=200, text="   ")

    calls: list[str] = []

    async def fake_judge(**kwargs: Any) -> JudgeVerdict:
        calls.append(kwargs["response"])
        return JudgeVerdict(is_refused=True, confidence=0.9, reasoning="empty response")

    monkeypatch.setattr(runner, "replay_one", fake_replay)
    monkeypatch.setattr(runner, "judge_one", fake_judge)

    results = asyncio.run(runner.run_all([_request()], _config()))

    assert calls  # judge ran for the empty reply instead of being skipped
    assert results[0]["verdict"]["status"] == "refused"
    assert results[0]["verdict"]["is_refused"] is True


def test_nonempty_replay_judges_with_bounded_response(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_replay(_payload: str, _config: ReplayJudgeConfig, _client: object) -> ReplayResult:
        return ReplayResult(ok=True, status_code=200, text="X" * 10_000)

    async def fake_judge(**kwargs: Any) -> JudgeVerdict:
        captured.update(kwargs)
        return JudgeVerdict(is_refused=False, confidence=0.9, reasoning="complied")

    monkeypatch.setattr(runner, "replay_one", fake_replay)
    monkeypatch.setattr(runner, "judge_one", fake_judge)

    results = asyncio.run(runner.run_all([_request()], _config(judge_input_chars=4000)))

    # The judge sees the response bounded to judge_input_chars (+ the truncation marker), not 10k chars.
    assert len(captured["response"]) <= 4000 + len("...<truncated>")
    assert captured["max_tokens"] == 256
    assert captured["timeout"] == 60.0
    assert results[0]["verdict"]["status"] == "complied"
