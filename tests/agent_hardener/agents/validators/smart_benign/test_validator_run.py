# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the split smart-benign entrypoints: replay-only ``run`` + cached ``synthesize``."""

from __future__ import annotations

import asyncio
import csv
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.smart_benign import driver, validator
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.state import SynthState
from agent_hardener.loggers import using_event_sink
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


async def _no_answers(_questions: Any) -> list[dict[str, Any]]:
    """Stub interview answer provider: never consulted in these tests (non-interactive)."""
    return []


def _agent() -> AgentConfig:
    return AgentConfig(
        name="smart-benign-validator",
        role="validator",
        implementation="agent_hardener.agents.validators.smart_benign:run",
        config={
            "kind": "benign",
            "skip_nl_parser": True,
            "skip_github_analysis": True,
            "replay_url": "http://victim.test/v1/chat/completions",
            "model": "test",
        },
    )


def _request(storage_root: Path) -> AgentRunInput:
    return AgentRunInput(
        round_id="s1",
        target=TargetInput(name="victim", base_url="http://victim.test/v1/chat/completions"),
        context={"storage_root": str(storage_root)},
        validator_kind="benign",
    )


def test_cache_key_folds_in_workflow_hash(tmp_path: Path) -> None:
    # Same config/endpoint, different victim workflow file → different cache key, so a swapped
    # agent behind the same endpoint re-synthesizes instead of reusing a stale suite.
    finance = tmp_path / "finance.yaml"
    finance.write_text("agent: finance\n", encoding="utf-8")
    review = tmp_path / "review.yaml"
    review.write_text("agent: code-review\n", encoding="utf-8")

    def _hash_for(workflow: Path) -> str:
        req = AgentRunInput(
            round_id="s1",
            target=TargetInput(name="victim", base_url="http://victim.test", agent_relay_plugins=workflow),
            context={},
            validator_kind="benign",
        )
        return validator.compute_input_hash(validator._build_inputs(req, _agent()))

    assert _hash_for(finance) != _hash_for(review)
    assert _hash_for(finance) == _hash_for(finance)


def _write_suite(artifact_dir: Path, rows: list[dict[str, str]]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / validator.REQUESTS_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tool", "payload", "label", "rationale", "persona"))
        for row in rows:
            writer.writerow([row["tool"], row["payload"], row.get("label", "benign"), "", row.get("persona", "")])


def test_run_hard_fails_when_no_suite(tmp_path: Path) -> None:
    report = asyncio.run(validator.run(_request(tmp_path), _agent()))
    assert report.ok is False
    assert report.error is not None
    assert "run synthesis first" in report.error or "synth-benign" in report.error


def test_run_replays_cached_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    artifact_dir = tmp_path / "benign_profiles" / "victim"
    _write_suite(artifact_dir, [{"tool": "bash_executor", "payload": "echo ok"}])

    async def fake_run_all(requests: list[Any], cfg: Any) -> list[dict[str, Any]]:
        assert len(requests) == 1
        assert requests[0].tool == "bash_executor"
        return [
            {
                "index": 1,
                "tool": "bash_executor",
                "label": "benign",
                "persona": None,
                "payload_excerpt": "echo ok",
                "verdict": {"status": "complied", "confidence": 0.9, "reasoning": "ok"},
            }
        ]

    monkeypatch.setattr(validator, "run_all", fake_run_all)
    report = asyncio.run(validator.run(_request(tmp_path), _agent()))
    assert report.ok is True
    assert report.metadata["complied_count"] == 1
    assert report.metadata["total"] == 1


def test_synthesize_reuses_cache_on_input_hash_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)
    inputs = validator._build_inputs(request, agent)

    _write_suite(artifact_dir, [{"tool": "bash_executor", "payload": "echo ok"}])
    (artifact_dir / validator.INPUT_HASH_FILE).write_text(validator.compute_input_hash(inputs) + "\n", encoding="utf-8")

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("synth DAG must not run on a cache hit")

    monkeypatch.setattr(validator, "drive_synth", _boom)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers))
    assert result.artifact_dir == artifact_dir
    assert result.cached is True
    assert result.request_count == 1
    assert result.viable is True


def test_synthesize_trusts_on_disk_suite_without_input_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A suite written out-of-band (e.g. a manifest-owned suite injected into a fresh run) has no matching
    # input_hash.txt, so it's a normal cache miss — but trust_cached_suite must replay it without the DAG.
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)
    _write_suite(artifact_dir, [{"tool": "bash_executor", "payload": "echo ok"}])
    assert not (artifact_dir / validator.INPUT_HASH_FILE).exists()

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("synth DAG must not run when trusting the on-disk suite")

    monkeypatch.setattr(validator, "drive_synth", _boom)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers, trust_cached_suite=True))
    assert result.cached is True
    assert result.request_count == 1


def test_synthesize_uses_supplied_source_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `run --benign-suite <path>`: the caller hands over a CSV; agent-hardener seeds it into the target's
    # artifact dir and replays it as-is, without running the synth DAG.
    agent = _agent()
    request = _request(tmp_path)
    source = tmp_path / "supplied-suite.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "payload", "label", "rationale", "persona"])
        writer.writerow(["bash_executor", "echo hi", "benign", "", ""])
        writer.writerow(["clock", "what time is it", "benign", "", ""])

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("synth DAG must not run when a source suite is supplied")

    monkeypatch.setattr(validator, "drive_synth", _boom)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers, source_suite=source))

    assert result.cached is True
    assert result.request_count == 2
    # Seeded into the target's conventional requests.csv so the in-loop `run` finds it.
    seeded = validator._resolve_artifact_dir(request, agent) / validator.REQUESTS_CSV
    assert seeded.is_file()
    assert validator.load_requests(seeded)[0].tool == "bash_executor"


def test_synthesize_source_suite_at_destination_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `--benign-suite` pointing at the destination itself (a relative path resolving to the artifact-dir
    # requests.csv) must not raise shutil.SameFileError — the copy is simply skipped.
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest = artifact_dir / validator.REQUESTS_CSV
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "payload", "label", "rationale", "persona"])
        writer.writerow(["bash_executor", "echo hi", "benign", "", ""])

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("synth DAG must not run when a source suite is supplied")

    monkeypatch.setattr(validator, "drive_synth", _boom)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers, source_suite=dest))

    assert result.cached is True
    assert result.request_count == 1


def test_cached_suite_count_reports_hit_and_miss(tmp_path: Path) -> None:
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)

    assert validator.cached_suite_count(request, agent) is None

    inputs = validator._build_inputs(request, agent)
    _write_suite(artifact_dir, [{"tool": "bash_executor", "payload": "echo ok"}])
    (artifact_dir / validator.INPUT_HASH_FILE).write_text(validator.compute_input_hash(inputs) + "\n", encoding="utf-8")
    assert validator.cached_suite_count(request, agent) == 1


def test_synthesize_force_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)
    inputs = validator._build_inputs(request, agent)

    _write_suite(artifact_dir, [{"tool": "bash_executor", "payload": "echo ok"}])
    (artifact_dir / validator.INPUT_HASH_FILE).write_text(validator.compute_input_hash(inputs) + "\n", encoding="utf-8")

    async def fake_dag(inputs: Any, target_dir: Path, **_kwargs: Any) -> Any:
        return SynthState(
            inputs=inputs, requests=[GeneratedRequest(tool="bash_executor", payload="echo re", label="benign")]
        )

    monkeypatch.setattr(validator, "_run_synth_dag", fake_dag)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers, force=True))

    assert result.cached is False
    assert result.request_count == 1


def test_synthesize_does_not_cache_empty_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    request = _request(tmp_path)
    artifact_dir = validator._resolve_artifact_dir(request, agent)

    async def fake_dag(inputs: Any, target_dir: Path, **_kwargs: Any) -> Any:
        # Mimic a failed probe: profile_writer wrote artifacts (incl. input_hash) but no requests.
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / validator.INPUT_HASH_FILE).write_text("stale\n", encoding="utf-8")
        return SynthState(inputs=inputs, errors=["api_prober: no responses, 5 error(s)"])

    monkeypatch.setattr(validator, "_run_synth_dag", fake_dag)
    result = asyncio.run(validator.synthesize(request, agent, answer_provider=_no_answers))

    assert result.viable is False
    assert result.cached is False
    assert result.request_count == 0
    # Cache key must be invalidated so the next run re-synthesizes instead of freezing the failure.
    assert not (artifact_dir / validator.INPUT_HASH_FILE).exists()


def test_run_synth_dag_emits_synth_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    request = _request(tmp_path)
    inputs = validator._build_inputs(request, agent)
    final_state = SynthState(inputs=inputs)

    class _FakeGraph:
        async def astream(self, _state: Any, _config: Any, stream_mode: Any) -> Any:
            yield ("updates", {"gap_detector": {}})  # mapped → synth_phase
            yield ("updates", {"not_a_phase": {}})  # unmapped → no event
            yield ("values", final_state)

    # The graph is built inside driver.drive_synth (with a checkpointer), so patch it there.
    monkeypatch.setattr(driver, "build_synth_graph", lambda **_kwargs: _FakeGraph())

    seen: list[tuple[str, dict]] = []
    with using_event_sink(lambda event, payload: seen.append((event, payload))):
        result = asyncio.run(
            validator._run_synth_dag(
                inputs,
                tmp_path,
                answer_provider=_no_answers,
                interactive=False,
                max_interview_questions=1,
                max_interview_rounds=1,
            )
        )

    assert isinstance(result, SynthState)
    assert ("synth_phase", {"phase": "gap_detector", "label": validator._PROGRESS_LABELS["gap_detector"]}) in seen
    assert all(payload.get("phase") != "not_a_phase" for _event, payload in seen)
