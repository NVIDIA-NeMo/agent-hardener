# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool verification run: passes, drop decisions, observed tools, and what lands on disk."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.smart_benign import tool_verification as tv
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.validator import load_requests
from agent_hardener.agents.validators.smart_benign.verify import ObservedCall, ToolVerdict, VerificationResult
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from agent_hardener.traces import AgentTrace


def _req(tool: str, payload: str) -> GeneratedRequest:
    return GeneratedRequest(tool=tool, payload=payload, label="benign")


def _spec(name: str, **kw: Any) -> ToolSpec:
    return ToolSpec(name=name, description="d", source=kw.pop("source", "nl"), confidence=1.0, **kw)


SUITE = [_req("read_balance", "my balance"), _req("teleport", "teleport me"), _req("read_balance", "balance ACC-9")]


def _state(requests: list[GeneratedRequest] = SUITE) -> SynthState:
    profile = VictimCapabilityProfile(
        target_name="bank",
        system_role="a bank bot",
        tools=[_spec("read_balance"), _spec("teleport")],
        input_hash="h",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return SynthState(inputs=SynthInputs(target_name="bank"), profile=profile, requests=requests)


def _verdict(tool: str, status: str) -> ToolVerdict:
    return ToolVerdict(tool=tool, status=status, rows=1, traced_rows=1, rows_calling_it=0, calls_anywhere=0)  # type: ignore[arg-type]


def _result(
    tools: list[ToolVerdict], *, refused: list[int] = (), observed: tuple[str, ...] = (), traced: bool = True
) -> VerificationResult:  # type: ignore[assignment]
    calls = {name: [ObservedCall(tool=name, prompt="my balance")] for name in observed}
    return VerificationResult(
        traced=traced, rows=3, traced_rows=3, tools=tools, observed_unlisted=calls, baseline_refused=list(refused)
    )


class _Script:
    """Scripted collaborators: one verification result per pass, and recorded calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, results: list[VerificationResult], new_rows: int = 1) -> None:
        self.batches: list[list[GeneratedRequest]] = []
        self.known: list[list[str]] = []
        results = list(results)

        async def verify_suite(batch: list[GeneratedRequest], _cfg: Any, _window: Any, **kw: Any) -> VerificationResult:
            self.batches.append(batch)
            self.known.append(list(kw["known_tools"]))
            return results.pop(0)

        async def describe(observed: dict[str, Any], role: str | None) -> tuple[list[ToolSpec], list[str]]:
            assert role == "a bank bot"
            return [_spec(n, source="trace", verification="verified") for n in observed], ["describe note"]

        async def generate(state: SynthState, specs: list[ToolSpec]) -> tuple[list[GeneratedRequest], list[str]]:
            assert state.profile is not None
            rows = [_req(s.name, f"use {s.name} {i}") for s in specs for i in range(new_rows)]
            return rows, []

        monkeypatch.setattr(tv, "verify_suite", verify_suite)
        monkeypatch.setattr(tv, "describe_observed_tools", describe)
        monkeypatch.setattr(tv, "generate_for_tools", generate)

    def run(
        self, decide: Any = tv.default_drop_decider, provider: Any = None, state: SynthState | None = None
    ) -> tv.ToolVerificationOutcome:
        return asyncio.run(
            tv.run_tool_verification(
                state or _state(),
                cfg=None,
                provider=provider or FakeTraceProvider(),
                decide=decide,  # type: ignore[arg-type]
            )
        )


def test_a_full_run_drops_the_missing_tool_and_verifies_rows_for_the_observed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        monkeypatch,
        [
            _result(
                [_verdict("read_balance", "verified"), _verdict("teleport", "missing")],
                refused=[2],
                observed=("lookup_contact",),
            ),
            # Pass 2 replays only the new rows; their tool not being called again does not un-verify it.
            _result([_verdict("lookup_contact", "missing")]),
        ],
    )

    outcome = script.run()

    assert [len(b) for b in script.batches] == [3, 1]
    assert script.known[1] == ["read_balance", "teleport", "lookup_contact"]
    assert [(r.tool, r.payload) for r in outcome.requests] == [
        ("read_balance", "my balance"),
        ("read_balance", "balance ACC-9"),  # refused, but the default keeps refused rows
        ("lookup_contact", "use lookup_contact 0"),
    ]
    assert outcome.profile is not None
    assert {t.name: t.verification for t in outcome.profile.tools} == {
        "read_balance": "verified",
        "teleport": "missing",
        "lookup_contact": "verified",
    }
    first, second = outcome.passes
    assert first.decision.decided_by == "default"
    assert [r.tool for r in first.dropped_rows] == ["teleport"]
    assert (first.added_tools, first.added_rows, first.errors) == (["lookup_contact"], 1, ["describe note"])
    assert second.decision.decided_by == "nothing_to_decide"
    assert (outcome.skipped, outcome.unhandled_tools) == (None, [])


def test_the_decider_can_drop_refused_rows_and_keep_a_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, [_result([_verdict("teleport", "missing")], refused=[2])])
    seen: list[Any] = []

    async def decide(missing: list[ToolVerdict], refused: list[Any], batch: list[GeneratedRequest]) -> tv.DropDecision:
        seen.append(([v.tool for v in missing], [i for i, _ in refused], len(batch)))
        return tv.DropDecision(tools=[], rows=[2], decided_by="operator")

    outcome = script.run(decide)

    assert seen == [(["teleport"], [2], 3)]
    assert [r.payload for r in outcome.requests] == ["my balance", "teleport me"]


def test_tools_seen_on_the_last_allowed_pass_are_reported_not_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    config = tv.defaults()
    monkeypatch.setattr(config.verification, "max_passes", 1)
    script = _Script(monkeypatch, [_result([_verdict("read_balance", "verified")], observed=("lookup_contact",))])

    outcome = script.run()

    assert outcome.unhandled_tools == ["lookup_contact"]
    assert len(outcome.passes) == 1
    assert outcome.passes[0].added_tools == []


def test_no_generated_rows_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, [_result([], observed=("lookup_contact",))], new_rows=0)

    outcome = script.run()

    assert len(script.batches) == 1
    assert outcome.passes[0].added_tools == ["lookup_contact"]


def test_a_failed_trace_step_leaves_the_suite_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, [_result([_verdict("teleport", "unknown")], traced=False)])

    outcome = script.run()

    assert outcome.requests == SUITE
    assert outcome.skipped == "the trace step failed; the suite is unchanged"
    assert len(outcome.passes) == 1


def test_unreadable_traces_skip_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(FakeTraceProvider):
        def get_traces(self) -> list[AgentTrace]:
            raise RuntimeError("no sandbox")

    script = _Script(monkeypatch, [])

    outcome = script.run(provider=Broken())

    assert (outcome.requests, outcome.passes, outcome.skipped) == (SUITE, [], "could not read the victim's traces")


def test_an_empty_suite_is_not_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _Script(monkeypatch, []).run(state=_state([]))
    assert outcome.skipped == "the suite is empty"


def _outcome(passes: bool) -> tv.ToolVerificationOutcome:
    state = _state()
    record = tv.PassRecord(number=1, result=_result([]), decision=tv.DropDecision(decided_by="default"))
    return tv.ToolVerificationOutcome(
        requests=SUITE[:1], profile=state.profile, passes=[record] if passes else [], skipped=None if passes else "x"
    )


def test_write_verification_rewrites_the_suite_and_keeps_the_synthesized_one(tmp_path: Path) -> None:
    (tmp_path / tv.REQUESTS_CSV).write_text("tool,payload,label,rationale,persona\nt,p,benign,,\n", encoding="utf-8")

    path = tv.write_verification(tmp_path, _outcome(passes=True))

    assert [r.payload for r in load_requests(tmp_path / tv.REQUESTS_CSV)] == ["my balance"]
    assert [r.payload for r in load_requests(tmp_path / tv.PRE_VERIFICATION_CSV)] == ["p"]
    assert json.loads((tmp_path / tv.PROFILE_JSON).read_text())["target_name"] == "bank"
    written = json.loads(path.read_text())
    assert "requests" not in written
    assert written["passes"][0]["number"] == 1


def test_write_verification_without_a_pass_only_writes_the_report(tmp_path: Path) -> None:
    tv.write_verification(tmp_path, _outcome(passes=False))
    assert sorted(p.name for p in tmp_path.iterdir()) == [tv.VERIFICATION_JSON]


class _Checkbox:
    def __init__(self, answers: list[Any]) -> None:
        self.answers, self.prompts = answers, []

    def __call__(self, message: str, choices: list[Any]) -> Any:
        self.prompts.append((message, [(c.title, c.value, c.checked) for c in choices]))
        answer = self.answers.pop(0)
        return type("Q", (), {"ask": staticmethod(lambda: answer)})()


def test_the_tty_prompt_preselects_missing_tools_and_not_refused_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    checkbox = _Checkbox([["teleport"], [2]])
    monkeypatch.setattr(tv.questionary, "checkbox", checkbox)
    missing = [_verdict("teleport", "missing")]

    decision = asyncio.run(tv.tty_drop_decider(missing, [(2, SUITE[2])], SUITE))

    assert decision == tv.DropDecision(tools=["teleport"], rows=[2], decided_by="operator")
    (_, tools), (_, rows) = checkbox.prompts
    assert tools == [("teleport — 1/1 row(s) traced, e.g. teleport me", "teleport", True)]
    assert rows == [("[read_balance] balance ACC-9", 2, False)]


def test_ctrl_c_in_the_tty_prompt_takes_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tv.questionary, "checkbox", _Checkbox([None, None]))
    long_row = _req("read_balance", "x" * 300)

    decision = asyncio.run(tv.tty_drop_decider([_verdict("teleport", "missing")], [(0, long_row)], SUITE))

    assert decision == tv.DropDecision(tools=["teleport"], rows=[], decided_by="default")


def test_the_default_decider_drops_missing_tools_only() -> None:
    decision = asyncio.run(tv.default_drop_decider([_verdict("teleport", "missing")], [(0, SUITE[0])], SUITE))
    assert decision == tv.DropDecision(tools=["teleport"], decided_by="default")
