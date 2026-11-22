# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rows for tools the traces show but the profile does not list: describe them, then generate + critique."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign import regenerate
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.nodes.merge import SOURCE_PRIORITY
from agent_hardener.agents.validators.smart_benign.verify import ObservedCall
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable

CALLS = {
    "lookup_contact": [
        ObservedCall(tool="lookup_contact", args={"name": "Ann"}, result_excerpt="ann@x", prompt="email Ann"),
        ObservedCall(tool="lookup_contact", args={"name": "Bo"}, result_excerpt="bo@x", prompt="email Bo"),
    ],
    "get_weather": [ObservedCall(tool="functions.get_weather", prompt="weather in Paris")],
}


def test_trace_outranks_every_other_source() -> None:
    assert SOURCE_PRIORITY["trace"] > max(v for k, v in SOURCE_PRIORITY.items() if k != "trace")


def test_extract_spec_is_built_from_the_trace_evidence_alone() -> None:
    spec = regenerate.extract_spec("lookup_contact", CALLS["lookup_contact"])

    assert (spec.source, spec.verification, spec.confidence) == ("trace", "verified", 1.0)
    assert (
        spec.description == 'Tool `lookup_contact` the agent was observed calling with arguments like {"name": "Ann"}.'
    )
    assert spec.example_inputs == ["email Ann", "email Bo"]  # the real prompts that triggered it
    no_args = regenerate.extract_spec("get_weather", CALLS["get_weather"])
    assert no_args.description == "Tool `get_weather` the agent was observed calling."


def test_extract_spec_clips_long_arguments() -> None:
    call = ObservedCall(tool="t", args={"blob": "x" * 500}, prompt="p")
    assert regenerate.extract_spec("t", [call]).description.endswith("….")


def test_refinement_cleans_the_description_and_adds_examples_after_the_observed_ones(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    refined = regenerate.ToolRefinement(
        description="Finds a contact's email address.",
        extra_example_inputs=["find Cy", " ", "email Ann", "a", "b", "c"],
    )
    model = fake_model(refined)
    monkeypatch.setattr(regenerate, "build_chat_model", lambda **_k: model)

    specs, errors = asyncio.run(
        regenerate.describe_observed_tools({"lookup_contact": CALLS["lookup_contact"]}, "a bot")
    )

    assert errors == []
    (spec,) = specs
    assert spec.description == "Finds a contact's email address."
    # Observed first; blanks and repeats dropped before the cap of 3 new ones.
    assert spec.example_inputs == ["email Ann", "email Bo", "find Cy", "a", "b"]
    assert spec.source == "trace"
    prompt = model.messages[1].content
    assert "a bot" in prompt
    assert "observed calling with arguments like" in prompt  # the extracted description is the starting point
    assert "email Bo" in prompt


def test_a_failed_refinement_keeps_the_extracted_spec(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    monkeypatch.setattr(regenerate, "build_chat_model", lambda **_k: fake_model(raises=ValueError("bad json")))

    specs, errors = asyncio.run(regenerate.describe_observed_tools(CALLS, None))

    assert specs == [regenerate.extract_spec(name, calls) for name, calls in CALLS.items()]
    assert errors == ["observed_tool[lookup_contact]: bad json", "observed_tool[get_weather]: bad json"]


def test_a_rate_limit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    limited = RateLimitError("429", source="x")
    monkeypatch.setattr(regenerate, "build_chat_model", lambda **_k: fake_model(raises=RuntimeError("429")))
    monkeypatch.setattr(regenerate, "rate_limit_error_from_exception", lambda _exc, **_k: limited)

    with pytest.raises(RateLimitError):
        asyncio.run(regenerate.describe_observed_tools(CALLS, None))


class _Agent:
    """A compiled subgraph stand-in: records the state it got and fills ``field`` with ``make(state)``."""

    def __init__(self, field: str, make: Callable[[Any], list[GeneratedRequest]]) -> None:
        self.field, self.make = field, make
        self.seen: Any = None

    async def ainvoke(self, state: Any) -> dict[str, Any]:
        self.seen = state
        return {**state.model_dump(), self.field: self.make(state), "errors": [f"{self.field} note"]}


def _state() -> SynthState:
    profile = VictimCapabilityProfile(
        target_name="bot",
        system_role="an email bot",
        tools=[ToolSpec(name="send_email", description="sends mail", source="nl", confidence=0.9)],
        input_hash="h",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    old = [GeneratedRequest(tool="send_email", payload="mail Bo", label="benign")]
    return SynthState(inputs=SynthInputs(target_name="bot"), profile=profile, requests=old)


def test_generate_for_tools_runs_generator_then_critic_on_the_new_tools_only(monkeypatch: pytest.MonkeyPatch) -> None:
    new = ToolSpec(name="lookup_contact", description="finds", source="trace", confidence=1.0)
    gen = _Agent(
        "generated", lambda s: [GeneratedRequest(tool=t.name, payload="find Ann", label="benign") for t in s.tools]
    )
    crit = _Agent("filtered_requests", lambda s: list(s.requests))
    monkeypatch.setattr(regenerate.request_generator, "build_request_generator_agent", lambda: gen)
    monkeypatch.setattr(regenerate.critic, "build_critic_agent", lambda: crit)

    rows, errors = asyncio.run(regenerate.generate_for_tools(_state(), [new]))

    assert [(r.tool, r.payload) for r in rows] == [("lookup_contact", "find Ann")]
    assert [t.name for t in gen.seen.tools] == ["lookup_contact"]
    assert gen.seen.system_role == "an email bot"
    assert crit.seen.tool_names == ["lookup_contact"]  # the existing suite never reaches the critic
    assert errors == ["generated note", "filtered_requests note"]


def test_generate_for_tools_without_profile_or_tools_is_a_no_op() -> None:
    assert asyncio.run(regenerate.generate_for_tools(_state(), [])) == ([], [])
    no_profile = _state().model_copy(update={"profile": None})
    assert asyncio.run(
        regenerate.generate_for_tools(no_profile, [ToolSpec(name="x", description="d", source="trace", confidence=1)])
    ) == ([], [])
