# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for profile_synthesizer.normalize: LLM-normalized globals, tool patches, and fallbacks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.nodes import normalize as nz
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.schemas import (
    NormalizedPersona,
    NormalizedProfileGlobals,
    ToolMergeProposal,
    ToolPatch,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.state import ProfileSynthesizerState
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable


def _tool(name: str, **overrides: Any) -> ToolSpec:
    base = {"name": name, "description": f"{name} desc", "source": "nl", "confidence": 0.8}
    return ToolSpec(**{**base, **overrides})


def _state(*tools: ToolSpec, **overrides: Any) -> ProfileSynthesizerState:
    base = {
        "target_name": "victim",
        "input_hash": "h1",
        "merged_tools": list(tools),
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return ProfileSynthesizerState(**{**base, **overrides})


def _patch(monkeypatch: pytest.MonkeyPatch, model: Any) -> Any:
    monkeypatch.setattr(nz, "build_chat_model", lambda **_kw: model)
    return model


def _profile(delta: dict[str, object]) -> VictimCapabilityProfile:
    return cast("VictimCapabilityProfile", delta["profile"])


def _by_name(profile: VictimCapabilityProfile) -> dict[str, ToolSpec]:
    return {t.name: t for t in profile.tools}


# --- normalize ---------------------------------------------------------------------------------------------


def test_no_merged_tools_builds_an_empty_profile_without_calling_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nz, "build_chat_model", lambda **_kw: pytest.fail("must not call the LLM"))

    delta = asyncio.run(nz.normalize(_state(source_notes={"nl_parser": "n"})))

    profile = _profile(delta)
    assert (profile.tools, profile.personas, profile.out_of_scope) == ([], [], [])
    assert profile.source_notes == {"nl_parser": "n"}
    assert profile.input_hash == "h1"
    assert delta["source_note"] == "no tools discovered"


def test_llm_globals_populate_the_profile(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(
        monkeypatch,
        fake_model(
            NormalizedProfileGlobals(
                system_role="support bot",
                personas=[NormalizedPersona(name="Dana", description="a nurse")],
                out_of_scope=["legal advice"],
                proposed_tool_merges=[
                    ToolMergeProposal(canonical_name="bash", aliases=["shell"], rationale="same thing")
                ],
            )
        ),
    )

    delta = asyncio.run(nz.normalize(_state(_tool("bash"), _tool("shell"))))

    profile = _profile(delta)
    assert profile.system_role == "support bot"
    assert [(p.name, p.description) for p in profile.personas] == [("Dana", "a nurse")]
    assert profile.out_of_scope == ["legal advice"]
    assert [(m.canonical_name, m.aliases) for m in profile.proposed_tool_merges] == [("bash", ["shell"])]
    assert [t.name for t in profile.tools] == ["bash", "shell"]
    assert delta["source_note"] == "merged 2 tool(s); 1 persona(s)"
    assert "errors" not in delta


def test_the_prompt_lists_tools_gaps_notes_and_interview_answers(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = _patch(monkeypatch, fake_model(NormalizedProfileGlobals()))
    state = _state(
        _tool("bash", example_inputs=["ls"]),
        _tool("bare", description=" ", example_inputs=[]),
        source_notes={"github_analyzer": "read 3 files"},
        interviewer_answers=[("scope gap", "Who uses it?", "Nurses"), ("", "Free-form?", "Yes")],
    )

    asyncio.run(nz.normalize(state))

    system, user = model.messages
    assert system.content
    text = user.content
    assert "bash (nl, confidence=0.80): bash desc" in text
    assert "- bare: missing description" in text
    assert "- bare: missing example_inputs" in text
    assert "- github_analyzer: read 3 files" in text
    assert "Gap: scope gap\nQ: Who uses it?\nA: Nurses" in text
    assert "Q: Free-form?\nA: Yes" in text
    assert "Gap: \n" not in text  # no empty gap header for pre-seeded answers


def test_prompt_placeholders_for_empty_inputs(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    model = _patch(monkeypatch, fake_model(NormalizedProfileGlobals()))

    asyncio.run(nz.normalize(_state(_tool("bash", example_inputs=["ls"]))))

    text = model.messages[1].content
    assert text.count("(none)") >= 3  # gaps, source notes, interview answers


def test_only_the_first_twenty_tools_are_summarised(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = _patch(monkeypatch, fake_model(NormalizedProfileGlobals()))
    tools = [_tool(f"tool_{i:02d}", example_inputs=["x"]) for i in range(23)]

    delta = asyncio.run(nz.normalize(_state(*tools)))

    text = model.messages[1].content
    assert "tool_19" in text
    assert "- tool_20 (" not in text
    assert "... and 3 more" in text
    assert len(_profile(delta).tools) == 23  # the cap is prompt-only; no tool is dropped from the profile


def test_an_llm_failure_keeps_the_tools_and_reports_the_error(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=ValueError("unparseable")))

    delta = asyncio.run(nz.normalize(_state(_tool("bash"))))

    profile = _profile(delta)
    assert [t.name for t in profile.tools] == ["bash"]
    assert (profile.system_role, profile.personas, profile.out_of_scope) == (None, [], [])
    assert delta["errors"] == ["profile_synthesizer.normalize: unparseable"]
    assert "LLM normalize failed" in str(delta["source_note"])


def test_a_rate_limit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(monkeypatch, fake_model(raises=RateLimitError("slow down", source="llm")))

    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(nz.normalize(_state(_tool("bash"))))


def test_a_provider_rate_limit_message_is_promoted(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=RuntimeError("quota exceeded")))

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(nz.normalize(_state(_tool("bash"))))

    assert exc_info.value.source == "smart_benign:profile_synthesizer"


def test_tool_patches_flow_through_normalize(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    result = NormalizedProfileGlobals(tool_patches=[ToolPatch(name="bash", description="runs shell")])
    _patch(monkeypatch, fake_model(result))

    delta = asyncio.run(nz.normalize(_state(_tool("bash", description="  "))))

    assert _by_name(_profile(delta))["bash"].description == "runs shell"


# --- _apply_tool_patches -----------------------------------------------------------------------------------


def test_no_patches_returns_the_tools_untouched() -> None:
    tools = [_tool("bash")]

    assert nz._apply_tool_patches(tools, []) is tools


def test_a_patch_only_fills_blank_description_and_examples() -> None:
    tools = [_tool("bash", description="kept", example_inputs=["kept-ex"]), _tool("py", description=" ")]
    patches = [
        ToolPatch(name="bash", description="ignored", example_inputs=["ignored-ex"]),
        ToolPatch(name="py", description="filled", example_inputs=["filled-ex"]),
    ]

    result = {t.name: t for t in nz._apply_tool_patches(tools, patches)}

    assert (result["bash"].description, result["bash"].example_inputs) == ("kept", ["kept-ex"])
    assert (result["py"].description, result["py"].example_inputs) == ("filled", ["filled-ex"])


def test_boundary_patterns_are_unioned_in_order_without_duplicates() -> None:
    tools = [_tool("bash", allowed_patterns=["ls", "pwd"], blocked_patterns=["rm"])]
    patch = ToolPatch(name="bash", allowed_patterns=["pwd", "cat"], blocked_patterns=["dd", "rm"])

    (result,) = nz._apply_tool_patches(tools, [patch])

    assert result.allowed_patterns == ["ls", "pwd", "cat"]
    assert result.blocked_patterns == ["rm", "dd"]


def test_a_drop_patch_removes_the_tool() -> None:
    tools = [_tool("bash"), _tool("shell")]

    result = nz._apply_tool_patches(tools, [ToolPatch(name="shell", drop=True)])

    assert [t.name for t in result] == ["bash"]


def test_a_patch_for_an_unknown_tool_adds_it_as_an_interview_tool() -> None:
    patch = ToolPatch(name="web_search", description="searches", example_inputs=["cats"], blocked_patterns=["nsfw"])

    result = nz._apply_tool_patches([_tool("bash")], [patch])

    added = result[-1]
    assert (added.name, added.source, added.confidence) == ("web_search", "interview", 0.7)
    assert (added.example_inputs, added.blocked_patterns) == (["cats"], ["nsfw"])


@pytest.mark.parametrize(
    "patch",
    [ToolPatch(name="ghost"), ToolPatch(name="ghost", drop=True, description="x")],
    ids=["no-description", "dropped"],
)
def test_unknown_tool_patches_without_a_description_or_with_drop_are_ignored(patch: ToolPatch) -> None:
    tools = [_tool("bash")]

    assert [t.name for t in nz._apply_tool_patches(tools, [patch])] == ["bash"]


def test_patching_does_not_mutate_the_input_tools() -> None:
    original = _tool("bash", description=" ")

    nz._apply_tool_patches([original], [ToolPatch(name="bash", description="new")])

    assert original.description == " "


def test_the_prompt_lists_only_the_tools_missing_example_inputs(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = _patch(monkeypatch, fake_model(NormalizedProfileGlobals()))

    asyncio.run(nz.normalize(_state(_tool("a", example_inputs=["x"]), _tool("b"))))

    text = model.messages[1].content
    assert "- b: missing example_inputs" in text
    assert "- a: missing" not in text
