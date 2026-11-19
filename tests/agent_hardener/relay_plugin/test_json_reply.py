# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every common way an LLM wraps or mangles JSON still reads as the schema; a non-answer never does."""

from __future__ import annotations

from typing import Literal

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field

from agent_hardener.relay_plugin.json_reply import JsonReplyError, parse_json_reply, reply_parts, strip_reasoning

pytestmark = pytest.mark.unit


class Row(BaseModel):
    payload: str
    label: Literal["benign", "negative_control"] = "benign"


class Verdict(BaseModel):
    is_refused: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    rows: list[Row] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class Score(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


class Tags(BaseModel):
    tags: list[str]
    count: int = 0


EXPECTED = Verdict(is_refused=True, confidence=0.9, reasoning="it said no")
CLEAN = '{"is_refused": true, "confidence": 0.9, "reasoning": "it said no"}'


@pytest.mark.parametrize(
    "reply",
    [
        CLEAN,
        f"```json\n{CLEAN}\n```",
        f"```\n{CLEAN}\n```",
        f"~~~JSON\n{CLEAN}\n~~~",
        f"Here is my verdict:\n\n```json\n{CLEAN}\n```\n\nLet me know if you need more.",
        f"Sure! {CLEAN} Hope this helps.",
        f'<think>The agent refused. Draft: {{"is_refused": false}}</think>\n{CLEAN}',
        f"I should check the response...</think>{CLEAN}",  # template emitted the opening tag
        f"<reasoning>hmm</reasoning><answer>{CLEAN}</answer>",
        f"<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>{CLEAN}<|return|>",
        f"\ufeff\u200b{CLEAN}",
        "{'is_refused': True, 'confidence': 0.9, 'reasoning': 'it said no'}",  # Python repr
        '{is_refused: true, confidence: .9, reasoning: "it said no",}',  # JS object, trailing comma
        '{\n  // verdict\n  "is_refused": true, /* sure */\n  "confidence": 0.9,\n  "reasoning": "it said no"\n}',
        '{"is_refused": true "confidence": 0.9\n "reasoning": "it said no"}',  # missing commas
        "{\u201cis_refused\u201d: true, \u201cconfidence\u201d: 0.9, \u201creasoning\u201d: \u201cit said no\u201d}",
        '{"is_refused": "yes", "confidence": "90%", "reasoning": "it said no"}',
        '{"isRefused": true, "Confidence": 0.9, "reasoning": "it said no"}',
        '{"verdict": {"is_refused": true, "confidence": 0.9, "reasoning": "it said no"}}',
        '{"name": "Verdict", "arguments": "{\\"is_refused\\": true, \\"confidence\\": 0.9, \\"reasoning\\": \\"it said no\\"}"}',
        f'"{CLEAN.replace(chr(34), chr(92) + chr(34))}"',  # double-encoded
        f"[{CLEAN}]",
    ],
)
def test_wrapped_or_mangled_json_reads_as_the_schema(reply: str) -> None:
    assert parse_json_reply(reply, Verdict) == EXPECTED


@pytest.mark.parametrize(
    ("reply", "reasoning"),
    [
        (
            'Verdict: {"is_refused": false, "confidence": 1, "reasoning": "C:\\Users said "no" \\_o_/ \\"q\\"\nok"}',
            'C:\\Users said "no" \\_o_/ "q"\nok',
        ),
        (
            "{'is_refused': false, 'confidence': 1, 'reasoning': 'caf\\u00e9 it\\'s \"x\"\\n\t\r\x01'}",
            'café it\'s "x"\n\t\r\x01',
        ),
    ],
)
def test_string_content_is_repaired_without_being_rewritten(reply: str, reasoning: str) -> None:
    assert parse_json_reply(reply, Verdict).reasoning == reasoning


def test_nested_fields_are_fixed_where_validation_points() -> None:
    reply = (
        '{"is_refused": false, "confidence": 0.5, "reasoning": 7, "tags": "solo",'
        ' "rows": "[{\\"payload\\": \\"ls\\", \\"label\\": \\"Negative-Control\\"}]"}'
    )
    verdict = parse_json_reply(reply, Verdict)
    assert (verdict.reasoning, verdict.tags) == ("7", ["solo"])
    assert verdict.rows == [Row(payload="ls", label="negative_control")]


@pytest.mark.parametrize("fence", ["```json\n", "Result: "])
def test_a_reply_cut_off_at_max_tokens_keeps_what_is_complete(fence: str) -> None:
    reply = fence + '{"is_refused": false, "confidence": 0.4, "rows": [{"payload": "a"}, {"payload": "b"}, {"pay'
    assert [row.payload for row in parse_json_reply(reply, Verdict).rows] == ["a", "b"]
    assert parse_json_reply('{"is_refused": 0, "confidence": -.0, "reasoning":', Verdict).confidence == 0.0


def test_a_bare_list_fills_the_one_list_field() -> None:
    assert parse_json_reply('["a", "b"]', Tags) == Tags(tags=["a", "b"])
    assert parse_json_reply('{"tags": null, "count": "007 items"}', Tags) == Tags(tags=[], count=7)
    assert parse_json_reply('{"tags": {"items": ["a"]}}', Tags) == Tags(tags=["a"])
    assert parse_json_reply('{"tags": 5}', Tags) == Tags(tags=["5"])


def test_extra_keys_are_dropped_only_when_the_schema_forbids_them() -> None:
    assert parse_json_reply('{"name": "x", "note": "extra"}', Strict) == Strict(name="x")


def test_a_bare_reply_stands_for_a_single_field_schema_only_when_it_is_the_whole_reply() -> None:
    assert parse_json_reply("0.8", Score) == Score(score=0.8)
    with pytest.raises(JsonReplyError):
        parse_json_reply("See [1] for details", Score)
    with pytest.raises(JsonReplyError):
        parse_json_reply("0.8", Score, allow_bare=False)


@pytest.mark.parametrize("reply", ["BLOCK", "ALLOW ping 127.0.0.1", "1.5", '{"score": "high"}', ""])
def test_a_non_answer_raises_with_the_reply(reply: str) -> None:
    with pytest.raises(JsonReplyError) as raised:
        parse_json_reply(reply, Score)
    assert raised.value.reply == reply


def test_the_error_names_the_closest_validation_failure() -> None:
    with pytest.raises(JsonReplyError, match=r"Verdict.*confidence: Input should be less than or equal to 1"):
        parse_json_reply('```json\n{"is_refused": true, "confidence": 7}\n```', Verdict)


def test_tool_call_arguments_are_tried_before_the_text() -> None:
    assert parse_json_reply(CLEAN, Verdict, objects=[{"unrelated": 1}]) == EXPECTED
    assert (
        parse_json_reply("", Verdict, objects=[{"is_refused": True, "confidence": 0.9, "reasoning": "it said no"}])
        == EXPECTED
    )


def test_reply_parts_reads_content_blocks_reasoning_and_tool_calls() -> None:
    message = AIMessage(
        content=[{"type": "reasoning", "reasoning": "thinking"}, {"type": "text", "text": "answer"}],
        additional_kwargs={"reasoning_content": "more"},
        tool_calls=[{"name": "Verdict", "args": {"is_refused": True, "confidence": 0.9}, "id": "1"}],
    )
    assert reply_parts(message) == (["answer", "thinking", "more"], [{"is_refused": True, "confidence": 0.9}])
    texts, _ = reply_parts(AIMessage(content="", additional_kwargs={"reasoning_content": CLEAN}))
    assert parse_json_reply(texts, Verdict) == EXPECTED


def test_strip_reasoning_keeps_an_unclosed_thought() -> None:
    assert strip_reasoning("<think>0.4") == "0.4"
