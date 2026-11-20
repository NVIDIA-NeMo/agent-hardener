# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON wrapped in reasoning, fences or prose reads as the schema; anything that is not JSON never does."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent_hardener.relay_plugin.json_reply import JsonReplyError, parse_json_reply, reply_text

pytestmark = pytest.mark.unit


class Verdict(BaseModel):
    is_refused: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


EXPECTED = Verdict(is_refused=True, confidence=0.9, reasoning="it said no")
CLEAN = '{"is_refused": true, "confidence": 0.9, "reasoning": "it said no"}'


@pytest.mark.parametrize(
    "reply",
    [
        CLEAN,
        f"```json\n{CLEAN}\n```",
        f"Here is my verdict:\n\n```\n{CLEAN}\n```\n\nLet me know if you need more.",
        f"Okay, let's see. The user asked {{so}} I think it refused.\n\n{CLEAN}",  # nemotron-nano-9b-v2
        f'<think>Draft: {{"is_refused": false}}</think>\n{CLEAN}',
        f"I should check the response...</think>{CLEAN}",  # the template emitted the opening tag
        f"<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>{CLEAN}<|return|>",
        f'{{"draft": 1}} then {CLEAN}',  # a value that does not validate is skipped
        f'First guess: {{"is_refused": false, "confidence": 0.5}}. On reflection:\n{CLEAN}',  # the last answer wins
        f"{CLEAN}\n\nHope this helps!",
    ],
)
def test_json_around_reasoning_fences_or_prose_reads_as_the_schema(reply: str) -> None:
    assert parse_json_reply(reply, Verdict) == EXPECTED


class Tuple(BaseModel):
    host: str = ""


class Tuples(BaseModel):
    tuples: list[Tuple] = Field(default_factory=list)


def test_an_object_nested_in_the_answer_never_stands_in_for_it() -> None:
    # {"host": "b"} alone would validate as Tuples(tuples=[]): fields with defaults, extra keys ignored.
    reply = 'Done: {"tuples": [{"host": "a"}, {"host": "b"}]}'
    assert parse_json_reply(reply, Tuples) == Tuples(tuples=[Tuple(host="a"), Tuple(host="b")])


def test_a_raw_newline_inside_a_string_is_accepted() -> None:
    reply = '{"is_refused": true, "confidence": 0.9, "reasoning": "line one\nline two"}'
    assert parse_json_reply(reply, Verdict).reasoning == "line one\nline two"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("", "empty reply"),
        ("<think>only thinking</think>", "empty reply"),
        ("BLOCK", "no JSON found"),
        ('{"is_refused": true, "confidence": 0.', "no JSON found"),  # cut off: never half-read
        ("{'is_refused': True, 'confidence': 0.9}", "no JSON found"),  # not JSON: left to the retry
        (f"[{CLEAN}]", "<root>: Input should be a valid dictionary or instance of Verdict"),
        ('{"is_refused": true, "confidence": 7}', "confidence: Input should be less than or equal to 1"),
    ],
)
def test_anything_else_raises_with_the_reason_and_the_reply(reply: str, reason: str) -> None:
    with pytest.raises(JsonReplyError, match=reason) as raised:
        parse_json_reply(reply, Verdict)
    assert (raised.value.reason, raised.value.reply) == (reason, reply)


def test_a_long_reply_is_quoted_in_part() -> None:
    with pytest.raises(JsonReplyError, match=r"\.\.\. \[1000 chars\]"):
        parse_json_reply("x" * 1000, Verdict)


def test_reply_text_reads_text_blocks_and_never_the_reasoning() -> None:
    blocks = [{"type": "reasoning", "reasoning": "x"}, {"type": "text", "text": "a"}, "b"]
    assert reply_text(AIMessage(content=blocks)) == "ab"
    # A model that drafted a verdict while thinking but never answered has no verdict.
    draft = AIMessage(content="", additional_kwargs={"reasoning_content": '{"is_refused": false, "confidence": 1}'})
    with pytest.raises(JsonReplyError, match="empty reply"):
        parse_json_reply(reply_text(draft), Verdict)
