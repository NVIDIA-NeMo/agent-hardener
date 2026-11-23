# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A structured answer cut off at the completion-token limit is asked for again, briefly, once."""

from __future__ import annotations

from typing import Any

import openai
import pytest
from langchain_core.messages import HumanMessage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agent_hardener.agents.defenders.traces_guardrails_defender.llm import CUT_OFF_NUDGE, invoke_structured


class Answer(BaseModel):
    value: str


def _cut_off() -> openai.LengthFinishReasonError:
    return openai.LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None))


class _Model:
    """Raises the cut-off error ``cut_offs`` times, then answers; records each prompt it was sent."""

    def __init__(self, cut_offs: int) -> None:
        self.cut_offs = cut_offs
        self.prompts: list[list[Any]] = []

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> _Model:
        return self

    def invoke(self, prompt: list[Any]) -> dict[str, str]:
        self.prompts.append(prompt)
        if len(self.prompts) <= self.cut_offs:
            raise _cut_off()
        return {"value": "ok"}


def test_a_cut_off_answer_is_asked_for_again_briefly() -> None:
    model = _Model(cut_offs=1)
    prompt = [HumanMessage(content="analyze")]

    assert invoke_structured(model, Answer, prompt) == Answer(value="ok")  # type: ignore[arg-type]
    assert model.prompts[0] == prompt
    assert model.prompts[1] == [*prompt, HumanMessage(content=CUT_OFF_NUDGE)]


def test_an_answer_still_cut_off_after_the_retry_raises() -> None:
    model = _Model(cut_offs=2)

    with pytest.raises(openai.LengthFinishReasonError):
        invoke_structured(model, Answer, [HumanMessage(content="analyze")])  # type: ignore[arg-type]
    assert len(model.prompts) == 2
