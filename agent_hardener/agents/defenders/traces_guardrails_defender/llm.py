# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA-hosted chat model for the guardrails defender's nodes (:func:`agent_hardener.llm.build_chat_model`).

The model is :func:`agent_hardener.llm.default_model` (``AGENT_HARDENER_MODEL``) for every role. The defender builds it
once per attack for the analyzer, the benign-args writer and the suggestor, and once more at temperature 0 for
the grader, and hands both to the graph, so a test can swap in a scripted model.

Every structured answer goes through :func:`invoke_structured`, which asks once more, for a short answer, when a
reply runs into the completion-token limit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from agent_hardener.llm import build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)

#: Low but not zero: the analysis should be repeatable, the rail wording may vary.
TEMPERATURE = 0.2
#: The grader scores drafts against a rubric, so it should give the same draft the same score.
GRADER_TEMPERATURE = 0.0


def get_llm(temperature: float = TEMPERATURE) -> ChatOpenAI:
    """Return the shared NVIDIA-hosted chat model for this defender's nodes."""
    return build_chat_model(temperature=temperature)


#: How many more times a structured answer is asked for after it was cut off at the completion-token limit.
LENGTH_RETRIES = 1
#: Appended to the prompt for that retry. A model that ran out of tokens usually reasoned at length or
#: repeated itself, so it is asked for the fields only.
CUT_OFF_NUDGE = (
    "Your previous answer was cut off at the output-token limit before it was complete. Answer again, briefly: "
    "fill in only the fields asked for, with no reasoning and no repetition."
)


def invoke_structured(
    llm: BaseChatModel, schema: type[ResponseT], prompt: list[BaseMessage], **kwargs: Any
) -> ResponseT:
    """``llm``'s answer to ``prompt`` as a ``schema``, asked again briefly if it hit the completion-token limit.

    A reasoning model can spend its whole output budget before it writes the object (``nemotron-3-ultra``
    produced 16,384 tokens on a benign-args call), and the OpenAI client then raises
    :class:`openai.LengthFinishReasonError` instead of returning a partial object. Without a retry the node
    loses that step. ``kwargs`` go to ``with_structured_output`` (e.g. ``method``).

    Raises:
        openai.LengthFinishReasonError: The answer was still cut off after :data:`LENGTH_RETRIES` retries.
    """
    for attempt in range(LENGTH_RETRIES + 1):
        try:
            response = llm.with_structured_output(schema, **kwargs).invoke(prompt)
        except openai.LengthFinishReasonError:
            if attempt == LENGTH_RETRIES:
                raise
            logger.warning(
                "traces guardrails defender: the %s answer hit the completion-token limit; asking again for a short one",
                schema.__name__,
            )
            prompt = [*prompt, HumanMessage(content=CUT_OFF_NUDGE)]
            continue
        return response if isinstance(response, schema) else schema.model_validate(response)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises
