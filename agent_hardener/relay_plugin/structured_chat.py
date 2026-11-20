# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ``ChatOpenAI`` whose structured output survives a change of model.

``ChatOpenAI.with_structured_output(Model)`` sends the schema through the OpenAI SDK's ``.parse()``,
which validates the reply text as exactly one JSON object. NVIDIA's endpoint and LiteLLM pass the
schema on, but only some models honour it, so a model swap turns a correct answer wrapped in a
``<think>`` block or a markdown fence into a ``ValidationError`` and loses the reply.

:class:`RobustChatOpenAI` sends the same schema as a plain dict, keeps the raw reply, and reads it
with :func:`~.json_reply.parse_json_reply`. A reply nothing reads is asked for once more with the
error and the schema spelled out, and an endpoint that rejects the schema is asked without it.
A reply cut off at ``max_tokens`` is never read, since its complete part is still an incomplete
answer: it is asked for again, then raised.
Every other ``with_structured_output`` call (dict schemas, ``function_calling``, ``include_raw``)
is ChatOpenAI's own.

Imported lazily: it needs LangChain, which a harness staging the guardrail plugin may lack.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .json_reply import JsonReplyError, parse_json_reply, reply_text

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput

logger = logging.getLogger("agent_hardener.structured_output")

T = TypeVar("T", bound=BaseModel)

#: Default fresh attempts after a reply nothing could read. One, because a second rarely differs and
#: each costs a full generation.
RETRIES = 1

_SCHEMA_REJECTED = ("response_format", "json_schema", "guided", "structured output")


class RobustChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` reading pydantic structured output tolerantly; see the module docstring.

    ``structured_retries`` is how many times an unreadable reply is asked for again. A caller with its
    own fallback, or one in a latency-critical path, sets it to 0.
    """

    structured_retries: int = Field(default=RETRIES, ge=0)

    def with_structured_output(self, schema: Any = None, **kwargs: Any) -> Runnable[LanguageModelInput, Any]:
        """Tolerant structured output for a pydantic ``schema`` in ``json_schema`` mode; otherwise ChatOpenAI's."""
        pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        plain = kwargs.get("method", "json_schema") == "json_schema" and set(kwargs) <= {"method", "strict"}
        if not (pydantic_schema and plain):
            return super().with_structured_output(schema, **kwargs)
        return structured_runnable(self, schema, retries=self.structured_retries)


def structured_runnable(
    model: ChatOpenAI, schema: type[T], *, retries: int = RETRIES
) -> Runnable[LanguageModelInput, T]:
    """A runnable that asks ``model`` for ``schema`` and reads the reply with :func:`parse_json_reply`.

    An unreadable reply is asked for again up to ``retries`` times before :class:`JsonReplyError` is raised.
    """

    def run(prompt: LanguageModelInput, config: RunnableConfig) -> T:
        request = _messages(model, prompt)
        for attempt in range(retries + 1):
            message = _ask(model, schema, request, config)
            try:
                return _read(message, schema)
            except JsonReplyError as exc:
                request = _retry(model, request, message, schema, exc, retries_left=retries - attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    async def arun(prompt: LanguageModelInput, config: RunnableConfig) -> T:
        request = _messages(model, prompt)
        for attempt in range(retries + 1):
            message = await _aask(model, schema, request, config)
            try:
                return _read(message, schema)
            except JsonReplyError as exc:
                request = _retry(model, request, message, schema, exc, retries_left=retries - attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    return RunnableLambda(run, afunc=arun, name=f"structured_{schema.__name__}")


def _messages(model: ChatOpenAI, prompt: LanguageModelInput) -> list[BaseMessage]:
    return model._convert_input(prompt).to_messages()


def _ask(model: ChatOpenAI, schema: type[BaseModel], messages: list[BaseMessage], config: RunnableConfig) -> Any:
    try:
        return model.bind(response_format=response_format(schema)).invoke(messages, config)
    except openai.LengthFinishReasonError as exc:
        return _cut_off(exc)
    except openai.BadRequestError as exc:
        return model.invoke(_without_schema(model, schema, messages, exc), config)


async def _aask(model: ChatOpenAI, schema: type[BaseModel], messages: list[BaseMessage], config: RunnableConfig) -> Any:
    try:
        return await model.bind(response_format=response_format(schema)).ainvoke(messages, config)
    except openai.LengthFinishReasonError as exc:
        return _cut_off(exc)
    except openai.BadRequestError as exc:
        return await model.ainvoke(_without_schema(model, schema, messages, exc), config)


def response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """The ``response_format`` ChatOpenAI itself sends for ``schema``, as a dict so the SDK does not parse the reply.

    The strict form (every field required, no additional properties) matters beyond OpenAI: an
    endpoint that enforces the schema by guided decoding can loop on a looser one until it times out.
    """
    function = openai.pydantic_function_tool(schema)["function"]
    return {
        "type": "json_schema",
        "json_schema": {"name": function["name"], "schema": function["parameters"], "strict": True},
    }


def _cut_off(exc: openai.LengthFinishReasonError) -> AIMessage:
    """The reply the SDK refused because it hit ``max_tokens``, kept so the retry can quote it."""
    return AIMessage(
        content=exc.completion.choices[0].message.content or "", response_metadata={"finish_reason": "length"}
    )


def _without_schema(
    model: ChatOpenAI, schema: type[BaseModel], messages: list[BaseMessage], exc: openai.BadRequestError
) -> list[BaseMessage]:
    """The request again with the schema in words, for an endpoint that refused it as ``response_format``."""
    if not any(marker in str(exc).lower() for marker in _SCHEMA_REJECTED):
        raise exc
    logger.warning("%s rejected the JSON schema (%s); asking without it", model.model_name, exc)
    return [*messages, HumanMessage(content=_instruction(schema))]


def _read(message: Any, schema: type[T]) -> T:
    text = reply_text(message)
    if _cut_off_at_max_tokens(message):
        raise JsonReplyError(schema, "cut off at max_tokens", text)
    return parse_json_reply(text, schema)


def _cut_off_at_max_tokens(message: Any) -> bool:
    return (getattr(message, "response_metadata", None) or {}).get("finish_reason") == "length"


def _retry(
    model: ChatOpenAI,
    request: list[BaseMessage],
    message: Any,
    schema: type[BaseModel],
    exc: JsonReplyError,
    *,
    retries_left: int,
) -> list[BaseMessage]:
    """The conversation for the next attempt, or raise ``exc`` when the attempts are spent."""
    if retries_left <= 0:
        logger.warning("%s gave no readable %s: %s", model.model_name, schema.__name__, exc)
        raise exc
    logger.info("%s gave no readable %s (%s); asking again", model.model_name, schema.__name__, exc.reason)
    feedback = f"Your reply could not be read as the required JSON: {exc.reason}."
    if _cut_off_at_max_tokens(message):
        feedback += " Keep every field brief so the whole object fits."
    reply = message if isinstance(message, AIMessage) else AIMessage(content=str(getattr(message, "content", "")))
    return [*request, reply, HumanMessage(content=f"{feedback}\n\n{_instruction(schema)}")]


def _instruction(schema: type[BaseModel]) -> str:
    return (
        "Reply with only one JSON object that validates against this JSON schema: no prose, no markdown "
        f"fences, no comments.\n{json.dumps(schema.model_json_schema())}"
    )
