# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a pydantic object out of an LLM reply that is not bare JSON.

A JSON-schema ``response_format`` is a request, not a guarantee: an OpenAI-compatible endpoint
honours it only when the model behind it does. Some models (``nemotron-nano-9b-v2``) write their
reasoning as prose before the JSON, or wrap it in a ``<think>`` block or a markdown fence. The
answer is there, but a strict ``model_validate_json`` of the whole reply rejects it.

:func:`parse_json_reply` drops the reasoning, then tries every ``{`` or ``[`` as the start of a JSON
value and returns the first value that validates. Anything else (Python literals, comments, a reply
cut off at ``max_tokens``) does not decode and is left to the caller's retry.

Standard library and pydantic only: this module is staged into the victim with the guardrail plugin.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_REASONING_TAGS = r"think|thinking|reasoning"
_REASONING_BLOCK = re.compile(rf"<({_REASONING_TAGS})\b[^>]*>.*?</\1\s*>", re.S | re.I)
_REASONING_CLOSE = re.compile(rf"</(?:{_REASONING_TAGS})\s*>", re.I)
# gpt-oss "harmony" output leaking through an endpoint that does not split the channels.
_HARMONY_FINAL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|(?:end|return|call)\|>|$)", re.S)
_VALUE_START = re.compile(r"[\[{]")
# strict=False: a raw newline or tab inside a string is the commonest near-miss.
_DECODER = json.JSONDecoder(strict=False)
_EXCERPT = 300


class JsonReplyError(ValueError):
    """No JSON value in the reply validates against the schema."""

    def __init__(self, schema: type[BaseModel], reason: str, reply: str) -> None:
        excerpt = reply if len(reply) <= _EXCERPT else reply[:_EXCERPT] + f"... [{len(reply)} chars]"
        super().__init__(f"could not read a {schema.__name__} from the model reply ({reason}): {excerpt!r}")
        self.reason = reason
        self.reply = reply


def response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """The OpenAI ``response_format`` asking for ``schema``, sent as a dict so no client parses it strictly."""
    return {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}


def reply_text(message: Any) -> str:
    """The visible text of a chat message, else its reasoning channel, where some endpoints put the answer.

    Duck-typed on LangChain's ``AIMessage`` (string or content-block ``content``) so this module does
    not import LangChain.
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
    return content if content.strip() or not isinstance(reasoning, str) else reasoning


def strip_reasoning(text: str) -> str:
    """``text`` without ``<think>``-style blocks or gpt-oss analysis channels."""
    if final := _HARMONY_FINAL.search(text):
        text = final.group(1)
    text = _REASONING_BLOCK.sub("", text)
    if closes := list(_REASONING_CLOSE.finditer(text)):  # the template emitted the opening tag itself
        text = text[closes[-1].end() :]
    return text.strip()


def parse_json_reply(text: str, schema: type[T]) -> T:
    """The first JSON value in ``text``, after its reasoning, that validates as ``schema``.

    Prose, fences and draft JSON around the answer don't matter, since every ``{`` or ``[`` is tried
    as the start of a value. A list is also tried element by element, for ``[{...}]``.

    Raises:
        JsonReplyError: when no value validates; carries the reply and the last validation error.
    """
    body = strip_reasoning(text)
    error: ValidationError | None = None
    for start in _VALUE_START.finditer(body):
        try:
            value, _ = _DECODER.raw_decode(body, start.start())
        except ValueError:
            continue
        for candidate in (value, *(value if isinstance(value, list) else ())):
            try:
                return schema.model_validate(candidate)
            except ValidationError as exc:
                error = exc
    if not body:
        reason = "empty reply"
    elif error is None:
        reason = "no JSON found"
    else:
        first = error.errors()[0]
        reason = f"{'.'.join(map(str, first['loc'])) or '<root>'}: {first['msg']}"
    raise JsonReplyError(schema, reason, text)
