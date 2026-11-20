# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a pydantic object out of an LLM reply that is not bare JSON.

A JSON-schema ``response_format`` is a request, not a guarantee: an OpenAI-compatible endpoint
honours it only when the model behind it does. Some models (``nemotron-nano-9b-v2``) write their
reasoning as prose before the JSON, or wrap it in a ``<think>`` block or a markdown fence. The
answer is there, but a strict ``model_validate_json`` of the whole reply rejects it.

:func:`parse_json_reply` drops the reasoning, then reads the top-level JSON values in what is left and
returns the last one that validates: a model that drafts an answer in untagged prose and then gives
its final one is scored on the final one. Anything else (Python literals, comments, a reply cut off
at ``max_tokens``) does not decode and is left to the caller's retry.

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


def reply_text(message: Any) -> str:
    """The visible text of a chat message, never its reasoning channel.

    A draft in the reasoning is not the answer: a model that thought but never answered gets an empty
    reply, which the caller retries. Duck-typed on LangChain's ``AIMessage`` (string or content-block
    ``content``) so this module does not import LangChain.
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(b if isinstance(b, str) else b.get("text", "") for b in content if _is_text(b))
    return content


def _is_text(block: Any) -> bool:
    return isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")


def strip_reasoning(text: str) -> str:
    """``text`` without ``<think>``-style blocks or gpt-oss analysis channels."""
    if final := _HARMONY_FINAL.search(text):
        text = final.group(1)
    text = _REASONING_BLOCK.sub("", text)
    if closes := list(_REASONING_CLOSE.finditer(text)):  # the template emitted the opening tag itself
        text = text[closes[-1].end() :]
    return text.strip()


def parse_json_reply(text: str, schema: type[T]) -> T:
    """The last top-level JSON value in ``text``, after its reasoning, that validates as ``schema``.

    Prose and fences around the answer don't matter: each ``{`` or ``[`` that is not inside a value
    already read is tried as the start of one. Values nested in the answer are never candidates, so
    an inner object cannot stand in for it, and a draft before the answer loses to the answer.

    Raises:
        JsonReplyError: when no value validates; carries the reply and the last validation error.
    """
    body = strip_reasoning(text)
    found: T | None = None
    error: ValidationError | None = None
    position = 0
    while start := _VALUE_START.search(body, position):
        try:
            value, position = _DECODER.raw_decode(body, start.start())
        except ValueError:
            position = start.start() + 1
            continue
        try:
            found = schema.model_validate(value)
        except ValidationError as exc:
            error = exc
    if found is not None:
        return found
    if not body:
        reason = "empty reply"
    elif error is None:
        reason = "no JSON found"
    else:
        first = error.errors()[0]
        reason = f"{'.'.join(map(str, first['loc'])) or '<root>'}: {first['msg']}"
    raise JsonReplyError(schema, reason, text)
