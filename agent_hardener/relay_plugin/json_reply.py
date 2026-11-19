# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a pydantic object out of whatever an LLM actually replied.

A JSON-schema ``response_format`` is a request, not a guarantee: an OpenAI-compatible endpoint
honours it only when the model behind it does. Change the model and the same call comes back with
the JSON wrapped in a reasoning block, a markdown fence or a sentence, written with Python literals
or single quotes, cut off at ``max_tokens``, or nested one level deeper than the schema. A strict
``model_validate_json`` rejects all of these although the answer is there.

:func:`parse_json_reply` looks for the answer in four steps, stopping at the first that validates:

1. **Where** — drop reasoning (``<think>``, harmony channels), then try the whole reply, each fenced
   or tagged block, and each balanced ``{...}``/``[...]`` span, largest first.
2. **Decode** — strict JSON, then a repaired copy (comments, trailing or missing commas, single or
   smart quotes, unquoted keys, Python/JS literals, bad escapes, raw newlines, unescaped inner
   quotes, truncation), then a Python literal.
3. **Shape** — the value as is, a double-encoded string, an object under a wrapper key
   (``{"data": ...}``, ``{"name": ..., "arguments": ...}``), the first object of a list, or a bare
   reply standing for a single-field schema.
4. **Fields** — fix what validation names: renamed keys (``systemRole``), a nested value sent as a
   JSON string, a scalar where a list belongs, a Literal in the wrong case, ``"85%"`` for a number,
   a number for a string, and extra keys a strict model forbids.

Standard library and pydantic only: this module is staged into the victim with the guardrail
plugin, where LangChain may be absent.
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger("agent_hardener.json_reply")

T = TypeVar("T", bound=BaseModel)

_REASONING_TAGS = r"think|thinking|thought|thoughts|reasoning|reflection|scratchpad|analysis"
_REASONING_BLOCK = re.compile(rf"<({_REASONING_TAGS})\b[^>]*>.*?</\1\s*>", re.S | re.I)
_REASONING_OPEN = re.compile(rf"<(?:{_REASONING_TAGS})\b[^>]*>", re.I)
_REASONING_CLOSE = re.compile(rf"</(?:{_REASONING_TAGS})\s*>", re.I)
# gpt-oss "harmony" output leaking through an endpoint that does not split the channels.
_HARMONY_FINAL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|(?:end|return|call)\|>|$)", re.S)
_SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")
# An unclosed fence is kept: a reply cut off at max_tokens never writes the closing one.
_FENCE = re.compile(r"(```|~~~)[^\n]*\n(.*?)(?:\n?\1|$)", re.S)
_TAG_BLOCK = re.compile(r"<(json|output|answer|result|response|final)\b[^>]*>(.*?)</\1\s*>", re.S | re.I)
_INVISIBLE = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060"))
_JSON_ARRAY = re.compile(r"\[\s*[\[{\"\]]")
_WORD = re.compile(r"[\w$.\-]+")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}": "{", "]": "["}
_SMART_QUOTES = {"\u201c": "\u201d", "\u201d": "\u201d", "\u2018": "\u2019", "\u2019": "\u2019"}
_WORD_LITERALS = {
    "true": "true",
    "false": "false",
    "null": "null",
    "none": "null",
    "nil": "null",
    "undefined": "null",
    "nan": "null",
    "infinity": "null",
}
# Keys a model wraps its answer in, beside the schema's own name.
_WRAPPER_KEYS = (
    "arguments",
    "parameters",
    "args",
    "input",
    "properties",
    "data",
    "result",
    "output",
    "response",
    "answer",
    "json",
    "object",
    "content",
    "value",
)
_MAX_FIX_ROUNDS = 8
_MAX_TRUNCATION_CUTS = 32
_EXCERPT = 300


class JsonReplyError(ValueError):
    """No reading of the reply validates against the schema."""

    def __init__(self, schema: type[BaseModel], reason: str, reply: str) -> None:
        excerpt = reply if len(reply) <= _EXCERPT else reply[:_EXCERPT] + f"... [{len(reply)} chars]"
        super().__init__(f"could not read a {schema.__name__} from the model reply ({reason}): {excerpt!r}")
        self.schema = schema
        self.reason = reason
        self.reply = reply


def response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """The OpenAI ``response_format`` asking for ``schema``, sent as a dict so no client parses it strictly."""
    return {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}


def reply_parts(message: Any) -> tuple[list[str], list[Any]]:
    """The texts and pre-decoded objects a chat message may hold its answer in, most likely first.

    Duck-typed on LangChain's ``AIMessage`` so this module does not import LangChain: the visible
    content (a string or a list of content blocks), then tool-call arguments, then the reasoning
    channel, where some endpoints put the whole answer when thinking is on.
    """
    content = getattr(message, "content", message)
    blocks = [content] if isinstance(content, str) else list(content or [])
    visible = "".join(_block_text(block, ("text", "output_text"), "text") for block in blocks)
    thought = "".join(_block_text(block, ("reasoning", "thinking"), "reasoning", "thinking") for block in blocks)
    extra = getattr(message, "additional_kwargs", None) or {}
    texts = [
        visible,
        thought,
        *(extra[key] for key in ("reasoning_content", "reasoning") if isinstance(extra.get(key), str)),
    ]
    objects = [call["args"] for call in getattr(message, "tool_calls", None) or [] if call.get("args")]
    return [text for text in texts if text.strip()] or [visible], objects


def _block_text(block: Any, kinds: tuple[str, ...], *keys: str) -> str:
    if isinstance(block, str):
        return block if "text" in kinds else ""
    if isinstance(block, dict) and block.get("type") in kinds:
        return next((str(block[key]) for key in keys if isinstance(block.get(key), str)), "")
    return ""


def strip_reasoning(text: str) -> str:
    """``text`` without invisible characters, reasoning blocks or chat-template tokens."""
    text = text.translate(_INVISIBLE)
    if final := _HARMONY_FINAL.search(text):
        text = final.group(1)
    text = _REASONING_BLOCK.sub("", text)
    if closes := list(_REASONING_CLOSE.finditer(text)):  # an opening tag the template already emitted
        text = text[closes[-1].end() :]
    if opened := _REASONING_OPEN.search(text):  # thinking never closed: the answer, if any, is inside
        text = text[: opened.start()] + text[opened.end() :]
    return _SPECIAL_TOKEN.sub("", text).strip()


def parse_json_reply(
    text: str | Sequence[str], schema: type[T], *, objects: Sequence[Any] = (), allow_bare: bool = True
) -> T:
    """Read ``schema`` out of an LLM reply, however it was wrapped or mangled.

    Args:
        text: The reply, or several texts to try in order (content, then reasoning).
        schema: The pydantic model the answer must validate as.
        objects: Already-decoded candidates, tried first (tool-call arguments).
        allow_bare: Whether a reply that is only a scalar or a list may stand for a schema with a
            single field (``0.9`` for ``{"score": 0.9}``). Only ever applied to the whole reply.

    Returns:
        The validated model.

    Raises:
        JsonReplyError: when no candidate validates; carries the reply and the closest failure.
    """
    texts = [text] if isinstance(text, str) else list(text)
    best: ValidationError | None = None
    for value in objects:
        result, error = _fit(value, schema, whole=False, allow_bare=allow_bare)
        if result is not None:
            return result
        best = _closer(best, error)
    for raw in texts:
        for candidate, whole in _candidates(raw):
            for value, how in _readings(candidate):
                result, error = _fit(value, schema, whole=whole, allow_bare=allow_bare)
                if result is not None:
                    if how != "json":
                        log = logger.warning if how == "truncated" else logger.debug
                        log("read a %s from a %s reply", schema.__name__, how)
                    return result
                best = _closer(best, error)
    reply = next((t for t in texts if t and t.strip()), "")
    if not reply:
        reason = "empty reply"
    elif best is None:
        reason = "no JSON found"
    else:
        reason = f"{best.error_count()} validation error(s); first: {_describe(best)}"
    raise JsonReplyError(schema, reason, reply)


def _closer(best: ValidationError | None, error: ValidationError) -> ValidationError:
    return error if best is None or error.error_count() < best.error_count() else best


def _describe(error: ValidationError) -> str:
    first = error.errors()[0]
    return f"{'.'.join(map(str, first['loc'])) or '<root>'}: {first['msg']}"


# --- 1. where -----------------------------------------------------------------------------------


def _candidates(text: str) -> Iterator[tuple[str, bool]]:
    """Substrings that may hold the answer, each with whether it is the whole reply."""
    seen: set[str] = set()

    def fresh(candidate: str) -> bool:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return False
        seen.add(candidate)
        return True

    for source in dict.fromkeys((strip_reasoning(text), text.translate(_INVISIBLE).strip())):
        if fresh(source):
            yield source.strip(), True
        blocks = [m.group(2) for m in _FENCE.finditer(source)] + [m.group(2) for m in _TAG_BLOCK.finditer(source)]
        for block in blocks:
            if fresh(block):
                yield block.strip(), False
        for span in sorted(_spans(source), key=len, reverse=True):
            if fresh(span):
                yield span.strip(), False


def _spans(text: str) -> list[str]:
    """Every top-level balanced ``{...}`` or JSON-looking ``[...]``, the last one possibly unclosed."""
    spans: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "{" or (char == "[" and _JSON_ARRAY.match(text, i)):
            end = _span_end(text, i)
            spans.append(text[i:end])
            i = end
        else:
            i += 1
    return spans


def _span_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        char = text[i]
        if in_string:
            if char == "\\":
                i += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


# --- 2. decode ----------------------------------------------------------------------------------


def _readings(candidate: str) -> Iterator[tuple[Any, str]]:
    """The candidate as Python values, most faithful first, each with how it was read.

    Strict JSON, else the repaired text and a Python literal; then, for a reply cut off before its
    brackets closed, the same with the unfinished last member dropped, one comma further back each time.
    """
    try:
        yield json.loads(candidate, strict=False), "json"
        return
    except ValueError:
        pass
    repaired, truncated = repair_json(candidate)
    with contextlib.suppress(ValueError):
        yield json.loads(repaired, strict=False), "truncated" if truncated else "repaired"
    with contextlib.suppress(ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        yield ast.literal_eval(candidate), "python-literal"
    cut = len(candidate)
    for _ in range(_MAX_TRUNCATION_CUTS if truncated else 0):
        cut = candidate.rfind(",", 0, cut)
        if cut <= 0:
            return
        try:
            yield json.loads(repair_json(candidate[:cut])[0], strict=False), "truncated"
        except ValueError:
            continue


def _decode(candidate: str) -> tuple[Any, str | None]:
    """The first reading of ``candidate``, or ``(None, None)`` when there is none."""
    return next(_readings(candidate), (None, None))


def repair_json(text: str) -> tuple[str, bool]:  # noqa: PLR0912 - one character scanner reads best flat
    """Rewrite near-JSON as JSON, returning the rewrite and whether brackets had to be closed.

    One pass over the characters, tracking strings and the bracket stack, so a repair never
    touches the inside of a string except to escape it correctly.
    """
    out: list[str] = []
    stack: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char in "\"'" or char in _SMART_QUOTES:
            _comma_if_needed(out, stack)
            i = _read_string(text, i, out)
            continue
        if (char == "/" and text.startswith("//", i)) or char == "#":
            i = _line_end(text, i)
            continue
        if text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = n if close < 0 else close + 2
            continue
        if char in _OPENERS:
            _comma_if_needed(out, stack)
            stack.append(char)
            out.append(char)
        elif char in _CLOSERS:
            if _CLOSERS[char] in stack:
                while stack:
                    opener = stack.pop()
                    _drop_trailing(out, ",")
                    out.append(_OPENERS[opener])
                    if opener == _CLOSERS[char]:
                        break
        elif char == ",":
            if stack and _last(out) not in {",", "[", "{", ":"}:
                out.append(",")
        elif char == ":":
            out.append(":")
        elif char.isspace():
            out.append(char)
        elif (char in "-+." or char.isdigit()) and (match := _NUMBER.match(text, i)):
            _comma_if_needed(out, stack)
            out.append(_number(match.group()))
            i = match.end()
            continue
        elif char.isalpha() or char in "_$":
            match = _WORD.match(text, i)
            assert match is not None
            word = match.group()
            _comma_if_needed(out, stack)
            if _next_significant(text, i + len(word)) == ":":
                out.append(json.dumps(word))  # an unquoted key
            else:
                out.append(_WORD_LITERALS.get(word.lower(), json.dumps(word)))
            i += len(word)
            continue
        i += 1
    truncated = bool(stack)
    _drop_trailing(out, ",")
    if _last(out) == ":":
        out.append("null")
    for opener in reversed(stack):
        _drop_trailing(out, ",")
        out.append(_OPENERS[opener])
    return "".join(out), truncated


def _read_string(text: str, start: int, out: list[str]) -> int:  # noqa: PLR0912 - a character scanner
    """Emit the string opening at ``start`` as a JSON string; return the index after it."""
    opener = text[start]
    closer = _SMART_QUOTES.get(opener, opener)
    closers = {closer, opener} if opener in _SMART_QUOTES else {closer}
    out.append('"')
    i, n = start + 1, len(text)
    while i < n:
        char = text[i]
        if char == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", text[i + 2 : i + 6]):
                out.append(text[i : i + 6])
                i += 6
                continue
            if nxt in '"\\/bfnrt':
                out.append(char + nxt)
            elif nxt == "'":
                out.append("'")
            else:  # \_ \$ \( C:\path — keep the backslash as text
                out.append("\\\\" + nxt)
            i += 2
            continue
        if char in closers and _ends_string(text, i + 1):
            out.append('"')
            return i + 1
        if char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
        i += 1
    out.append('"')  # cut off inside the string
    return n


def _ends_string(text: str, after: int) -> bool:
    """Whether a quote is the end of its string rather than an unescaped quote inside it."""
    j = after
    while j < len(text) and text[j] in " \t\r":
        j += 1
    if j >= len(text) or text[j] in ",:}]\n":
        return True
    return text.startswith("//", j) or text.startswith("/*", j)


def _comma_if_needed(out: list[str], stack: list[str]) -> None:
    """Insert the comma a model left out between two values."""
    if stack and _last(out) not in {"", ",", ":", "[", "{"}:
        out.append(",")


def _last(out: list[str]) -> str:
    return next((piece.strip()[-1] for piece in reversed(out) if piece.strip()), "")


def _drop_trailing(out: list[str], char: str) -> None:
    while out and not out[-1].strip():
        out.pop()
    if out and out[-1] == char:
        out.pop()


def _line_end(text: str, i: int) -> int:
    end = text.find("\n", i)
    return len(text) if end < 0 else end


def _next_significant(text: str, i: int) -> str:
    while i < len(text) and text[i].isspace():
        i += 1
    return text[i] if i < len(text) else ""


def _number(token: str) -> str:
    token = re.sub(r"^(-?)\.", r"\g<1>0.", token.lstrip("+").rstrip("."))
    return re.sub(r"^(-?)0+(\d)", r"\1\2", token)  # leading zeros


# --- 3. shape -----------------------------------------------------------------------------------


def _fit(value: Any, schema: type[T], *, whole: bool, allow_bare: bool) -> tuple[T | None, ValidationError]:
    best: ValidationError | None = None
    for shape in _shapes(value, schema, whole=whole and allow_bare):
        try:
            return _validate(shape, schema), None  # type: ignore[return-value]
        except ValidationError as error:
            best = _closer(best, error)
    return None, best  # type: ignore[return-value]  # _shapes always yields value itself


def _shapes(value: Any, schema: type[BaseModel], *, whole: bool) -> Iterator[Any]:
    """``value`` itself, then the ways an answer is commonly nested or flattened around it.

    Each step goes one level into ``value`` or decodes a string to a container, so it ends.
    """
    yield value
    if isinstance(value, str):
        decoded, how = _decode(value.strip()) if value.strip()[:1] in {"{", "["} else (None, None)
        if how is not None and not isinstance(decoded, str):
            yield from _shapes(decoded, schema, whole=whole)
        return
    fields = schema.model_fields
    if isinstance(value, dict):
        own = {_norm(name) for name in fields}
        for key, inner in value.items():
            wrapper = len(value) == 1 or _norm(key) == _norm(schema.__name__) or str(key).lower() in _WRAPPER_KEYS
            if wrapper and _norm(key) not in own and isinstance(inner, (dict, list, str)):
                yield from _shapes(inner, schema, whole=False)
    elif isinstance(value, list):
        objects = [item for item in value if isinstance(item, dict)]
        if objects:
            yield from _shapes(objects[0], schema, whole=False)
        list_fields = [name for name, field in fields.items() if _is_list(field.annotation)]
        if len(list_fields) == 1 and (whole or len(fields) == 1):
            yield {list_fields[0]: value}
    elif whole and len(fields) == 1:
        yield {next(iter(fields)): value}


def _is_list(annotation: Any) -> bool:
    return getattr(annotation, "__origin__", None) in {list, tuple, set} or annotation in {list, tuple, set}


def _norm(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


# --- 4. fields ----------------------------------------------------------------------------------


def _validate(value: Any, schema: type[T]) -> T:
    """Validate, fixing what each round's errors name, until it passes or nothing more can be fixed."""
    rounds = 0
    while True:
        try:
            return schema.model_validate(value)
        except ValidationError as error:
            rounds += 1
            fixed = _fix(value, error)
            if fixed is None or rounds > _MAX_FIX_ROUNDS:
                raise
            value = fixed


def _fix(value: Any, error: ValidationError) -> Any | None:
    """A copy of ``value`` with every error that can be repaired repaired, or ``None`` if none can."""
    value = json.loads(json.dumps(value, default=str))  # a private copy, so a failed shape stays intact
    changed = False
    for detail in error.errors():
        loc = [part for part in detail["loc"] if isinstance(part, (str, int))]
        if not loc or (parent := _resolve(value, loc[:-1])) is None:
            continue
        key = loc[-1]
        if detail["type"] == "missing":
            changed |= _rename(parent, key)
        elif detail["type"] == "extra_forbidden" and isinstance(parent, dict):
            changed |= parent.pop(key, _ABSENT) is not _ABSENT
        else:
            current = _get(parent, key)
            if current is not _ABSENT:
                replacement = _coerce(current, detail)
                if replacement is not _ABSENT:
                    parent[key] = replacement
                    changed = True
    return value if changed else None


_ABSENT = object()


def _resolve(value: Any, path: list[str | int]) -> Any:
    for part in path:
        value = _get(value, part)
    return value if isinstance(value, (dict, list)) else None


def _get(container: Any, key: str | int) -> Any:
    if isinstance(container, dict):
        return container.get(key, _ABSENT)
    in_list = isinstance(container, list) and isinstance(key, int) and -len(container) <= key < len(container)
    return container[key] if in_list else _ABSENT


def _rename(parent: Any, field: str | int) -> bool:
    """Move a key written as ``systemRole``/``System Role``/``system-role`` to ``field``."""
    wanted = _norm(field)
    for key in list(parent) if isinstance(parent, dict) else []:
        if key != field and _norm(key) == wanted:
            parent[field] = parent.pop(key)
            return True
    return False


def _coerce(current: Any, detail: Any) -> Any:  # noqa: PLR0911, PLR0912 - flat dispatch on the error type
    kind = detail["type"]
    if isinstance(current, str) and kind in {"list_type", "dict_type", "model_type", "model_attributes_type"}:
        decoded, how = _decode(current.strip()) if current.strip()[:1] in {"{", "["} else (None, None)
        if how is not None:
            return decoded
        if kind == "list_type":
            return [current]
    if kind == "list_type":
        if current is None:
            return []
        if isinstance(current, dict) and len(current) == 1 and isinstance(next(iter(current.values())), list):
            return next(iter(current.values()))
        return [current]
    if kind in {"literal_error", "enum"} and isinstance(current, str):
        expected = re.findall(r"'([^']*)'", str(detail.get("ctx", {}).get("expected", "")))
        for option in expected:
            if _norm(option) == _norm(current):
                return option
    if kind in {"float_parsing", "int_parsing", "float_type", "int_type"} and isinstance(current, str):
        match = _NUMBER.search(current.replace(",", ""))
        if match:
            number = float(match.group())
            if "%" in current[match.end() :].lstrip()[:1]:
                number /= 100
            return int(number) if kind.startswith("int") and number.is_integer() else number
    if kind == "string_type" and isinstance(current, (int, float, bool, dict, list)):
        return json.dumps(current)
    return _ABSENT
