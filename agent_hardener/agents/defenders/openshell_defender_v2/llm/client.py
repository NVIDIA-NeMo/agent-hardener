# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single LLM call boundary for openshell_defender_v2.

Every model call in the package — attack-tuple extraction fallback, harm-certificate reasoning,
and benign-tuple structural extraction — goes through ``complete_structured``/``complete_batch``.
Stubbing these two functions in tests makes the entire deterministic pipeline (policy/, analysis/,
nodes/ formatters) exercisable end to end without a real inference call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agent_hardener.llm import build_chat_model

if TYPE_CHECKING:
    from ..config import DefenderConfig

T = TypeVar("T", bound=BaseModel)


def _client(config: DefenderConfig):
    kwargs = {"temperature": 0.0}
    if config.model:
        kwargs["model"] = config.model
    return build_chat_model(**kwargs)


def complete_structured(prompt: str, schema: type[T], config: DefenderConfig) -> T:
    """Run ``prompt`` through the shared chat model, forced into ``schema``."""
    llm = _client(config).with_structured_output(schema)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm
    result = chain.invoke({"prompt": prompt})
    if not isinstance(result, schema):
        raise TypeError(f"structured output did not match {schema.__name__}: {result!r}")
    return result


def complete_batch(prompts: list[str], schema: type[T], config: DefenderConfig) -> list[T]:
    """Run each prompt in ``prompts`` through :func:`complete_structured`, in order."""
    return [complete_structured(p, schema, config) for p in prompts]
