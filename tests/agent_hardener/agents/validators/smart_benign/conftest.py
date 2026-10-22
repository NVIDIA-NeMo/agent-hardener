# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fake chat model for the smart_benign node tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeStructuredModel:
    """Stands in for ``build_chat_model(...)``: returns a scripted result or raises, and records the call."""

    def __init__(self, result: Any = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.schema: object = None
        self.messages: list[Any] = []

    def with_structured_output(self, schema: object) -> FakeStructuredModel:
        self.schema = schema
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


@pytest.fixture
def fake_model() -> Callable[..., Any]:
    return FakeStructuredModel
