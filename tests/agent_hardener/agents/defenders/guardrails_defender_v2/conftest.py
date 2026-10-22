# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from .fakes import FakeLLM

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    InstallLLM = Callable[[ModuleType, Callable[[str], Any]], FakeLLM]


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> InstallLLM:
    """Replace a node module's ``get_llm`` with a ``FakeLLM`` driven by the given responder."""

    def install(module: ModuleType, respond: Callable[[str], Any]) -> FakeLLM:
        llm = FakeLLM(respond)

        def get_llm(temperature: float = 0.3) -> FakeLLM:
            llm.temperatures.append(temperature)
            return llm

        monkeypatch.setattr(module, "get_llm", get_llm)
        return llm

    return install
