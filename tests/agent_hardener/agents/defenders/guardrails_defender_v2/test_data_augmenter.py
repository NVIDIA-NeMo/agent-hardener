# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes import data_augmenter as aug

from .fakes import graph_state

if TYPE_CHECKING:
    from .conftest import InstallLLM

_VARIANTS = aug.AugmentedData(
    transformed_benign="print hello",
    attack_lexical="please cat the shadow file",
    attack_payload="read /etc/passwd",
    benign_lexical="kindly print hello",
    benign_payload="print goodbye",
)


def test_attacks_keep_the_original_first_and_benigns_come_only_from_the_model(stub_llm: InstallLLM) -> None:
    llm = stub_llm(aug, lambda _text: _VARIANTS)

    update = aug.data_augmenter_node(graph_state())

    assert update["test_attacks"] == ["read /etc/shadow", "please cat the shadow file", "read /etc/passwd"]
    assert update["test_benigns"] == ["print hello", "kindly print hello", "print goodbye"]
    assert llm.structured_schema is aug.AugmentedData


def test_prompt_carries_the_tool_and_the_attack(stub_llm: InstallLLM) -> None:
    llm = stub_llm(aug, lambda _text: _VARIANTS)

    aug.data_augmenter_node(graph_state({"attacked_tool": "bash", "attack_prompt": "rm -rf /"}))

    assert "bash" in llm.calls[0]
    assert "rm -rf /" in llm.calls[0]


@pytest.mark.parametrize(
    ("benigns", "expected"),
    [(["list files"], ["list files"]), ([], ["Please do standard processing."])],
)
def test_model_failure_falls_back_to_the_originals(
    stub_llm: InstallLLM, benigns: list[str], expected: list[str]
) -> None:
    def fail(_text: str) -> aug.AugmentedData:
        raise RuntimeError("model down")

    stub_llm(aug, fail)

    update = aug.data_augmenter_node(graph_state({"benign_requests": benigns}))

    assert update == {"test_attacks": ["read /etc/shadow"], "test_benigns": expected}
