# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from agent_hardener.llm import build_chat_model


def test_nemotron_analysis_model_runs_without_reasoning() -> None:
    model = build_chat_model(model="nvidia/nemotron-3.5-lightning-30b-a3b", api_key="k")
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_other_models_get_no_reasoning_toggle() -> None:
    assert build_chat_model(model="openai/gpt-oss-20b", api_key="k").extra_body is None


def test_caller_extra_body_wins() -> None:
    model = build_chat_model(model="nvidia/nemotron-3.5-lightning-30b-a3b", api_key="k", extra_body={"x": 1})
    assert model.extra_body == {"x": 1}
