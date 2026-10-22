# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes.feedback_analyzer import feedback_analyzer_node

from .fakes import graph_state


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_retries_within_budget_only_count_up(retries: int) -> None:
    update = feedback_analyzer_node(graph_state(retry_counter=retries))
    assert update == {"retry_counter": retries + 1}


def test_fourth_failure_forces_the_loop_to_stop() -> None:
    update = feedback_analyzer_node(graph_state(retry_counter=3))
    assert update == {
        "eval_passed": True,
        "feedback_log": ["Forced break after 3 retries."],
        "retry_counter": 4,
    }
