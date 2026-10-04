# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ..schemas import GraphState


def feedback_analyzer_node(state: GraphState) -> dict[str, Any]:
    """Handles turnaround logic when validation fails."""
    # Increment retry counter to prevent infinite loops
    current_retries = state.retry_counter + 1

    if current_retries > 3:
        # Break the loop forcefully by overriding eval_passed
        return {
            "eval_passed": True,
            "feedback_log": ["Forced break after 3 retries."],
            "retry_counter": current_retries,
        }

    return {"retry_counter": current_retries}
