# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The defender's prompt files, rendered with the shared frontmatter + Jinja loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt

PROMPTS_DIR = Path(__file__).parent / "prompts"


def messages(name: str, variables: dict[str, Any]) -> list[BaseMessage]:
    """The ``<name>.system.md`` and rendered ``<name>.user.md`` pair as chat messages."""
    return [
        SystemMessage(content=load_prompt(PROMPTS_DIR / f"{name}.system.md")),
        HumanMessage(content=render_prompt(PROMPTS_DIR / f"{name}.user.md", variables)),
    ]


def as_text(value: Any) -> str:
    """Tool arguments as compact JSON, the same shape the guard model is shown."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)
