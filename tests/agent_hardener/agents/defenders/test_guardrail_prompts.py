# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every placeholder in the guardrail-authoring prompts must be supplied where the prompt is used.

A missing key is not caught by importing the module or by formatting the template in isolation with
explicit arguments — only by the real call site, at runtime, inside a defender that then fails the
whole round. Checking the source structurally catches it at test time instead.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

GENERATOR = (
    Path(__file__).resolve().parents[4]
    / "agent_hardener/agents/defenders/guardrails_defender_v2/nodes/custom_guardrails_generator.py"
)


def _module() -> ast.Module:
    return ast.parse(GENERATOR.read_text(encoding="utf-8"))


def _string_constants(tree: ast.Module) -> dict[str, str]:
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _format_calls(tree: ast.Module, templates: dict[str, str]) -> list[tuple[str, set[str]]]:
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in templates
        ):
            calls.append((node.func.value.id, {kw.arg for kw in node.keywords if kw.arg}))
    return calls


def test_every_prompt_placeholder_is_supplied_by_its_call_site() -> None:
    tree = _module()
    templates = _string_constants(tree)
    calls = _format_calls(tree, templates)

    assert calls, "no prompt .format() call sites found — this test is checking nothing"
    for name, supplied in calls:
        required = set(re.findall(r"\{(\w+)\}", templates[name]))
        assert not (required - supplied), f"{name}.format() is missing {sorted(required - supplied)}"
