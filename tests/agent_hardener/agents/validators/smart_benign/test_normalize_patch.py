# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile_synthesizer normalize._apply_tool_patches — new-tool creation."""

from __future__ import annotations

from agent_hardener.agents.validators.smart_benign.models import ToolSpec
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.nodes.normalize import (
    _apply_tool_patches,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.schemas import ToolPatch


def _tool(name: str, *, description: str = "does things", source: str = "nl") -> ToolSpec:
    return ToolSpec(name=name, description=description, source=source, confidence=0.9, example_inputs=["hi"])


# ---------------------------------------------------------------------------
# Existing behaviour (regression)
# ---------------------------------------------------------------------------


def test_no_patches_returns_tools_unchanged() -> None:
    tools = [_tool("bash_executor")]
    assert _apply_tool_patches(tools, []) == tools


def test_patch_fills_empty_description() -> None:
    tool = ToolSpec(name="bash_executor", description="x", source="nl", confidence=0.9, example_inputs=[])
    # description already set — should not be overwritten
    result = _apply_tool_patches([tool], [ToolPatch(name="bash_executor", description="new desc")])
    assert result[0].description == "x"


def test_patch_fills_empty_example_inputs() -> None:
    tool = ToolSpec(name="bash_executor", description="runs bash", source="nl", confidence=0.9, example_inputs=[])
    result = _apply_tool_patches([tool], [ToolPatch(name="bash_executor", example_inputs=["run ls"])])
    assert result[0].example_inputs == ["run ls"]


# ---------------------------------------------------------------------------
# New behaviour: creating tools from interview patches
# ---------------------------------------------------------------------------


def test_new_tool_created_from_patch_with_description() -> None:
    existing = [_tool("bash_executor")]
    patch = ToolPatch(name="file_reader", description="reads files from disk", example_inputs=["read config.yaml"])
    result = _apply_tool_patches(existing, [patch])
    names = {t.name for t in result}
    assert "bash_executor" in names
    assert "file_reader" in names


def test_new_tool_has_interview_source_and_reasonable_confidence() -> None:
    patch = ToolPatch(name="web_search", description="searches the web", example_inputs=[])
    result = _apply_tool_patches([], [patch])
    new_tool = next(t for t in result if t.name == "web_search")
    assert new_tool.source == "interview"
    assert new_tool.confidence >= 0.5


def test_new_tool_not_created_without_description() -> None:
    existing = [_tool("bash_executor")]
    patch = ToolPatch(name="mystery_tool")  # no description
    result = _apply_tool_patches(existing, [patch])
    assert all(t.name != "mystery_tool" for t in result)


def test_existing_tool_not_duplicated_by_patch() -> None:
    existing = [_tool("bash_executor")]
    patch = ToolPatch(name="bash_executor", description="runs bash commands")
    result = _apply_tool_patches(existing, [patch])
    assert sum(1 for t in result if t.name == "bash_executor") == 1


# ---------------------------------------------------------------------------
# Boundary patches: allowed/blocked patterns are unioned onto the tool
# ---------------------------------------------------------------------------


def test_patch_unions_allowed_and_blocked_patterns() -> None:
    tool = ToolSpec(
        name="bash_executor",
        description="runs bash",
        source="nl",
        confidence=0.9,
        example_inputs=["run ls"],
        allowed_patterns=["ls *"],
        blocked_patterns=["rm -rf /"],
    )
    patch = ToolPatch(
        name="bash_executor",
        allowed_patterns=["grep *", "ls *"],  # one new, one duplicate
        blocked_patterns=["curl *"],
    )
    result = _apply_tool_patches([tool], [patch])
    assert result[0].allowed_patterns == ["ls *", "grep *"]  # order-preserving, deduped
    assert result[0].blocked_patterns == ["rm -rf /", "curl *"]


def test_new_tool_from_patch_carries_patterns() -> None:
    patch = ToolPatch(
        name="file_reader",
        description="reads files",
        allowed_patterns=["read *.yaml"],
        blocked_patterns=["read /etc/shadow"],
    )
    result = _apply_tool_patches([], [patch])
    new_tool = next(t for t in result if t.name == "file_reader")
    assert new_tool.allowed_patterns == ["read *.yaml"]
    assert new_tool.blocked_patterns == ["read /etc/shadow"]


# ---------------------------------------------------------------------------
# drop=True (merge approval)
# ---------------------------------------------------------------------------


def test_drop_removes_alias_tool() -> None:
    tools = [_tool("shell_executor"), _tool("bash_executor")]
    patches = [ToolPatch(name="shell_executor", drop=True)]
    result = _apply_tool_patches(tools, patches)
    names = {t.name for t in result}
    assert "shell_executor" not in names
    assert "bash_executor" in names


def test_drop_and_create_canonical_in_one_pass() -> None:
    tools = [_tool("shell_executor")]
    patches = [
        ToolPatch(name="shell_executor", drop=True),
        ToolPatch(name="bash_executor", description="runs bash commands", example_inputs=["ls -la"]),
    ]
    result = _apply_tool_patches(tools, patches)
    names = {t.name for t in result}
    assert "shell_executor" not in names
    assert "bash_executor" in names
    new_tool = next(t for t in result if t.name == "bash_executor")
    assert new_tool.source == "interview"


def test_drop_false_does_not_remove_tool() -> None:
    tools = [_tool("bash_executor")]
    patches = [ToolPatch(name="bash_executor", drop=False)]
    result = _apply_tool_patches(tools, patches)
    assert any(t.name == "bash_executor" for t in result)
