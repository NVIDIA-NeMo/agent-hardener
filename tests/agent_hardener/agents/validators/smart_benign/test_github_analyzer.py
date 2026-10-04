# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-level tests for the github_analyzer subgraph (clone_and_scan + summarize).

Git/network is never hit: the pure scanners run against a real ``tmp_path``, the clone is stubbed at
the ``_git_clone``/``_head_sha`` helpers, and the summarizer LLM is the canonical ``_FakeModel`` /
``build_chat_model`` monkeypatch.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign.models import ToolSpec
from agent_hardener.agents.validators.smart_benign.subgraphs.github_analyzer.nodes import clone_and_scan as cs
from agent_hardener.agents.validators.smart_benign.subgraphs.github_analyzer.nodes import summarize as sm
from agent_hardener.agents.validators.smart_benign.subgraphs.github_analyzer.schemas import (
    SummarizerOutput,
    SummarizerPersona,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.github_analyzer.state import GitHubAnalyzerState
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from pathlib import Path


def _state(**overrides: Any) -> GitHubAnalyzerState:
    base = {
        "repo_url": "https://github.com/org/repo",
        "target_name": "victim",
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return GitHubAnalyzerState(**{**base, **overrides})


# --- clone_and_scan ---------------------------------------------------------------------------------


def test_clone_and_scan_skips_without_repo_url() -> None:
    delta = asyncio.run(cs.clone_and_scan(_state(repo_url="")))
    assert "skipped" in delta["source_note"]
    assert "errors" not in delta  # a skip is not a failure


def test_clone_and_scan_reports_clone_failure_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_state: Any, _target: Any) -> None:
        raise RuntimeError("git clone exit=128: not found")

    monkeypatch.setattr(cs, "_git_clone", _boom)  # no git/network runs
    delta = asyncio.run(cs.clone_and_scan(_state()))

    assert delta["source_note"].startswith("github_analyzer failed")
    assert delta["errors"]
    assert "exit=128" in delta["errors"][0]


def test_clone_and_scan_success_returns_scan_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_git_clone", lambda _state, _target: None)
    monkeypatch.setattr(cs, "_head_sha", lambda _repo: "deadbeef")
    monkeypatch.setattr(cs, "_scan_declared_tools", lambda _repo: ([_tool("bash_executor")], 2))
    monkeypatch.setattr(cs, "_read_readme", lambda _repo: "# Repo\nA research agent.")

    delta = asyncio.run(cs.clone_and_scan(_state()))

    assert delta["head_sha"] == "deadbeef"
    assert delta["yaml_files_scanned"] == 2
    assert [t.name for t in delta["declared_tools"]] == ["bash_executor"]
    assert "research agent" in delta["readme_text"]
    assert "tools" not in delta  # clone_and_scan populates declared_tools; summarize copies them to tools


def test_scan_finds_tools_declared_for_a_fabric_agent(tmp_path: Path) -> None:
    """Static discovery has to work before the victim is ever invoked, so the benign suite can be built."""
    (tmp_path / "agent.yaml").write_text(
        "tools:\n  enabled: [read_file]\n  definitions:\n    transfer_funds:\n      kind: http\n",
        encoding="utf-8",
    )
    tools, scanned = cs._scan_declared_tools(tmp_path)

    assert scanned == 1
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"read_file", "transfer_funds"}
    assert by_name["read_file"].source == "github"


def test_scan_finds_tools_exposed_through_an_mcp_server(tmp_path: Path) -> None:
    """Custom tool code reaches a config-only agent as an MCP server, so that is where it declares them."""
    (tmp_path / "agent.yaml").write_text(
        "mcp:\n  servers:\n    finance:\n      allowed_tools: [transfer_funds, check_balance]\n",
        encoding="utf-8",
    )
    tools, _ = cs._scan_declared_tools(tmp_path)
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"transfer_funds", "check_balance"}
    assert "finance MCP server" in by_name["transfer_funds"].description


def test_scan_skips_symlinked_yaml_in_an_untrusted_clone(tmp_path: Path) -> None:
    """A symlink's size is the target's, so following one lets a hostile repo pick what gets read."""
    outside = tmp_path / "outside.yaml"
    outside.write_text("tools:\n  enabled: [exfiltrated]\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tools.yaml").symlink_to(outside)
    (repo / "real.yaml").write_text("tools:\n  enabled: [legit]\n", encoding="utf-8")

    tools, scanned = cs._scan_declared_tools(repo)

    assert scanned == 1
    assert [t.name for t in tools] == ["legit"]


def test_scan_ignores_a_config_that_declares_no_tools(tmp_path: Path) -> None:
    (tmp_path / "other.yaml").write_text("unrelated: true\n", encoding="utf-8")
    tools, scanned = cs._scan_declared_tools(tmp_path)
    assert tools == []
    assert scanned == 1


def test_read_readme_prefers_markdown_and_truncates(tmp_path: Path) -> None:
    assert cs._read_readme(tmp_path) is None  # no README present
    (tmp_path / "README.md").write_text("hello readme", encoding="utf-8")
    assert cs._read_readme(tmp_path) == "hello readme"


def test_clone_env_keeps_token_out_of_url_and_argv() -> None:
    env = cs._clone_env("tok")
    # Token is carried in an auth header via env config, never embedded in a URL.
    assert env["GIT_CONFIG_KEY_0"] == "http.extraHeader"
    assert "tok" not in env["GIT_CONFIG_KEY_0"]
    expected = base64.b64encode(b"oauth2:tok").decode()
    assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {expected}"
    # No token → no injected git config.
    assert "GIT_CONFIG_COUNT" not in cs._clone_env(None)


# --- summarize --------------------------------------------------------------------------------------


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description="d", source="github", confidence=0.9)


class _FakeModel:
    def __init__(self, output: SummarizerOutput | None = None, raises: Exception | None = None) -> None:
        self._output = output
        self._raises = raises

    def with_structured_output(self, _schema: object) -> _FakeModel:
        return self

    async def ainvoke(self, _messages: list[object]) -> SummarizerOutput:
        if self._raises is not None:
            raise self._raises
        assert self._output is not None
        return self._output


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(sm, "build_chat_model", lambda **_kw: model)


def test_summarize_no_readme_skips_llm() -> None:
    # readme_text=None → early return, no model call (no patch needed).
    delta = asyncio.run(sm.summarize(_state(readme_text=None, declared_tools=[_tool("bash_executor")])))
    assert [t.name for t in delta["tools"]] == ["bash_executor"]
    assert "personas" not in delta  # no LLM enrichment happened


def test_summarize_success_enriches_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(
        monkeypatch,
        _FakeModel(
            SummarizerOutput(
                system_role="research assistant",
                personas=[SummarizerPersona(name="dev", description="a developer")],
                out_of_scope=["deleting files"],
            )
        ),
    )
    delta = asyncio.run(sm.summarize(_state(readme_text="# Repo", declared_tools=[_tool("bash_executor")])))

    assert delta["system_role"] == "research assistant"
    assert [p.name for p in delta["personas"]] == ["dev"]
    assert delta["out_of_scope"] == ["deleting files"]
    assert [t.name for t in delta["tools"]] == ["bash_executor"]


def test_summarize_rate_limit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(monkeypatch, _FakeModel(raises=Exception("429 too many requests")))
    with pytest.raises(RateLimitError):
        asyncio.run(sm.summarize(_state(readme_text="# Repo", declared_tools=[_tool("bash_executor")])))


def test_summarize_swallows_non_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(monkeypatch, _FakeModel(raises=ValueError("bad json")))
    delta = asyncio.run(sm.summarize(_state(readme_text="# Repo", declared_tools=[_tool("bash_executor")])))
    assert delta["errors"]  # error recorded
    assert [t.name for t in delta["tools"]] == ["bash_executor"]  # tools still passed through
