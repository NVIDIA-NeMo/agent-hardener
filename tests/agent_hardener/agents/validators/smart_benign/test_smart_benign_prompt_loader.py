# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prompt loader: frontmatter parsing, input-contract enforcement, and Jinja rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, UndefinedError, select_autoescape
from jinja2 import meta as jinja_meta

from agent_hardener.agents.validators.smart_benign import prompt_loader as pl
from agent_hardener.agents.validators.smart_benign.prompt_loader import (
    PromptFormatError,
    load_prompt,
    render_prompt,
)

SUBGRAPHS = Path(pl.__file__).parent / "subgraphs"


def _write(tmp_path: Path, text: str, name: str = "p.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _prompt(inputs: list[str], body: str = "Hello {{ who }}") -> str:
    declared = "\n".join(f"  - {i}" for i in inputs)
    return (
        f"---\nname: t.user\ninputs:\n{declared}\n---\n\n{body}\n"
        if inputs
        else f"---\nname: t.sys\ninputs: []\n---\n{body}\n"
    )


# --- split_frontmatter -------------------------------------------------------------------------------------


def test_a_file_without_frontmatter_is_all_body() -> None:
    assert pl._split_frontmatter("just text\n") == ({}, "just text\n")


def test_frontmatter_is_parsed_and_leading_newlines_stripped_from_the_body() -> None:
    meta, body = pl._split_frontmatter("---\nname: a.b\nversion: 2\n---\n\n\nBody\n")

    assert meta == {"name": "a.b", "version": 2}
    assert body == "Body\n"


def test_an_empty_frontmatter_block_yields_empty_meta() -> None:
    assert pl._split_frontmatter("---\n---\nBody") == ({}, "Body")


def test_a_horizontal_rule_in_the_body_does_not_end_the_frontmatter_early() -> None:
    meta, body = pl._split_frontmatter("---\nname: a\n---\nTop\n---\nBottom\n")

    assert meta == {"name": "a"}
    assert body == "Top\n---\nBottom\n"


def test_an_unclosed_frontmatter_is_rejected() -> None:
    with pytest.raises(PromptFormatError, match="no closing"):
        pl._split_frontmatter("---\nname: a\nBody without a closer")


@pytest.mark.parametrize("frontmatter", ["- a\n- b", "just a string"], ids=["list", "scalar"])
def test_non_mapping_frontmatter_is_rejected(frontmatter: str) -> None:
    with pytest.raises(PromptFormatError, match="must be a mapping"):
        pl._split_frontmatter(f"---\n{frontmatter}\n---\nBody")


# --- load_prompt -------------------------------------------------------------------------------------------


def test_load_prompt_returns_the_unrendered_body(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt([], "You are {{ literal }}."))

    assert load_prompt(path) == "You are {{ literal }}.\n"


def test_load_prompt_refuses_a_templated_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt(["who"]))

    with pytest.raises(PromptFormatError, match="Use render_prompt"):
        load_prompt(path)


def test_load_prompt_accepts_a_file_with_no_frontmatter(tmp_path: Path) -> None:
    assert load_prompt(_write(tmp_path, "plain")) == "plain"


# --- render_prompt -----------------------------------------------------------------------------------------


def test_render_substitutes_variables_and_runs_control_flow(tmp_path: Path) -> None:
    body = "{% for t in tools %}- {{ t }}\n{% endfor %}{% if note %}Note: {{ note }}{% endif %}"
    path = _write(tmp_path, _prompt(["tools", "note"], body))

    assert render_prompt(path, {"tools": ["a", "b"], "note": "hi"}) == "- a\n- b\nNote: hi"
    assert render_prompt(path, {"tools": [], "note": ""}) == ""


def test_render_does_not_html_escape(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt(["who"], "{{ who }}"))

    assert render_prompt(path, {"who": "<b>&"}) == "<b>&"


def test_missing_inputs_are_named_in_the_error(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt(["a", "b"], "{{ a }}{{ b }}"))

    with pytest.raises(PromptFormatError, match=r"missing \['b'\]"):
        render_prompt(path, {"a": 1})


def test_undeclared_extra_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt(["a"], "{{ a }}"))

    with pytest.raises(PromptFormatError, match=r"undeclared keys \['zzz'\]"):
        render_prompt(path, {"a": 1, "zzz": 2})


def test_a_body_placeholder_that_was_never_declared_fails_loudly(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt(["a"], "{{ a }} {{ typo }}"))

    with pytest.raises(UndefinedError):
        render_prompt(path, {"a": 1})


def test_a_prompt_without_frontmatter_renders_with_no_variables(tmp_path: Path) -> None:
    assert render_prompt(_write(tmp_path, "static"), {}) == "static"
    with pytest.raises(PromptFormatError, match="undeclared"):
        render_prompt(_write(tmp_path, "static", "q.md"), {"x": 1})


def test_files_are_cached_by_path_until_cleared(tmp_path: Path) -> None:
    path = _write(tmp_path, _prompt([], "v1"))
    assert load_prompt(path) == "v1\n"

    path.write_text(_prompt([], "v2"), encoding="utf-8")
    assert load_prompt(path) == "v1\n"  # served from the cache

    pl._read.cache_clear()
    assert load_prompt(path) == "v2\n"


# --- the shipped prompts -----------------------------------------------------------------------------------


def _shipped_prompts() -> list[Path]:
    return sorted(SUBGRAPHS.glob("*/prompts/*.md"))


def test_shipped_prompts_exist() -> None:
    assert len(_shipped_prompts()) >= 16


@pytest.mark.parametrize("path", _shipped_prompts(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_shipped_prompt_declares_exactly_the_variables_its_body_uses(path: Path) -> None:
    frontmatter, body = pl._read(path)
    declared = set(frontmatter.get("inputs") or [])
    used = jinja_meta.find_undeclared_variables(Environment(autoescape=select_autoescape()).parse(body))

    assert "name" in frontmatter
    assert used <= declared, f"{path.name} uses undeclared variables {sorted(used - declared)}"
    assert declared <= used, f"{path.name} declares unused inputs {sorted(declared - used)}"
    if not declared:
        assert load_prompt(path).strip()
