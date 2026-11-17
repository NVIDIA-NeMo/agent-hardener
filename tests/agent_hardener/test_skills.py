# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills: they point at code that exists, carry routing cases, and ``init`` installs the customer one.

A skill that names a moved file or a removed flag fails quietly — the agent follows it anyway — so the
static checks here are what keep the skills honest as the code changes. Whether each ``tests.json`` prompt
really routes to its skill needs a live agent: ``scripts/skill_routing.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import pytest
import typer.main
import yaml
from typer.testing import CliRunner

from agent_hardener.cli import app
from agent_hardener.skills import SKILL_NAME, bundled_skill_dir, install_skill

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
DEV_SKILLS = sorted(p.parent for p in (REPO / ".agents" / "skills").glob("*/SKILL.md"))
ALL_SKILLS = [*DEV_SKILLS, bundled_skill_dir()]

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_SPAN = re.compile(r"`([^`\n]+)`")
_REPO_PATH = re.compile(r"^(?:agent_hardener|tests|examples|scripts|\.agents)/\S*$")
_JUST = re.compile(r"(?<![\w-])just\s+([a-z][\w-]*)")
_CLI = re.compile(r"(?<![\w./-])agent-hardener((?:[ \t]+[^\s|;&#`]+)*)")
_RECIPE = re.compile(r"^(?![^\n]*:=)@?([A-Za-z_][\w-]*)(?:[ \t][^\n]*)?:(?!=)", re.MULTILINE)
_COMMENT = re.compile(r"(?:^|\s)#.*")


def _frontmatter(skill: Path) -> dict:
    match = _FRONTMATTER.match((skill / "SKILL.md").read_text(encoding="utf-8"))
    assert match, f"{skill.name}: SKILL.md must open with YAML frontmatter"
    return yaml.safe_load(match.group(1))


def _markdown(skill: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(skill.rglob("*.md")))


def _code(skill: Path) -> list[str]:
    """Every inline code span and fenced-block line: where paths, recipes and commands are written."""
    text = _markdown(skill)
    # Shell comments in fenced blocks are prose, not commands.
    lines = [_COMMENT.sub("", line) for block in _FENCE.findall(text) for line in block.splitlines()]
    return lines + _SPAN.findall(_FENCE.sub("", text))


def _cli_errors(code: str, root: click.Command) -> list[str]:
    errors = []
    for match in _CLI.finditer(code):
        command, tokens = root, match.group(1).split()
        while tokens and isinstance(command, click.Group) and not tokens[0].startswith("-"):
            name = tokens.pop(0)
            if name not in command.commands:
                errors.append(f"unknown command `agent-hardener ... {name}`")
                break
            command = command.commands[name]
        else:
            known = {opt for param in command.params for opt in param.opts} | {"--help"}
            flags = [token.split("=")[0] for token in tokens if token.startswith("--")]
            errors += [f"unknown flag {flag} for `{command.name}`" for flag in flags if flag not in known]
    return errors


@pytest.mark.parametrize("skill", ALL_SKILLS, ids=lambda p: p.name)
def test_frontmatter_routes_on_name_and_short_description(skill: Path) -> None:
    fields = _frontmatter(skill)
    assert set(fields) == {"name", "description"}
    assert fields["name"] == skill.name
    # The description is all an agent sees when choosing a skill, so it has to stay short enough to read.
    assert 0 < len(fields["description"].split()) <= 50


@pytest.mark.parametrize("skill", ALL_SKILLS, ids=lambda p: p.name)
def test_skill_never_mentions_nat(skill: Path) -> None:
    # NAT victims are deprecated; a skill that teaches them sends users down a dead path.
    assert not re.search(r"\bNAT\b|NeMo Agent Toolkit", _markdown(skill))


@pytest.mark.parametrize("skill", ALL_SKILLS, ids=lambda p: p.name)
def test_named_paths_recipes_and_commands_exist(skill: Path) -> None:
    recipes = set(_RECIPE.findall((REPO / "justfile").read_text(encoding="utf-8")))
    root = typer.main.get_command(app)
    errors = []
    for code in _code(skill):
        paths = [token.rstrip(".,:;)") for token in code.split() if _REPO_PATH.match(token)]
        errors += [f"missing path `{path}`" for path in paths if not (REPO / path).exists()]
        errors += [f"unknown recipe `just {r}`" for r in _JUST.findall(code) if r not in recipes]
        errors += _cli_errors(code, root)
    assert errors == []


def test_customer_skill_names_nothing_only_the_repo_has() -> None:
    # Customers get the package, not the repo: repo paths and just recipes would be dead ends for them.
    code = _code(bundled_skill_dir())
    assert [c for c in code if _JUST.search(c) or any(_REPO_PATH.match(t) for t in c.split())] == []


@pytest.mark.parametrize("skill", ALL_SKILLS, ids=lambda p: p.name)
def test_routing_cases_cover_every_kind(skill: Path) -> None:
    cases = json.loads((skill / "tests.json").read_text(encoding="utf-8"))
    assert cases["skill"] == skill.name
    assert {case["type"] for case in cases["tests"]} >= {"explicit", "implicit", "contextual", "negative-control"}
    for case in cases["tests"]:
        # A positive case names the skill it must load; a negative control names the one it must not.
        assert ("expected_skill" in case) != ("expected_skill_not" in case), case["prompt"]
        assert case.get("expected_skill", case.get("expected_skill_not")) == skill.name, case["prompt"]


def test_install_skill_copies_without_tests_and_keeps_edits(tmp_path: Path) -> None:
    dest, written = install_skill(tmp_path)
    assert written
    assert dest == tmp_path / ".claude" / "skills" / SKILL_NAME
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == (bundled_skill_dir() / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert not (dest / "tests.json").exists()

    (dest / "SKILL.md").write_text("edited", encoding="utf-8")
    assert install_skill(tmp_path) == (dest, False)
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "edited"
    assert install_skill(tmp_path, force=True) == (dest, True)
    assert (dest / "SKILL.md").read_text(encoding="utf-8") != "edited"


@pytest.mark.parametrize("skip", [False, True])
def test_init_installs_the_skill_unless_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip: bool) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = ["init", "--project-dir", str(root), "--name", "proj", "--secrets", "K", "--secrets-file", "agent.env"]
    args += ["--binary", "/app/**", "--start-command", "/app/serve", "--yes", "--force"]
    args += ["--skip-skill"] if skip else []

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude" / "skills" / SKILL_NAME / "SKILL.md").exists() is not skip
