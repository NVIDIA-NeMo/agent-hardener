# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``init`` run again on an existing manifest.

It updates in place and keeps the user's settings; a failure leaves the file exactly as it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from agent_hardener.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def _inference_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='proj'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (tmp_path / "agent.env").write_text("INFERENCE_API_KEY=x\n", encoding="utf-8")
    return root


def _init(root: Path, output: Path, *extra: str, input: str | None = None) -> object:  # noqa: A002
    args = ["init", "--project-dir", str(root), "--output", str(output), *extra]
    return runner.invoke(app, args if input is not None else [*args, "--yes"], input=input)


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A project and a manifest from a first init, hand-edited the way users do."""
    monkeypatch.chdir(tmp_path)
    root, output = _project(tmp_path), tmp_path / "agent-hardener.yaml"
    first = _init(
        root,
        output,
        *("--name", "proj", "--secrets-file", "agent.env", "--egress", "kept.example.com"),
        *("--binary", "/app/**", "--start-command", "/app/serve"),
    )
    assert first.exit_code == 0, first.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    data["agent"].update(harness="langgraph", relay_integration_confirmed=True, env={"MODE": "demo"})
    data["defenders"] = {"timeout_s": 900}
    data["overrides"] = {"rounds": 3}
    output.write_text("# my notes\n" + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (tmp_path / "garak-scan.yaml").write_text("edited: true\n", encoding="utf-8")
    return root, output


def test_init_again_updates_in_place_and_keeps_what_the_user_set(saved: tuple[Path, Path], tmp_path: Path) -> None:
    root, output = saved

    result = _init(root, output, "--port", "9001", "--start-command", "/app/serve-v2")

    assert result.exit_code == 0, result.output
    assert f"Updated {output}" in result.output
    assert output.read_text(encoding="utf-8").startswith("# my notes\n")  # the user's comments survive
    assert "port           8000 -> 9001" in result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    agent = data["agent"]
    assert (agent["port"], agent["start_command"]) == (9001, "/app/serve-v2")
    assert (agent["harness"], agent["relay_integration_confirmed"], agent["env"]) == (
        "langgraph",
        True,
        {"MODE": "demo"},
    )
    assert agent["egress"] == ["kept.example.com"]
    assert (data["defenders"], data["overrides"]) == ({"timeout_s": 900}, {"rounds": 3})
    assert (tmp_path / "garak-scan.yaml").read_text(encoding="utf-8") == "edited: true\n"


def test_init_again_with_nothing_new_leaves_the_file_untouched(saved: tuple[Path, Path]) -> None:
    root, output = saved
    before = output.read_bytes()

    result = _init(root, output)

    assert result.exit_code == 0, result.output
    assert "is up to date" in result.output
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("remove_dockerfile", "extra", "reason"),
    [
        (True, (), "the Dockerfile 'Dockerfile' is not in"),
        (False, ("--backend", "nonsense"), "--backend must be NAME:PORT"),
        (False, ("--port", "0"), "agent.port"),
    ],
)
def test_a_failed_update_says_why_and_keeps_the_previous_manifest(
    saved: tuple[Path, Path], remove_dockerfile: bool, extra: tuple[str, ...], reason: str
) -> None:
    root, output = saved
    before = output.read_bytes()
    if remove_dockerfile:
        (root / "Dockerfile").unlink()

    result = _init(root, output, *extra)

    assert result.exit_code == 1
    assert f"Could not update {output}" in result.output
    assert reason in result.output
    assert f"{output} was not changed." in result.output
    assert output.read_bytes() == before


def test_a_missing_project_dir_keeps_the_previous_manifest(saved: tuple[Path, Path], tmp_path: Path) -> None:
    _, output = saved
    before = output.read_bytes()

    result = runner.invoke(app, ["init", "--project-dir", str(tmp_path / "gone"), "--output", str(output), "--yes"])

    assert result.exit_code == 1
    assert "does not exist; pass --project-dir" in result.output
    assert output.read_bytes() == before


def test_an_unreadable_manifest_is_not_overwritten(saved: tuple[Path, Path]) -> None:
    root, output = saved
    output.write_text("agent: [\n", encoding="utf-8")

    result = _init(root, output)

    assert result.exit_code == 1
    assert "exists but cannot be read" in result.output
    assert output.read_text(encoding="utf-8") == "agent: [\n"


def test_declining_the_update_keeps_the_manifest(saved: tuple[Path, Path]) -> None:
    root, output = saved
    before = output.read_bytes()

    result = _init(root, output, input="n\n")

    assert result.exit_code == 1
    assert f"{output} was not changed" in result.output
    assert output.read_bytes() == before


def test_an_interactive_update_offers_hosts_the_project_now_calls(saved: tuple[Path, Path]) -> None:
    root, output = saved
    (root / "config.yaml").write_text("base_url: https://api.tavily.com/search\n", encoding="utf-8")

    result = _init(root, output, input="y\ny\n")  # update it; allow the new host

    assert result.exit_code == 0, result.output
    assert "egress         +api.tavily.com" in result.output
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]["egress"] == [
        "kept.example.com",
        "api.tavily.com",
    ]


def test_force_still_starts_from_scratch(saved: tuple[Path, Path]) -> None:
    root, output = saved

    result = _init(root, output, "--force", "--name", "proj", "--binary", "/app/**", "--start-command", "/app/serve")

    assert result.exit_code == 0, result.output
    assert "harness" not in yaml.safe_load(output.read_text(encoding="utf-8"))["agent"]


def test_flags_given_again_replace_secrets_and_backends(saved: tuple[Path, Path]) -> None:
    root, output = saved
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    del data["agent"]["egress"]
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = _init(root, output, "--secrets", "OTHER_KEY", "--backend", "db:5432")

    assert result.exit_code == 0, result.output
    assert "backends       (none) -> db" in result.output
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["agent"]["secrets"] == ["OTHER_KEY"]
    assert "egress" not in data["agent"]
    assert data["backends"][0]["name"] == "db"
