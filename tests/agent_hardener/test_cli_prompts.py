# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer

from agent_hardener import cli_prompts


class Script:
    """Scripted stand-in for ``typer.prompt`` / ``typer.confirm`` that records every call."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, **kwargs: Any) -> Any:
        self.calls.append((text, kwargs))
        return self.answers.pop(0)


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {"info": [], "fail": [], "ok": []}
    for name, sink in seen.items():
        monkeypatch.setattr(cli_prompts, name, sink.append)
    monkeypatch.setattr(cli_prompts, "is_rich", lambda: False)
    return seen


def script_prompt(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> Script:
    script = Script(list(answers))
    monkeypatch.setattr(typer, "prompt", script)
    return script


def script_confirm(monkeypatch: pytest.MonkeyPatch, *answers: bool) -> Script:
    script = Script(list(answers))
    monkeypatch.setattr(typer, "confirm", script)
    return script


class TestPromptWithHelp:
    def test_returns_raw_value_without_parse(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script = script_prompt(monkeypatch, "hello")
        assert cli_prompts.prompt_with_help("Name", "agent_name", default="x") == "hello"
        label, kwargs = script.calls[0]
        assert label == "Name (? for help)"
        assert kwargs == {"default": "x"}

    def test_question_mark_shows_help_then_reasks(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script = script_prompt(monkeypatch, " ? ", "real")
        assert cli_prompts.prompt_with_help("Port", "port") == "real"
        assert len(script.calls) == 2
        assert any("garak attacks" in line for line in messages["info"])

    def test_parse_retries_until_valid(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script_prompt(monkeypatch, "abc", "8000")
        assert cli_prompts.prompt_with_help("Port", "port", parse=int) == 8000
        assert messages["fail"] == ["Please enter a valid value."]

    def test_rich_hint_is_styled(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        monkeypatch.setattr(cli_prompts, "is_rich", lambda: True)
        script = script_prompt(monkeypatch, "v")
        cli_prompts.prompt_with_help("Name", "agent_name")
        assert "\x1b[" in script.calls[0][0]


class TestConfirmWithHelp:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("y", True), ("YES", True), (" n ", False), ("No", False)],
    )
    def test_accepts_yes_no_variants(
        self, monkeypatch: pytest.MonkeyPatch, messages: dict, answer: str, expected: bool
    ) -> None:
        script_prompt(monkeypatch, answer)
        assert cli_prompts.confirm_with_help("ok?", "backends") is expected

    @pytest.mark.parametrize("default", [True, False])
    def test_blank_returns_default_and_offers_it(
        self, monkeypatch: pytest.MonkeyPatch, messages: dict, default: bool
    ) -> None:
        script = script_prompt(monkeypatch, "")
        assert cli_prompts.confirm_with_help("ok?", "backends", default=default) is default
        assert script.calls[0][1] == {"default": "y" if default else "n", "show_default": False}

    def test_help_and_invalid_answers_reask(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script = script_prompt(monkeypatch, "?", "maybe", "y")
        assert cli_prompts.confirm_with_help("ok?", "backends") is True
        assert len(script.calls) == 3
        assert messages["fail"] == ["Please answer y or n (or ? for help)."]
        assert any("host-side service" in line for line in messages["info"])

    def test_rich_hint_is_styled(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        monkeypatch.setattr(cli_prompts, "is_rich", lambda: True)
        script = script_prompt(monkeypatch, "y")
        cli_prompts.confirm_with_help("ok?", "backends")
        assert "\x1b[" in script.calls[0][0]


class TestWriteGarakScaffold:
    def test_writes_yaml_with_port_in_target_uri(self, tmp_path: Path, messages: dict) -> None:
        target = tmp_path / "garak.yaml"
        assert cli_prompts.write_garak_scaffold(9123, path=target) == target
        assert "http://127.0.0.1:9123/v1/chat/completions" in target.read_text()
        assert messages["ok"] == [f"Wrote {target}"]

    def test_existing_file_is_kept(self, tmp_path: Path, messages: dict) -> None:
        target = tmp_path / "garak.yaml"
        target.write_text("edited: true\n")
        cli_prompts.write_garak_scaffold(8000, path=target)
        assert target.read_text() == "edited: true\n"
        assert messages["info"] == [f"{target} exists; keeping it."]

    def test_force_overwrites_existing_file(self, tmp_path: Path, messages: dict) -> None:
        target = tmp_path / "garak.yaml"
        target.write_text("edited: true\n")
        cli_prompts.write_garak_scaffold(8000, force=True, path=target)
        assert "edited" not in target.read_text()


class TestPromptBackends:
    def test_declining_returns_empty(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script_prompt(monkeypatch, "n")
        assert cli_prompts.prompt_backends() == []

    def test_collects_two_backends_with_derived_names_and_filtered_ports(
        self, monkeypatch: pytest.MonkeyPatch, messages: dict
    ) -> None:
        script_prompt(
            monkeypatch,
            "y",
            # backend 1: compose file -> name defaults to its folder
            "deploy/pg/docker-compose.yml", "pg", "5432, x, 5433", "http://h/health",
            # backend 2: already running, blank name accepted -> backend2
            "", "", "6379", "",
        )  # fmt: skip
        script_confirm(monkeypatch, True, False)
        assert cli_prompts.prompt_backends() == [
            {
                "name": "pg",
                "ports": [5432, 5433],
                "compose_file": "deploy/pg/docker-compose.yml",
                "health_url": "http://h/health",
            },
            {"name": "backend2", "ports": [6379]},
        ]

    def test_blank_name_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script = script_prompt(monkeypatch, "y", "deploy/redis/compose.yml", "  ", "6379", "")
        script_confirm(monkeypatch, False)
        (backend,) = cli_prompts.prompt_backends()
        assert backend["name"] == "redis"
        assert script.calls[2][1]["default"] == "redis"

    def test_default_name_without_compose_is_numbered(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script = script_prompt(monkeypatch, "y", "", "", "80", "")
        script_confirm(monkeypatch, False)
        (backend,) = cli_prompts.prompt_backends()
        assert backend["name"] == "backend1"
        assert script.calls[2][1]["default"] == "backend1"


class TestResolveProjectLayout:
    def test_walks_up_to_project_root_and_finds_dockerfiles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        (tmp_path / "pyproject.toml").touch()
        agent = tmp_path / "agents" / "a"
        agent.mkdir(parents=True)
        (agent / "Dockerfile").touch()
        monkeypatch.chdir(tmp_path)

        project_dir, root, dockerfiles = cli_prompts.resolve_project_layout(agent, interactive=True)

        assert project_dir == "."
        assert root.resolve() == tmp_path.resolve()
        assert [d.name for d in dockerfiles] == ["Dockerfile"]
        assert any(line.startswith("Searching") for line in messages["info"])

    def test_interactive_without_root_or_dockerfiles_explains(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        monkeypatch.setattr(cli_prompts, "find_project_root", lambda _path: None)
        monkeypatch.setattr(cli_prompts, "discover_dockerfiles", lambda _path: [])
        monkeypatch.chdir(tmp_path)

        layout = cli_prompts.resolve_project_layout(tmp_path, interactive=True)

        assert layout == (".", tmp_path, [])
        assert any("no pyproject.toml/uv.lock found" in line for line in messages["info"])
        assert any("None found" in line for line in messages["info"])

    def test_non_interactive_is_silent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict) -> None:
        monkeypatch.setattr(cli_prompts, "find_project_root", lambda _path: None)
        monkeypatch.chdir(tmp_path)
        cli_prompts.resolve_project_layout(tmp_path, interactive=False)
        assert messages["info"] == []

    def test_falls_back_to_absolute_path_on_relpath_valueerror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        def different_drive(*_args: object) -> str:
            raise ValueError

        monkeypatch.setattr(cli_prompts.os.path, "relpath", different_drive)
        assert cli_prompts.resolve_project_layout(tmp_path, interactive=False)[0] == str(tmp_path.resolve())


class TestPromptByo:
    def test_uses_detected_dockerfile_as_default_and_splits_binaries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        script = script_prompt(monkeypatch, " deploy/Dockerfile ", " /a/*, ,/b/** ")
        result = cli_prompts.prompt_byo("deploy/Dockerfile", tmp_path)
        assert result == {"dockerfile": "deploy/Dockerfile", "binaries": ["/a/*", "/b/**"]}
        assert script.calls[0][1] == {"default": "deploy/Dockerfile"}

    def test_no_detected_dockerfile_means_no_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        script = script_prompt(monkeypatch, "Dockerfile", "")
        result = cli_prompts.prompt_byo(None, tmp_path)
        assert "default" not in script.calls[0][1]
        assert result == {"dockerfile": "Dockerfile", "binaries": []}


class TestPromptLaunchMode:
    def test_detected_dockerfile_is_relative_and_counts_extras(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        seen: dict[str, object] = {}

        def fake_byo(detected: str | None, root: Path) -> dict:
            seen["detected"] = detected
            return {"dockerfile": detected, "binaries": []}

        monkeypatch.setattr(cli_prompts, "prompt_byo", fake_byo)
        (tmp_path / "deploy").mkdir()
        dockerfiles = [tmp_path / "deploy" / "Dockerfile", tmp_path / "Dockerfile.dev"]

        cli_prompts.prompt_launch_mode(dockerfiles, tmp_path)

        assert seen["detected"] == "deploy/Dockerfile"
        assert any("detected: deploy/Dockerfile (+1 more)" in line for line in messages["info"])

    def test_no_dockerfiles_says_none_will_be_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        monkeypatch.setattr(cli_prompts, "prompt_byo", lambda *_args: {"dockerfile": "", "binaries": []})
        cli_prompts.prompt_launch_mode([], tmp_path)
        assert any("does not build one for you" in line for line in messages["info"])


class TestPromptStartCommand:
    def test_yes_without_value_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no --start-command given"):
            cli_prompts.prompt_start_command(yes=True)

    def test_interactive_strips_answer(self, monkeypatch: pytest.MonkeyPatch, messages: dict) -> None:
        script_prompt(monkeypatch, "  /app/.venv/bin/python -m x  ")
        assert cli_prompts.prompt_start_command(yes=False) == "/app/.venv/bin/python -m x"


class TestPromptEgress:
    @pytest.fixture
    def discovered(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        entries = ["api.example.com:443", "db.example.com"]
        monkeypatch.setattr(cli_prompts, "discover_egress_entries", lambda _root: list(entries))
        return entries

    def test_approved_entries_plus_deduplicated_extras(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict, discovered: list[str]
    ) -> None:
        script_confirm(monkeypatch, True)
        script_prompt(monkeypatch, "db.example.com, new.example.com , ,new.example.com")
        assert cli_prompts.prompt_egress(tmp_path) == ["api.example.com:443", "db.example.com", "new.example.com"]

    def test_rejecting_discovered_hosts_keeps_only_manual_ones(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict, discovered: list[str]
    ) -> None:
        script_confirm(monkeypatch, False)
        script_prompt(monkeypatch, "only.example.com")
        assert cli_prompts.prompt_egress(tmp_path) == ["only.example.com"]

    def test_nothing_discovered_skips_confirmation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        monkeypatch.setattr(cli_prompts, "discover_egress_entries", lambda _root: [])
        confirm = script_confirm(monkeypatch)
        script_prompt(monkeypatch, "")
        assert cli_prompts.prompt_egress(tmp_path) == []
        assert confirm.calls == []
        assert "No egress hosts found by scanning." in messages["info"]


class TestResolveLaunchMode:
    def test_explicit_dockerfile_wins_and_copies_binaries(self, tmp_path: Path) -> None:
        binaries = ["/a/**"]
        result = cli_prompts.resolve_launch_mode([], tmp_path, yes=True, dockerfile="D", binaries=binaries)
        assert result == {"dockerfile": "D", "binaries": ["/a/**"]}
        assert result["binaries"] is not binaries

    def test_explicit_dockerfile_without_binaries_has_empty_list(self, tmp_path: Path) -> None:
        assert cli_prompts.resolve_launch_mode([], tmp_path, yes=False, dockerfile="D")["binaries"] == []

    def test_yes_without_dockerfiles_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pass --dockerfile"):
            cli_prompts.resolve_launch_mode([], tmp_path, yes=True, binaries=["/a/**"])

    def test_yes_with_detected_dockerfile_requires_binaries(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--binary"):
            cli_prompts.resolve_launch_mode([tmp_path / "Dockerfile"], tmp_path, yes=True)

    def test_yes_picks_first_dockerfile_relative_to_root(self, tmp_path: Path) -> None:
        (tmp_path / "deploy").mkdir()
        result = cli_prompts.resolve_launch_mode(
            [tmp_path / "deploy" / "Dockerfile", tmp_path / "Dockerfile"], tmp_path, yes=True, binaries=["/x"]
        )
        assert result == {"dockerfile": "deploy/Dockerfile", "binaries": ["/x"]}

    def test_interactive_delegates_to_prompt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(cli_prompts, "prompt_launch_mode", lambda *_a: {"dockerfile": "P", "binaries": []})
        assert cli_prompts.resolve_launch_mode([], tmp_path, yes=False) == {"dockerfile": "P", "binaries": []}


class TestResolvePointedAt:
    def test_explicit_project_dir_wins(self, tmp_path: Path) -> None:
        assert cli_prompts.resolve_pointed_at(tmp_path, yes=False) == tmp_path

    def test_yes_uses_cwd(self) -> None:
        assert cli_prompts.resolve_pointed_at(None, yes=True) == Path()

    @pytest.mark.parametrize(("typed", "expected"), [(" sub/dir ", Path("sub/dir")), ("  ", Path())])
    def test_prompts_otherwise(
        self, monkeypatch: pytest.MonkeyPatch, messages: dict, typed: str, expected: Path
    ) -> None:
        script = script_prompt(monkeypatch, typed)
        assert cli_prompts.resolve_pointed_at(None, yes=False) == expected
        assert script.calls[0][1]["default"] == "."


class TestResolveAgentName:
    def test_explicit_name_wins(self) -> None:
        assert cli_prompts.resolve_agent_name("mine", "dflt", yes=False) == "mine"

    def test_yes_takes_default(self) -> None:
        assert cli_prompts.resolve_agent_name(None, "dflt", yes=True) == "dflt"

    @pytest.mark.parametrize(("typed", "expected"), [(" custom ", "custom"), ("  ", "dflt")])
    def test_prompt_strips_and_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, messages: dict, typed: str, expected: str
    ) -> None:
        script_prompt(monkeypatch, typed)
        assert cli_prompts.resolve_agent_name(None, "dflt", yes=False) == expected


class TestResolveSecretsFile:
    def test_explicit_value_is_stripped(self, tmp_path: Path) -> None:
        assert cli_prompts.resolve_secrets_file(" ../x/.env ", tmp_path, tmp_path, yes=False) == "../x/.env"

    def test_yes_uses_dot_env(self, tmp_path: Path) -> None:
        assert cli_prompts.resolve_secrets_file(None, tmp_path, tmp_path, yes=True) == ".env"

    def test_detected_env_is_cwd_relative_default_and_labelled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        agent = tmp_path / "agent"
        agent.mkdir()
        env = agent / ".env"
        env.touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_prompts, "discover_env_files", lambda _root, _pointed: [env])
        script = script_prompt(monkeypatch, "  agent/.env  ")

        assert cli_prompts.resolve_secrets_file(None, agent, agent, yes=False) == "agent/.env"
        label, kwargs = script.calls[0]
        assert "Agent secrets file (detected)" in label
        assert kwargs["default"] == "agent/.env"

    def test_no_detected_env_defaults_to_dot_env_without_marker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, messages: dict
    ) -> None:
        monkeypatch.setattr(cli_prompts, "discover_env_files", lambda _root, _pointed: [])
        script = script_prompt(monkeypatch, ".env")
        cli_prompts.resolve_secrets_file(None, tmp_path, tmp_path, yes=False)
        label, kwargs = script.calls[0]
        assert "detected" not in label
        assert kwargs["default"] == ".env"
