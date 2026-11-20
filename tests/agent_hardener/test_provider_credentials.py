# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rotated API key reaches the OpenShell provider, and a sandbox holding the old one is replaced."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from agent_hardener.openshell.credentials import (
    CredentialRecord,
    read_credential_record,
    write_credential_record,
    write_revision,
)
from agent_hardener.openshell.lifecycle import OpenShellConfig, OpenShellLifecycle
from agent_hardener.swarm_tracker.layout import (
    provider_credentials_path,
    sandbox_provider_revision_path,
    write_victim_image,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_JUDGE = "AGENT_HARDENER_SAFETY_API_KEY"
_IMAGE = "registry.example.com/team/agent:1"


class SequenceRunner:
    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []
        self.responses = responses

    def run(
        self, args: list[str], cwd: Path | None = None, *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        self.environments.append(env)
        returncode, stdout = self.responses.pop(0)
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)


@pytest.fixture(autouse=True)
def _no_judge_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The judge inherits the analysis key when it has none of its own.
    monkeypatch.delenv(_JUDGE, raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)


def _lifecycle(
    tmp_path: Path, runner: SequenceRunner, key: str | None = "new-key", **overrides: object
) -> OpenShellLifecycle:
    env_file = tmp_path / ".env"
    if key is not None:
        env_file.write_text(f"INFERENCE_API_KEY={key}\n", encoding="utf-8")
    fields: dict[str, object] = {
        "gateway": "gw",
        "sandbox": "sb",
        "policy_path": tmp_path / "policy.yaml",
        "image": _IMAGE,
        "provider": "p",
        "provider_env_file": env_file if key is not None else None,
        "provider_credentials": ["INFERENCE_API_KEY"],
        "cwd": tmp_path,
    }
    config = OpenShellConfig(**{**fields, **overrides})  # type: ignore[arg-type]
    return OpenShellLifecycle(config, runner=runner, log_dir=tmp_path)


def _record(tmp_path: Path, **credentials: str) -> CredentialRecord:
    record = CredentialRecord.of(credentials)
    write_credential_record(provider_credentials_path(tmp_path, "gw", "p"), record)
    return record


def _recorded(tmp_path: Path) -> CredentialRecord:
    record = read_credential_record(provider_credentials_path(tmp_path, "gw", "p"))
    assert record is not None
    return record


# --- the record ------------------------------------------------------------------------------------------


def test_a_record_names_each_added_removed_or_changed_key_and_never_holds_a_value(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    write_credential_record(path, CredentialRecord.of({"A": "secret-a", "B": "secret-b"}))
    record = read_credential_record(path)

    assert record is not None
    assert record.changed({"A": "secret-a", "B": "secret-b"}) == []
    assert record.changed({"A": "secret-a", "B": "rotated", "C": "secret-c"}) == ["B", "C"]
    assert record.changed({"A": "secret-a"}) == ["B"]
    assert "secret" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("content", [None, "not json", '{"salt": "s"}'], ids=["missing", "garbage", "partial"])
def test_an_unreadable_record_reads_as_none(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "credentials.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    assert read_credential_record(path) is None


# --- refreshing an existing provider ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "sent"),
    [(None, ["INFERENCE_API_KEY", _JUDGE]), ("old-key", ["INFERENCE_API_KEY"])],
    ids=["unrecorded-sends-every-key", "rotated-sends-only-the-changed-key"],
)
def test_an_existing_provider_is_updated_in_place_with_the_new_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None, sent: list[str]
) -> None:
    monkeypatch.setenv(_JUDGE, "judge-key")
    if recorded:
        _record(tmp_path, INFERENCE_API_KEY=recorded, **{_JUDGE: "judge-key"})
    runner = SequenceRunner([(0, "provider exists"), (0, "profile imported"), (0, "provider updated")])

    result = _lifecycle(tmp_path, runner).ensure_provider()

    assert result.ok is True
    assert "refreshing provider" in result.output
    assert runner.commands[1][1:4] == ["provider", "profile", "import"]
    # Every key is passed by name and read from a child-only environment, so no value reaches argv.
    assert runner.commands[2] == [
        "openshell",
        "provider",
        "update",
        "p",
        "--gateway",
        "gw",
        *[part for name in sent for part in ("--credential", name)],
    ]
    child_env = runner.environments[2]
    assert child_env is not None
    assert child_env["INFERENCE_API_KEY"] == "new-key"
    assert not any(key in str(runner.commands) for key in ("new-key", "judge-key"))
    assert _recorded(tmp_path).changed({"INFERENCE_API_KEY": "new-key", _JUDGE: "judge-key"}) == []


def test_an_existing_provider_with_current_keys_is_left_alone(tmp_path: Path) -> None:
    _record(tmp_path, INFERENCE_API_KEY="new-key")
    runner = SequenceRunner([(0, "provider exists")])

    assert _lifecycle(tmp_path, runner).ensure_provider().ok is True
    assert len(runner.commands) == 1


def test_a_custom_provider_type_is_updated_without_a_generated_profile(tmp_path: Path) -> None:
    _record(tmp_path, INFERENCE_API_KEY="old-key")
    runner = SequenceRunner([(0, "provider exists"), (0, "provider updated")])

    assert _lifecycle(tmp_path, runner, provider_type="custom").ensure_provider().ok is True
    assert runner.commands[1][1:3] == ["provider", "update"]


def test_a_key_dropped_from_the_config_is_recorded_without_an_update(tmp_path: Path) -> None:
    _record(tmp_path, INFERENCE_API_KEY="new-key", RETIRED_KEY="gone")
    runner = SequenceRunner([(0, "provider exists")])

    assert _lifecycle(tmp_path, runner).ensure_provider().ok is True
    assert len(runner.commands) == 1
    assert _recorded(tmp_path).changed({"INFERENCE_API_KEY": "new-key"}) == []


def test_with_refreshing_off_a_stale_provider_is_only_warned_about(tmp_path: Path) -> None:
    _record(tmp_path, INFERENCE_API_KEY="old-key")
    runner = SequenceRunner([(0, "provider exists")])
    lifecycle = _lifecycle(tmp_path, runner, refresh_provider_credentials=False)

    result = lifecycle.ensure_provider()

    assert result.ok is True
    assert "credentials changed: INFERENCE_API_KEY; not refreshed" in result.output
    assert len(runner.commands) == 1
    assert _recorded(tmp_path).changed({"INFERENCE_API_KEY": "old-key"}) == []
    assert lifecycle.credentials_drift() is None


@pytest.mark.parametrize(
    "responses",
    [
        [(0, "provider exists"), (0, "profile imported"), (1, "update rejected")],
        [(0, "provider exists"), (1, "import failed"), (1, "export failed")],
    ],
    ids=["update-fails", "profile-fails"],
)
def test_a_failed_refresh_fails_and_keeps_the_old_record(tmp_path: Path, responses: list[tuple[int, str]]) -> None:
    _record(tmp_path, INFERENCE_API_KEY="old-key")

    assert _lifecycle(tmp_path, SequenceRunner(responses)).ensure_provider().ok is False
    assert _recorded(tmp_path).changed({"INFERENCE_API_KEY": "old-key"}) == []


def test_an_unreadable_env_file_leaves_an_existing_provider_alone(tmp_path: Path) -> None:
    (tmp_path / ".env").mkdir()
    runner = SequenceRunner([(0, "provider exists")])
    lifecycle = _lifecycle(tmp_path, runner, key=None, provider_env_file=tmp_path / ".env")

    result = lifecycle.ensure_provider()

    assert result.ok is True
    assert "warning: failed to read provider_env_file" in result.output
    assert lifecycle.credentials_drift() is None


def test_a_provider_without_configured_values_is_refreshed_from_existing_state(tmp_path: Path) -> None:
    runner = SequenceRunner([(0, "provider exists"), (0, "profile imported"), (0, "provider updated")])

    assert _lifecycle(tmp_path, runner, key=None).ensure_provider().ok is True
    assert runner.commands[2][-1] == "--from-existing"


# --- replacing a sandbox that holds the old key ----------------------------------------------------------


def test_up_updates_the_provider_and_replaces_a_sandbox_holding_the_old_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_hardener.openshell.support.time.sleep", lambda _seconds: None)
    write_victim_image(tmp_path, "sb", _IMAGE)
    old = _record(tmp_path, INFERENCE_API_KEY="old-key")
    write_revision(sandbox_provider_revision_path(tmp_path, "sb"), old.revision)
    runner = SequenceRunner(
        [
            (0, "gateway ok"),
            (0, "Phase: Ready"),
            (0, "deletion accepted"),
            (1, "sandbox not found: NotFound"),
            (0, "provider exists"),
            (0, "profile imported"),
            (0, "provider updated"),
            (0, "sandbox created"),
            (0, "Phase: Ready"),
        ]
    )
    lifecycle = _lifecycle(tmp_path, runner)
    assert lifecycle.credentials_drift() == "provider p credentials changed: INFERENCE_API_KEY"

    assert lifecycle.up().ok is True

    verbs = [command[1:3] for command in runner.commands]
    assert verbs.index(["sandbox", "delete"]) < verbs.index(["provider", "update"]) < verbs.index(["sandbox", "create"])
    update = verbs.index(["provider", "update"])
    assert runner.commands[update][-2:] == ["--credential", "INFERENCE_API_KEY"]
    assert (runner.environments[update] or {})["INFERENCE_API_KEY"] == "new-key"
    # The new sandbox records the provider revision it started under, so the next run reuses it.
    assert lifecycle.credentials_drift() is None


def test_up_keeps_a_sandbox_whose_keys_are_current(tmp_path: Path) -> None:
    write_victim_image(tmp_path, "sb", _IMAGE)
    write_revision(
        sandbox_provider_revision_path(tmp_path, "sb"), _record(tmp_path, INFERENCE_API_KEY="new-key").revision
    )
    runner = SequenceRunner([(0, "gateway ok"), (0, "Phase: Ready")])

    assert _lifecycle(tmp_path, runner).up().ok is True
    assert len(runner.commands) == 2


@pytest.mark.parametrize("sandbox_revision", [None, "an-older-revision"], ids=["unrecorded", "older"])
def test_a_sandbox_started_before_the_provider_was_updated_is_stale(
    tmp_path: Path, sandbox_revision: str | None
) -> None:
    """Another run may have refreshed a shared provider; this sandbox still resolves the old key."""
    _record(tmp_path, INFERENCE_API_KEY="new-key")
    write_revision(sandbox_provider_revision_path(tmp_path, "sb"), sandbox_revision)

    drift = _lifecycle(tmp_path, SequenceRunner([])).credentials_drift()

    assert drift == "sandbox sb was created before provider p was last updated"


def test_without_a_provider_there_is_nothing_to_drift(tmp_path: Path) -> None:
    lifecycle = OpenShellLifecycle(
        OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", cwd=tmp_path),
        runner=SequenceRunner([]),
    )

    assert lifecycle.credentials_drift() is None
    assert lifecycle._provider_revision() is None
