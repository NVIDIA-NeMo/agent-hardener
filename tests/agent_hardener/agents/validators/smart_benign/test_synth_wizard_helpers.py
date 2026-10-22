# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wizard tests: validator resolution, answer modes, the printed summary, and the Langfuse dataset upload."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign import driver, wizard
from agent_hardener.agents.validators.smart_benign.models import (
    GeneratedRequest,
    Persona,
    ToolSpec,
    VictimCapabilityProfile,
)
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.tracing import TracingContext
from agent_hardener.models import SessionConfig

if TYPE_CHECKING:
    from pathlib import Path

_ENDPOINT = "http://victim.test/v1/chat/completions"


def _validator(name: str, implementation: str = wizard.SMART_BENIGN_IMPL, **config: Any) -> dict[str, Any]:
    return {
        "name": name,
        "role": "validator",
        "implementation": implementation,
        "config": {"kind": "benign", "skip_nl_parser": True, "skip_github_analysis": True, **config},
    }


def _session(root: Path, *validators: dict[str, Any]) -> SessionConfig:
    return SessionConfig.model_validate(
        {
            "storage": {"root_dir": root},
            "target": {"name": "victim", "base_url": _ENDPOINT},
            "victim": {"name": "victim", "role": "victim", "implementation": "pkg:run"},
            "benign_validators": list(validators) or [_validator("smart")],
        }
    )


def _request(payload: str = "ls") -> GeneratedRequest:
    return GeneratedRequest(tool="bash", payload=payload, label="benign")


# --- _resolve_validator ------------------------------------------------------------------------------------


def test_the_only_smart_validator_is_chosen_without_a_name(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("smart"))

    assert wizard._resolve_validator(session, None).name == "smart"


def test_other_validator_implementations_are_ignored(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("classic", "pkg:other"), _validator("smart"))

    assert wizard._resolve_validator(session, None).name == "smart"


def test_a_named_validator_is_picked_among_several(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("a"), _validator("b"))

    assert wizard._resolve_validator(session, "b").name == "b"


def test_several_validators_without_a_name_ask_the_user_to_choose(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("a"), _validator("b"))

    with pytest.raises(SystemExit, match=r"--validator <name>") as exc_info:
        wizard._resolve_validator(session, None)

    assert "['a', 'b']" in str(exc_info.value)


def test_an_unknown_name_lists_the_available_ones(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("a"))

    with pytest.raises(SystemExit, match="no benign_validator named 'zzz'") as exc_info:
        wizard._resolve_validator(session, "zzz")

    assert "['a']" in str(exc_info.value)


def test_naming_a_validator_with_another_implementation_is_not_found(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("classic", "pkg:other"), _validator("smart"))

    with pytest.raises(SystemExit, match="no benign_validator named 'classic'"):
        wizard._resolve_validator(session, "classic")


def test_no_smart_validator_at_all_is_an_error(tmp_path: Path) -> None:
    session = _session(tmp_path, _validator("classic", "pkg:other"))

    with pytest.raises(SystemExit, match="no benign_validator entry with implementation"):
        wizard._resolve_validator(session, None)


# --- run_synth_wizard --------------------------------------------------------------------------------------


class _Run:
    """Runs the real wizard with config loading, tracing and the compiled graph faked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, root: Path, final: SynthState, *validators: Any) -> None:
        self.root, self.final = root, final
        self.started: list[SynthState] = []
        self.lf_configs: list[Any] = []
        self.tracing = TracingContext()
        session = _session(root, *validators)
        monkeypatch.setattr(wizard, "load_config", lambda _path: session)
        monkeypatch.setattr(wizard, "make_langfuse_callbacks", lambda **_kw: self.tracing)
        monkeypatch.setattr(driver, "build_synth_graph", lambda **_kw: self)

    async def astream(self, state: Any, config: Any, stream_mode: Any) -> Any:
        self.started.append(state)
        self.lf_configs.append(config)
        yield ("values", self.final)

    def __call__(self, **kwargs: Any) -> int:
        async def no_answers(_questions: Any) -> list[dict[str, Any]]:
            return []

        return wizard.run_synth_wizard(self.root / "cfg.yaml", answer_provider=no_answers, **kwargs)


def _final(**kwargs: Any) -> SynthState:
    return SynthState(inputs=SynthInputs(target_name="victim"), **kwargs)


def _tty(monkeypatch: pytest.MonkeyPatch, *, is_tty: bool) -> None:
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: is_tty, raising=False)


@pytest.mark.parametrize(
    ("kwargs", "tty", "interactive", "label"),
    [
        ({"yes": True}, False, True, "auto-accepted defaults"),  # --yes needs no TTY
        ({"yes": True, "no_interactive": True}, True, True, "auto-accepted defaults"),
        ({}, True, True, "prompted (TTY)"),
        ({}, False, False, "skipped (rules-only)"),  # piped stdin silently degrades to rules-only
        ({"no_interactive": True}, True, False, "skipped (rules-only)"),
    ],
    ids=["yes-no-tty", "yes-beats-no-interactive", "tty", "no-tty", "no-interactive"],
)
def test_answer_mode_is_derived_from_flags_and_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    kwargs: dict[str, Any],
    tty: bool,
    interactive: bool,
    label: str,
) -> None:
    run = _Run(monkeypatch, tmp_path, _final(), _validator("smart"))
    _tty(monkeypatch, is_tty=tty)

    assert run(**kwargs) == 0

    assert run.started[0].interactive is interactive
    assert f"Answers:       {label}" in capsys.readouterr().out


def test_the_artifact_dir_is_under_the_storage_root_by_target_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    run = _Run(monkeypatch, tmp_path, _final(), _validator("smart"))

    run(no_interactive=True)

    expected = tmp_path / "benign_profiles" / "victim"
    assert run.started[0].artifact_dir == expected
    assert f"Artifact dir:  {expected}" in capsys.readouterr().out


def test_preseeded_interview_answers_are_not_reported_as_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    final = _final(
        interviewer_answers=[("", "Seeded?", "yes"), ("scope", "New one?", "only reads")],
    )
    run = _Run(monkeypatch, tmp_path, final, _validator("smart", interview_answers=[["Seeded?", "yes"]]))

    run(no_interactive=True)

    out = capsys.readouterr().out
    assert '- ["New one?", "only reads"]' in out
    assert '- ["Seeded?", "yes"]' not in out


def test_tracing_callbacks_are_forwarded_only_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(monkeypatch, tmp_path, _final(), _validator("smart"))
    run(no_interactive=True)
    assert "callbacks" not in run.lf_configs[0]

    handler = object()
    run.tracing = TracingContext(callbacks=[handler], enabled=True)
    run(no_interactive=True)
    assert run.lf_configs[1]["callbacks"] == [handler]


def test_an_unknown_validator_name_aborts_before_running_synth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(monkeypatch, tmp_path, _final(), _validator("smart"))

    with pytest.raises(SystemExit):
        run(no_interactive=True, validator="nope")

    assert run.started == []


def test_upload_dataset_goes_through_the_tracing_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeLangfuse()
    final = _final(requests=[_request("ls")])
    run = _Run(monkeypatch, tmp_path, final, _validator("smart"))
    run.tracing = TracingContext(client=client, enabled=True)

    assert run(no_interactive=True, upload_dataset=True) == 0

    assert [i["dataset_name"] for i in client.items] == ["victim-benign"]


# --- _print_summary ----------------------------------------------------------------------------------------


def _profile() -> VictimCapabilityProfile:
    return VictimCapabilityProfile(
        target_name="victim",
        tools=[ToolSpec(name="bash", description="d", source="nl", confidence=0.9)],
        personas=[Persona(name="Dana", description="nurse"), Persona(name="Eli", description="dev")],
        out_of_scope=["legal"],
        input_hash="h",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_summary_counts_unresolved_gaps_against_answered_ones(tmp_path: Path, capsys: Any) -> None:
    final = _final(
        profile=_profile(),
        gaps=["scope", "personas", "limits"],
        interviewer_answers=[("scope", "Q?", "A"), ("", "seeded", "x")],
        interview_questions_asked=3,
        requests=[_request()],
    )

    wizard._print_summary(final, tmp_path, answers_mode="prompted (TTY)")

    out = capsys.readouterr().out
    assert "Tools:         1" in out
    assert "Personas:      2" in out
    assert "Out-of-scope:  1" in out
    assert "Gaps:          2 unresolved (3 total)" in out
    assert "Q&A collected: 2 (3 question(s) asked)" in out
    assert "Requests:      1" in out


def test_summary_without_a_profile_reports_zeroes(tmp_path: Path, capsys: Any) -> None:
    wizard._print_summary(_final(), tmp_path, answers_mode="x")

    out = capsys.readouterr().out
    assert "Tools:         0" in out
    assert "Personas:      0" in out
    assert "Out-of-scope:  0" in out


def test_summary_shows_only_the_first_five_errors(tmp_path: Path, capsys: Any) -> None:
    final = _final(errors=[f"err-{i}" for i in range(8)])

    wizard._print_summary(final, tmp_path, answers_mode="x")

    out = capsys.readouterr().out
    assert "Errors:        8" in out
    assert "- err-4" in out
    assert "err-5" not in out


def test_summary_without_errors_has_no_error_section(tmp_path: Path, capsys: Any) -> None:
    wizard._print_summary(_final(), tmp_path, answers_mode="x")

    assert "First errors:" not in capsys.readouterr().out


def test_summary_points_at_the_editable_suite_and_the_run_command(tmp_path: Path, capsys: Any) -> None:
    wizard._print_summary(_final(), tmp_path, answers_mode="x")

    out = capsys.readouterr().out
    assert f"uv run agent-hardener run --benign-suite {tmp_path / 'requests.csv'}" in out


def test_summary_only_suggests_newly_collected_answers(tmp_path: Path, capsys: Any) -> None:
    final = _final(interviewer_answers=[("", "old?", "a"), ("g", "new?", "b")])

    wizard._print_summary(final, tmp_path, answers_mode="x", preseeded=1)

    out = capsys.readouterr().out
    assert "interview_answers:" in out
    assert '["new?", "b"]' in out
    assert "old?" not in out


def test_summary_without_new_answers_omits_the_hint(tmp_path: Path, capsys: Any) -> None:
    final = _final(interviewer_answers=[("", "old?", "a")])

    wizard._print_summary(final, tmp_path, answers_mode="x", preseeded=1)

    assert "interview_answers:" not in capsys.readouterr().out


# --- _upload_to_langfuse_dataset ---------------------------------------------------------------------------


class _FakeLangfuse:
    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.datasets: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.flushed = 0

    def create_dataset(self, **kwargs: Any) -> None:
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs: Any) -> None:
        if kwargs["input"]["payload"] in self.fail_on:
            raise RuntimeError("langfuse down")
        self.items.append(kwargs)

    def flush(self) -> None:
        self.flushed += 1


def test_upload_is_skipped_when_tracing_is_disabled(capsys: Any) -> None:
    wizard._upload_to_langfuse_dataset([_request()], "victim", TracingContext())

    assert "skipped" in capsys.readouterr().out


def test_upload_is_skipped_when_enabled_but_the_client_is_missing(capsys: Any) -> None:
    wizard._upload_to_langfuse_dataset([_request()], "victim", TracingContext(enabled=True, client=None))

    assert "skipped" in capsys.readouterr().out


def test_upload_creates_a_named_dataset_and_one_item_per_request(capsys: Any) -> None:
    client = _FakeLangfuse()
    requests = [
        GeneratedRequest(tool="bash", payload="ls", label="benign", rationale="basic", persona="dev"),
        GeneratedRequest(tool="py", payload="print(1)", label="negative_control"),
    ]

    wizard._upload_to_langfuse_dataset(requests, "victim", TracingContext(enabled=True, client=client))

    assert client.datasets == [{"name": "victim-benign", "description": "Benign requests for victim"}]
    assert client.items[0] == {
        "dataset_name": "victim-benign",
        "input": {"payload": "ls", "tool": "bash", "persona": "dev"},
        "expected_output": {"label": "benign"},
        "metadata": {"rationale": "basic"},
    }
    assert client.items[1]["input"]["persona"] is None
    assert client.flushed == 1
    assert "uploaded 2 item(s) to 'victim-benign'" in capsys.readouterr().out


def test_one_failing_item_does_not_stop_the_rest(capsys: Any, caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeLangfuse(fail_on={"bad"})
    requests = [_request("ok-1"), _request("bad"), _request("ok-2")]

    wizard._upload_to_langfuse_dataset(requests, "victim", TracingContext(enabled=True, client=client))

    assert [i["input"]["payload"] for i in client.items] == ["ok-1", "ok-2"]
    assert client.flushed == 1
    assert "uploaded 2 item(s)" in capsys.readouterr().out
    assert "failed to upload dataset item" in caplog.text
