# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the profile_writer subgraph: artifact layout on disk and partial-failure reporting."""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.nodes import write as wr
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.nodes.write import CSV_HEADER
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.state import ProfileWriterState
from agent_hardener.agents.validators.smart_benign.validator import load_requests

if TYPE_CHECKING:
    from pathlib import Path


def _profile(input_hash: str = "abc123") -> VictimCapabilityProfile:
    return VictimCapabilityProfile(
        target_name="victim",
        system_role="support bot",
        input_hash=input_hash,
        generated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


def _request(payload: str, persona: str | None = None, rationale: str = "") -> GeneratedRequest:
    return GeneratedRequest(tool="bash", payload=payload, label="benign", rationale=rationale, persona=persona)


def _state(target_dir: Path, **overrides: Any) -> ProfileWriterState:
    return ProfileWriterState(target_dir=target_dir, **overrides)


def _written(delta: dict[str, object]) -> list[Path]:
    return cast("list[Path]", delta["written_files"])


def test_no_profile_writes_nothing_and_reports_an_error(tmp_path: Path) -> None:
    target = tmp_path / "out"

    delta = asyncio.run(wr.write(_state(target, profile=None)))

    assert delta["written_files"] == []
    assert delta["errors"] == ["profile_writer: no profile to write"]
    assert not target.exists()


def test_all_artifacts_land_in_the_target_dir_creating_parents(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "victim"
    state = _state(
        target,
        profile=_profile("hash-1"),
        requests=[_request("ls")],
        source_notes={"nl_parser": "parsed", "api_prober": "probed 3/5"},
    )

    delta = asyncio.run(wr.write(state))

    assert sorted(p.relative_to(target).as_posix() for p in _written(delta)) == [
        "input_hash.txt",
        "profile.json",
        "requests.csv",
        "sources/api_prober.json",
        "sources/nl_parser.json",
    ]
    assert all(p.is_file() for p in _written(delta))
    assert "errors" not in delta
    assert delta["source_note"] == f"wrote 5 artifact(s) to {target}"
    assert (target / "input_hash.txt").read_text(encoding="utf-8") == "hash-1\n"
    assert json.loads((target / "sources" / "api_prober.json").read_text(encoding="utf-8")) == {"note": "probed 3/5"}


def test_profile_json_round_trips(tmp_path: Path) -> None:
    profile = _profile()

    asyncio.run(wr.write(_state(tmp_path, profile=profile)))

    reloaded = VictimCapabilityProfile.model_validate_json((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert reloaded == profile


def test_no_source_notes_means_no_sources_dir(tmp_path: Path) -> None:
    delta = asyncio.run(wr.write(_state(tmp_path, profile=_profile())))

    assert not (tmp_path / "sources").exists()
    assert len(_written(delta)) == 3


def test_requests_csv_keeps_awkward_payloads_intact_and_loads_back(tmp_path: Path) -> None:
    awkward = 'say "hi", then\nrun ls; échappé'
    requests = [_request(awkward, persona="dev", rationale="has, comma"), _request("plain")]

    asyncio.run(wr.write(_state(tmp_path, profile=_profile(), requests=requests)))

    with (tmp_path / "requests.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CSV_HEADER
    assert rows[1] == ["bash", awkward, "benign", "has, comma", "dev"]
    assert rows[2][4] == ""  # a missing persona is written blank, not "None"
    assert [(r.payload, r.persona) for r in load_requests(tmp_path / "requests.csv")] == [
        (awkward, "dev"),
        ("plain", None),
    ]


def test_an_empty_suite_still_writes_a_header_only_csv(tmp_path: Path) -> None:
    asyncio.run(wr.write(_state(tmp_path, profile=_profile())))

    assert (tmp_path / "requests.csv").read_text(encoding="utf-8").strip() == ",".join(CSV_HEADER)


def test_rewriting_overwrites_previous_artifacts(tmp_path: Path) -> None:
    asyncio.run(wr.write(_state(tmp_path, profile=_profile("old"), requests=[_request("one"), _request("two")])))

    asyncio.run(wr.write(_state(tmp_path, profile=_profile("new"), requests=[_request("only")])))

    assert (tmp_path / "input_hash.txt").read_text(encoding="utf-8") == "new\n"
    assert [r.payload for r in load_requests(tmp_path / "requests.csv")] == ["only"]


def test_an_io_failure_reports_the_files_already_written(tmp_path: Path) -> None:
    (tmp_path / "requests.csv").mkdir()  # a directory squatting on the CSV path makes that write fail

    delta = asyncio.run(wr.write(_state(tmp_path, profile=_profile(), requests=[_request("ls")])))

    assert [p.name for p in _written(delta)] == ["profile.json"]
    assert len(cast("list[str]", delta["errors"])) == 1
    assert cast("list[str]", delta["errors"])[0].startswith("profile_writer: ")
    assert delta["source_note"] == f"partial write to {tmp_path} (1 file(s))"
    assert not (tmp_path / "input_hash.txt").exists()


def test_a_target_dir_that_is_a_file_fails_before_writing_anything(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir", encoding="utf-8")

    delta = asyncio.run(wr.write(_state(blocker, profile=_profile())))

    assert delta["written_files"] == []
    assert delta["source_note"] == f"partial write to {blocker} (0 file(s))"
