# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading recorded attack hits (``run --replay``) into AttackRecords."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.attackers.preloaded import load_preloaded_attacks
from agent_hardener.models import PreloadedAttackConfig

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _load(path: Path, **kwargs):
    return load_preloaded_attacks([PreloadedAttackConfig(name="My Attacker", path=path, **kwargs)])


def test_jsonl_skips_blank_lines_and_tags_each_hit_with_source_and_index(tmp_path: Path) -> None:
    path = tmp_path / "hits.jsonl"
    path.write_text('{"probe": "a"}\n\n   \n{"probe": "b", "source": "kept"}\n', encoding="utf-8")

    attacks = _load(path, source="recorded", metadata={"run": 3})

    assert [a.agent_id for a in attacks] == ["preloaded-my-attacker-0", "preloaded-my-attacker-1"]
    first, second = (a.records[0] for a in attacks)
    assert first == {"probe": "a", "source": "recorded", "hit_index": 0}
    assert second["source"] == "kept"  # an existing source is never overwritten
    assert second["hit_index"] == 1
    assert attacks[0].agent_name == "My Attacker"
    assert attacks[0].metadata == {"preloaded": True, "path": str(path), "run": 3}


def test_invalid_jsonl_line_reports_file_and_line_number(tmp_path: Path) -> None:
    path = tmp_path / "hits.jsonl"
    path.write_text('{"ok": 1}\n{not json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid JSONL in .*hits\.jsonl at line 2"):
        _load(path)


@pytest.mark.parametrize("key", ["records", "hits", "attacks"])
def test_json_object_unwraps_known_list_keys(tmp_path: Path, key: str) -> None:
    path = tmp_path / "hits.json"
    path.write_text(json.dumps({key: [{"probe": "p1"}, {"probe": "p2"}], "other": [1]}), encoding="utf-8")

    attacks = _load(path)

    assert [a.records[0]["probe"] for a in attacks] == ["p1", "p2"]


def test_json_list_payload_wraps_non_dict_items(tmp_path: Path) -> None:
    path = tmp_path / "hits.json"
    path.write_text(json.dumps([{"probe": "p"}, "bare string"]), encoding="utf-8")

    attacks = _load(path)

    assert attacks[1].records[0]["value"] == "bare string"


def test_json_object_without_known_keys_is_a_single_record(tmp_path: Path) -> None:
    path = tmp_path / "hits.json"
    path.write_text(json.dumps({"probe": "solo", "records": "not-a-list"}), encoding="utf-8")

    attacks = _load(path)

    assert len(attacks) == 1
    assert attacks[0].records[0]["probe"] == "solo"


def test_json_scalar_payload_becomes_value_record(tmp_path: Path) -> None:
    path = tmp_path / "hits.json"
    path.write_text("42", encoding="utf-8")

    assert _load(path)[0].records[0]["value"] == 42


def test_suffix_check_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "HITS.JSONL"
    path.write_text('{"probe": "a"}\n', encoding="utf-8")

    assert len(_load(path)) == 1


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _load(tmp_path / "nope.jsonl")


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hits.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be \.jsonl or \.json"):
        _load(path)


def test_multiple_configs_are_concatenated_in_order(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"probe": "a"}\n', encoding="utf-8")
    b.write_text('{"probe": "b"}\n', encoding="utf-8")

    attacks = load_preloaded_attacks(
        [PreloadedAttackConfig(name="first", path=a), PreloadedAttackConfig(name="second", path=b)]
    )

    assert [x.agent_name for x in attacks] == ["first", "second"]
