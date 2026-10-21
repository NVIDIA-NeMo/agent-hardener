# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from agent_hardener.yaml_diff import YamlDiff, build_yaml_diff, parse_unified_diff

_BEFORE = "a: 1\nb: 2\nc: 3\n"


@pytest.mark.parametrize(
    ("before", "after"),
    [(None, "a: 1"), ("a: 1", None), (None, None), ("a: 1\n", "a: 1\n")],
    ids=["missing-before", "missing-after", "both-missing", "identical"],
)
def test_no_diff_when_a_side_is_missing_or_unchanged(before: str | None, after: str | None) -> None:
    diff = build_yaml_diff(before, after)

    assert diff.is_empty
    assert (diff.added, diff.removed) == (0, 0)


def test_labels_are_kept_even_when_the_diff_is_empty() -> None:
    diff = build_yaml_diff("a", "a", from_label="live", to_label="proposed")

    assert (diff.from_label, diff.to_label) == ("live", "proposed")


def test_changed_value_is_one_remove_and_one_add_with_line_numbers() -> None:
    diff = build_yaml_diff(_BEFORE, "a: 1\nb: 9\nc: 3\n", context=1)

    assert (diff.added, diff.removed) == (1, 1)
    (hunk,) = diff.hunks
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 3, 1, 3)
    triples = [(line.kind, line.old_lineno, line.new_lineno, line.text) for line in hunk.lines]
    assert triples == [
        ("context", 1, 1, "a: 1"),
        ("remove", 2, None, "b: 2"),
        ("add", None, 2, "b: 9"),
        ("context", 3, 3, "c: 3"),
    ]


def test_context_width_controls_surrounding_lines() -> None:
    before = "\n".join(f"k{i}: {i}" for i in range(10))
    after = before.replace("k5: 5", "k5: changed")

    tight = build_yaml_diff(before, after, context=0)
    wide = build_yaml_diff(before, after, context=2)

    assert [line.kind for line in tight.hunks[0].lines] == ["remove", "add"]
    assert sum(line.kind == "context" for line in wide.hunks[0].lines) == 4


def test_distant_changes_become_separate_hunks() -> None:
    before = "\n".join(f"k{i}: {i}" for i in range(30))
    after = before.replace("k1: 1", "k1: x").replace("k28: 28", "k28: y")

    diff = build_yaml_diff(before, after, context=1)

    assert len(diff.hunks) == 2
    assert (diff.added, diff.removed) == (2, 2)


def test_pure_addition_and_pure_removal_are_counted_separately() -> None:
    added = build_yaml_diff("a: 1", "a: 1\nb: 2", context=0)
    removed = build_yaml_diff("a: 1\nb: 2", "a: 1", context=0)

    assert (added.added, added.removed) == (1, 0)
    assert (removed.added, removed.removed) == (0, 1)


def test_to_json_round_trips_through_the_model() -> None:
    diff = build_yaml_diff(_BEFORE, "a: 1\nb: 9\nc: 3\n")

    assert YamlDiff.model_validate(diff.to_json()) == diff
    assert diff.to_json()["hunks"][0]["lines"][0]["kind"] == "context"


def test_parse_unified_diff_matches_build_for_the_same_change() -> None:
    built = build_yaml_diff(_BEFORE, "a: 1\nb: 9\nc: 3\n", from_label="x", to_label="y")
    text = "--- x\n+++ y\n@@ -1,3 +1,3 @@\n a: 1\n-b: 2\n+b: 9\n c: 3\n"

    parsed = parse_unified_diff(text, from_label="x", to_label="y")

    assert parsed == built


def test_parse_ignores_file_headers_and_text_before_the_first_hunk() -> None:
    text = "diff --git a/p b/p\nindex 1..2\n--- a/p\n+++ b/p\n@@ -1 +1 @@\n-old\n+new\n"

    diff = parse_unified_diff(text)

    (hunk,) = diff.hunks
    assert (hunk.old_count, hunk.new_count) == (1, 1)
    assert [line.text for line in hunk.lines] == ["old", "new"]


def test_parse_treats_an_empty_line_as_blank_context() -> None:
    diff = parse_unified_diff("@@ -1,2 +1,2 @@\n\n-x\n+y\n")

    first = diff.hunks[0].lines[0]
    assert (first.kind, first.text, first.old_lineno, first.new_lineno) == ("context", "", 1, 1)


def test_parse_skips_unrecognised_line_prefixes() -> None:
    diff = parse_unified_diff("@@ -1 +1 @@\n\\ No newline at end of file\n-old\n+new\n")

    assert [line.kind for line in diff.hunks[0].lines] == ["remove", "add"]


def test_parse_with_no_hunks_is_empty() -> None:
    assert parse_unified_diff("not a diff at all").is_empty
