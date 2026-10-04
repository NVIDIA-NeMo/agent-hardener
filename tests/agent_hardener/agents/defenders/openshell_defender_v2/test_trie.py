# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table-driven tests for PathTrie.shallowest_clean_prefix — add_deny_rule's REST-branch algorithm."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.analysis.trie import PathTrie

pytestmark = pytest.mark.unit


def test_returns_shallowest_clean_prefix_when_attack_path_is_isolated() -> None:
    trie = PathTrie()
    trie.insert("/repos/foo/bar/issues", "benign")
    assert trie.shallowest_clean_prefix("/admin/delete") == "admin"


def test_returns_none_when_benign_path_nests_under_attack() -> None:
    # A benign path lives inside the attack path's own subtree, so any glob at or shallower than
    # the attack path itself would also deny that benign traffic — the axis is unusable.
    trie = PathTrie()
    trie.insert("/repos/foo/bar/issues", "benign")
    assert trie.shallowest_clean_prefix("/repos/foo/bar") is None


def test_returns_none_when_attack_path_exactly_matches_benign() -> None:
    trie = PathTrie()
    trie.insert("/repos/foo/bar", "benign")
    assert trie.shallowest_clean_prefix("/repos/foo/bar") is None


def test_finds_deeper_clean_prefix_when_shallow_segment_collides() -> None:
    trie = PathTrie()
    trie.insert("/repos/foo/bar/issues", "benign")
    # "repos" collides (benign lives under it), but "repos/foo/bar/delete" doesn't.
    assert trie.shallowest_clean_prefix("/repos/foo/bar/delete") == "repos/foo/bar/delete"


def test_empty_path_returns_none() -> None:
    trie = PathTrie()
    assert trie.shallowest_clean_prefix("") is None


def test_subtree_owners_empty_for_unknown_prefix() -> None:
    trie = PathTrie()
    trie.insert("/a/b", "benign")
    assert trie.subtree_owners("/x/y") == set()
    assert trie.subtree_owners("/a") == {"benign"}
