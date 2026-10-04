# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A path trie over benign request paths, used to find the shallowest glob that carves the

attack path out without touching any benign path — the deterministic core of the REST branch of
``add_deny_rule``.
"""

from __future__ import annotations


class _TrieNode:
    __slots__ = ("children", "owners")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.owners: set[str] = set()


def _segments(path: str) -> list[str]:
    return [seg for seg in path.strip("/").split("/") if seg != ""]


class PathTrie:
    """Segment trie keyed by ``owner`` (only ``"benign"`` is used in practice)."""

    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, path: str, owner: str) -> None:
        node = self._root
        for seg in _segments(path):
            node = node.children.setdefault(seg, _TrieNode())
        node.owners.add(owner)

    def subtree_owners(self, prefix: str) -> set[str]:
        """Union of owners at or below ``prefix``; empty if ``prefix`` isn't in the trie at all."""
        node = self._root
        for seg in _segments(prefix):
            child = node.children.get(seg)
            if child is None:
                return set()
            node = child
        return self._collect(node)

    def shallowest_clean_prefix(self, attack_path: str) -> str | None:
        """The shortest prefix of ``attack_path`` whose subtree contains no benign path.

        Subtree size only shrinks as the prefix gets longer (more specific), so "contains a
        benign path" is monotone: once a prefix is clean, every longer prefix of the same
        attack path is clean too. This walks from the shortest prefix (one segment) toward the
        full path and returns the first clean one — the most general (shallowest), and thus most
        bypass-resistant, separator. Returns ``None`` if even the full attack path collides —
        either it exactly equals a benign path, or a benign path lives inside its own subtree (a
        deny glob rooted at the attack path would then also catch that benign traffic) — that
        axis is provably unusable, not merely uncertain.
        """
        segments = _segments(attack_path)
        if not segments:
            return None
        node = self._root
        chain = [node]
        for seg in segments:
            node = node.children.get(seg) or _TrieNode()
            chain.append(node)
        for i in range(1, len(chain)):
            if "benign" not in self._collect(chain[i]):
                return "/".join(segments[:i])
        return None

    def _collect(self, node: _TrieNode) -> set[str]:
        owners = set(node.owners)
        for child in node.children.values():
            owners |= self._collect(child)
        return owners
