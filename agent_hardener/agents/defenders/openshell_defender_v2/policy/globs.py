# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell glob semantics for policy path matching.

``*`` and ``**`` both match zero or more characters and may cross ``/`` (OpenShell doesn't
distinguish single-segment vs. multi-segment globs the way shell globbing does); ``?`` matches
exactly one character; bracket classes (``[0-9]``, ``[!0]``) are supported.
"""

from __future__ import annotations

import fnmatch
import re


def _to_regex(pattern: str) -> re.Pattern[str]:
    # translate() handles *, ?, and [..] the way we want; OpenShell's ** is equivalent to * here
    # since neither is anchored to path segments.
    return re.compile(fnmatch.translate(pattern.replace("**", "*")))


def glob_matches(pattern: str, path: str) -> bool:
    """Whether ``path`` matches an OpenShell-style ``pattern``."""
    return bool(_to_regex(pattern).match(path))


def glob_is_subset(narrow: str, wide: str) -> bool:
    """Whether every concrete path matched by ``narrow`` is also matched by ``wide``.

    Exact for the literal case (``narrow == wide``) and for the common prefix-glob case
    (``narrow`` is ``wide`` with more literal path segments appended before its own wildcard).
    Not a full glob-containment solver — used only as a best-effort lint hint, never a hard gate
    (``lint.assert_contraction`` uses concrete before/after path sets instead).
    """
    if narrow == wide:
        return True
    wide_prefix = wide.split("*", 1)[0]
    narrow_prefix = narrow.split("*", 1)[0]
    return narrow_prefix.startswith(wide_prefix) and glob_matches(wide, narrow_prefix + "x")


def prefix_to_glob(prefix: str) -> str:
    """Turn a path prefix (e.g. from ``PathTrie.shallowest_clean_prefix``) into a policy glob."""
    prefix = prefix.rstrip("/")
    return f"{prefix}/**" if prefix else "**"
