# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for excerpting replayed victim output.

``excerpt`` only reads ``excerpt_chars`` and ``redact_outputs`` off the config, so these build a
stub rather than a full :class:`ReplayConfig`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_hardener.agents.validators.garak_replay import redaction

pytestmark = pytest.mark.unit


def _cfg(excerpt_chars: int = 500, *, redact_outputs: bool = True) -> Any:
    return cast("Any", SimpleNamespace(excerpt_chars=excerpt_chars, redact_outputs=redact_outputs))


def test_excerpt_still_fills_its_budget_when_redaction_shrinks_the_text() -> None:
    """What the margin is for: a redacted credential frees budget, so later text must still land.

    Redacting replaces the credential with a short marker, so the excerpt reaches further into the
    text than the raw 500-char offset. Without head room above ``excerpt_chars`` the window would be
    cut before that content and the excerpt would come back near-empty.
    """
    text = "GITHUB_TOKEN=" + "x" * 400 + " tail-that-must-survive " + "y" * 5000

    result = redaction.excerpt(text, _cfg())

    assert "x" * 400 not in result  # the credential value is gone
    assert "tail-that-must-survive" in result
    assert len(result) == 500 + len("...<truncated>")


def test_excerpt_redacts_a_secret_inside_the_window() -> None:
    text = "leaked sk-live-secret-value here"

    assert "sk-live-secret-value" not in redaction.excerpt(text, _cfg())


def test_excerpt_only_redacts_a_bounded_window_of_a_huge_response() -> None:
    """A hostile victim can answer with any volume; the regex work must not scale with it."""
    huge = "x" * 5_000_000

    result = redaction.excerpt(huge, _cfg())

    assert len(result) == 500 + len("...<truncated>")


def test_excerpt_returns_short_text_untouched() -> None:
    assert redaction.excerpt("all clear", _cfg()) == "all clear"


def test_excerpt_still_truncates_when_redaction_is_disabled() -> None:
    text = "ghp_visible" + "b" * 600

    result = redaction.excerpt(text, _cfg(redact_outputs=False))

    assert result.startswith("ghp_visible")
    assert result.endswith("...<truncated>")
