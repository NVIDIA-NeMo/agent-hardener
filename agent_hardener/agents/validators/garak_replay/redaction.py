# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret redaction and excerpting for replayed victim outputs.

Config-dependent but monkeypatch-free leaf helpers. ``ReplayConfig`` is used only in annotations, so it is
imported under ``TYPE_CHECKING`` — no runtime import cycle with the validator module.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hardener.agents.validators.garak_replay.config import ReplayConfig

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)([\"']?)\b([A-Z][A-Z0-9_]*(?:API_)?(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*)\1"
            r"\s*[:=]\s*[\"']?([^\s,\"'`}\]]+)"
        ),
        r"\2=[redacted]",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]+\b"), "[redacted-github-token]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]+\b"), "[redacted-github-token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]+\b"), "[redacted-gitlab-token]"),
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]+\b"), "[redacted-nvidia-key]"),
    (re.compile(r"\btvly-[A-Za-z0-9_\-]+\b"), "[redacted-tavily-key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]+\b"), "[redacted-api-key]"),
    (re.compile(r"\bhf_[A-Za-z0-9_]+\b"), "[redacted-huggingface-token]"),
    (re.compile(r"/Users/[^ \n\r\t,\"'`)\]}]+"), "/Users/[redacted]"),
)


def redact(text: str, config: ReplayConfig) -> str:
    if not config.redact_outputs:
        return text
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


#: Head room redacted above ``excerpt_chars``. Redacting replaces a long credential with a short
#: marker, so the excerpt budget reaches further into the text than the raw offset suggests; without
#: the head room the excerpt would be starved of the content the old whole-string redaction showed.
#: Sized well past the plausible amount of that shrink — a response dense enough to redact its way
#: through this much is not one whose tail is worth reconstructing — and still a fixed bound, so the
#: regex work stays flat no matter how much the victim returns.
_REDACTION_MARGIN_CHARS = 65_536


def excerpt(text: str, config: ReplayConfig) -> str:
    """Redact ``text`` and cut it down to ``config.excerpt_chars``.

    Only a bounded window is redacted. The replayed text is the victim's own response, and a
    compromised or simply broken victim can return a body of any size; redacting all of it would run
    every pattern in :data:`SECRET_PATTERNS` across the whole string just to discard the result here.
    """
    window_limit = config.excerpt_chars + _REDACTION_MARGIN_CHARS
    redacted = redact(text[:window_limit], config)
    if len(text) <= window_limit and len(redacted) <= config.excerpt_chars:
        return redacted
    return f"{redacted[: config.excerpt_chars]}...<truncated>"


def redacted_errors(errors: list[dict[str, str]], config: ReplayConfig) -> list[dict[str, str]]:
    return [
        {
            "target": error.get("target", ""),
            "error": excerpt(error.get("error", ""), config),
        }
        for error in errors
    ]
