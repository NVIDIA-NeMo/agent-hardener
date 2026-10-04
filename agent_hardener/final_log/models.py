# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models and constants shared by the final-log renderers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SCANNER_LABELS = {
    "indirect": "indirect",
    "agent_breaker": "agent_breaker",
    "other": "other",
}
# The two real garak scanners, and the same plus the "other" catch-all — iterate these instead of
# re-spelling the tuple at each call site.
SCANNERS = tuple(scanner for scanner in SCANNER_LABELS if scanner != "other")
SCANNERS_WITH_OTHER = tuple(SCANNER_LABELS)
MAX_EXAMPLES_PER_SCANNER = 5
MAX_POLICY_ITEMS = 8


@dataclass
class HitExample:
    """One compact hitlog example for the final log."""

    source_path: Path
    probe: str
    detector: str
    prompt: str
    output: str
    score: Any


@dataclass
class ScanStats:
    """Aggregated scanner counts from reports, hitlogs, and Agent Hardener records."""

    scanner: str
    agent_hardener_hits: int = 0
    hitlog_hits: int = 0
    garak_successes: int = 0
    garak_failures: int = 0
    garak_total: int = 0
    has_garak_eval: bool = False
    eval_rows: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    yaml_names: set[str] = field(default_factory=set)
    job_ids: set[str] = field(default_factory=set)
    files: set[Path] = field(default_factory=set)
    probes: Counter[str] = field(default_factory=Counter)
    detectors: Counter[str] = field(default_factory=Counter)
    examples: list[HitExample] = field(default_factory=list)

    @property
    def displayed_successes(self) -> int:
        """Return the best available successful exploit count."""
        if self.has_garak_eval:
            return self.garak_successes
        return max(self.agent_hardener_hits, self.hitlog_hits)

    @property
    def displayed_failures(self) -> int | None:
        """Return failed/no-hit checks when an eval report is available."""
        if self.has_garak_eval:
            return self.garak_failures
        return None

    @property
    def displayed_total(self) -> int | None:
        """Return total checks when an eval report is available."""
        if self.has_garak_eval:
            return self.garak_total
        return None


@dataclass
class FinalLogData:
    """The parsed inputs both final-log renderers (plain text and rich) consume.

    Built once by :func:`_collect_final_log_data` so the two presentations share one source of truth
    and never re-implement the parsing/aggregation.
    """

    report_dicts: list[dict[str, Any]]
    stats: dict[str, ScanStats]
    garak_files: list[Path]


@dataclass
class AttemptTranscript:
    """One attack attempt's prompt + victim response (hit or not), for the verbose transcript."""

    scanner: str
    probe: str
    prompt: str
    response: str
    hit: bool
