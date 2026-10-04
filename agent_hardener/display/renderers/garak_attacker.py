# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Garak AgentBreaker attacker display renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.table import Table
from rich.text import Text

from agent_hardener.final_log import (
    SCANNER_LABELS,
    SCANNERS_WITH_OTHER,
    AttemptTranscript,
    ScanStats,
    collect_attack_stats,
    collect_attempt_transcripts,
    counter_keys,
    scan_has_data,
    short_text,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.display.context import DisplayContext

_ATTACKER_STYLE = "yellow"
_VICTIM_STYLE = "cyan"


def _exchange(
    index: int,
    header: str,
    prompt: str,
    response: str,
    *,
    hit: bool | None = None,
    status: str = "",
    sender_label: str = "⚔ attacker (red-team)",
    note: str = "",
) -> Text:
    """One colored sender→victim exchange: sender (yellow), 🤖 victim agent (cyan)."""
    if status:
        is_bad = status in ("NOT_BLOCKED", "REFUSED", "ERROR")
        badge = f"  [{status}]"
    elif hit is not None:
        is_bad = hit
        badge = "  [HIT]" if hit else "  [miss]"
    else:
        is_bad = False
        badge = ""
    text = Text()
    text.append(
        f"— #{index}  {header}{badge} —\n",
        style="bold red" if is_bad else "bold green" if status and not is_bad else "bold",
    )
    text.append(f"  {sender_label}: ", style=f"bold {_ATTACKER_STYLE}")
    text.append(f"{prompt or '<none>'}\n", style=_ATTACKER_STYLE)
    text.append("  🤖 victim agent:        ", style=f"bold {_VICTIM_STYLE}")
    text.append(f"{response or '<no response captured>'}\n", style=_VICTIM_STYLE)
    if note:
        text.append(f"  · {short_text(note, 200)}\n", style="dim")
    return text


def _transcript_conversation(transcripts: list[AttemptTranscript]) -> Group:
    """Chat-style attacker↔victim conversation, one colored exchange per attempt."""
    return Group(*(_exchange(i, t.probe, t.prompt, t.response, hit=t.hit) for i, t in enumerate(transcripts, start=1)))


def _attacker_table(stats: dict[str, ScanStats]) -> Table:
    """Per-scanner attacker-results table (red when exploits got through)."""
    table = Table(title="Attacker results", title_justify="left")
    table.add_column("Scanner")
    table.add_column("Successful hits", justify="right")
    table.add_column("Probes", overflow="fold")
    for scanner in SCANNERS_WITH_OTHER:
        scan = stats[scanner]
        if scanner == "other" and not scan_has_data(scan):
            continue
        hits = scan.displayed_successes
        table.add_row(
            SCANNER_LABELS[scanner],
            Text(str(hits), style="bold red" if hits else "green"),
            counter_keys(scan.probes) if scan.probes else "",
        )
    return table


def _declared_artifact_paths(attacks: list[dict[str, Any]]) -> list[Path]:
    """Extract Garak artifact paths declared by agents via the artifacts field."""
    from pathlib import Path  # noqa: PLC0415

    paths = []
    for attack in attacks:
        for artifact in attack.get("artifacts") or []:
            if artifact.get("type") in {"garak_report", "garak_hitlog"} and artifact.get("path"):
                paths.append(Path(artifact["path"]))
    return paths


def attacker_table_for_attacks(attacks: list[dict[str, Any]]) -> Table:
    """Build the attacker-results table from raw attack dicts (for live mid-run rendering)."""
    return _attacker_table(collect_attack_stats([{"attacks": attacks}]))


def attack_transcript_for_attacks(attacks: list[dict[str, Any]]) -> Group | None:
    """Chat-style attacker↔victim transcript for these attacks (None if no Garak report.jsonl)."""
    paths = _declared_artifact_paths(attacks)
    transcripts = collect_attempt_transcripts(paths)
    return _transcript_conversation(transcripts) if transcripts else None


class GarakAttackerRenderer:
    """Render Garak attacker tables and transcripts."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Any | None:
        table = attacker_table_for_attacks([output])
        transcript = attack_transcript_for_attacks([output])
        if transcript is None:
            return table
        return Group(table, transcript)
