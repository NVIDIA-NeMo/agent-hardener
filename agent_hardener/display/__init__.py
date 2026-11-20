# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System-owned CLI display for agent outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.text import Text

from agent_hardener.display.context import DisplayContext
from agent_hardener.display.registry import resolve_renderer
from agent_hardener.display.renderers.generic import GenericRenderer

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DisplayContext",
    "render_attack_summary",
    "render_defender_summary",
    "render_outputs_verbose",
    "render_validator_summary",
    "resolve_renderer",
]


def render_outputs_verbose(outputs: Iterable[dict[str, Any]], ctx: DisplayContext) -> list[Any]:
    """Render verbose sections for a list of agent outputs."""
    if not ctx.verbose:
        return []
    parts: list[Any] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        renderer = resolve_renderer(output)
        detail = renderer.verbose(output, ctx)
        if detail is not None:
            parts.append(detail)
    return parts


def render_attack_summary(payload: dict[str, Any], ctx: DisplayContext) -> tuple[list[Text], list[Any]]:
    """Build the attack phase headline (``hits / attempts``, plus each failed attacker) and verbose renderables."""
    attacks = [a for a in (payload.get("attacks") or []) if isinstance(a, dict)]
    hits = sum(len(a.get("records") or []) for a in attacks)
    attempts = _attempt_total(attacks)
    failed = [a for a in attacks if not a.get("ok", True)]
    if attempts == 0 and not failed:
        # The attacker ran but sent nothing (e.g. garak found no tools): not a clean bill of health.
        line = Text("! ATTACK complete — ").append("no attempts were made; see the attacker log", style="yellow")
    else:
        line = Text(f"{'✗' if failed else '✓'} ATTACK complete — ")
        landed = (
            f"{hits} / {attempts} attempts landed" if attempts is not None else f"{hits} hit{'' if hits == 1 else 's'}"
        )
        line.append(landed, style="bold red" if hits else "green")
    head = [line]
    for attack in failed:
        reason = attack.get("error") or attack.get("summary") or "failed"
        head.append(Text(f"  {attack.get('agent_name', '<unknown>')}: ").append(str(reason), style="red"))
    verbose_parts: list[Any] = []
    if ctx.verbose and attacks:
        from agent_hardener.display.renderers.garak_attacker import (  # noqa: PLC0415
            GarakAttackerRenderer,
            attack_transcript_for_attacks,
            attacker_table_for_attacks,
        )

        verbose_parts.append(attacker_table_for_attacks(attacks))
        transcript = attack_transcript_for_attacks(attacks)
        if transcript is not None:
            verbose_parts.append(transcript)
        for attack in attacks:
            renderer = resolve_renderer(attack)
            # The attacker table + transcript above already cover every Garak attack, so skip the
            # per-attack GarakAttackerRenderer.verbose() to avoid rendering the same table twice.
            if isinstance(renderer, GarakAttackerRenderer):
                continue
            detail = renderer.verbose(attack, ctx)
            if detail is not None:
                verbose_parts.append(detail)
    return head, verbose_parts


def _attempt_total(attacks: list[dict[str, Any]]) -> int | None:
    """Attempts behind the hits, summed over attackers — ``None`` unless every attacker reported its count.

    A hit count without its denominator reads as a rate it is not; when any attacker cannot say how many
    attempts it made, the headline falls back to the bare hit count rather than a wrong total.
    """
    counts = [(a.get("metadata") or {}).get("attempts") for a in attacks]
    if not counts or not all(isinstance(c, int) for c in counts):
        return None
    return sum(counts)


def _failed_because(error: object) -> str:
    """``failed`` plus the first line of the agent's error, when it gave one."""
    reason = str(error or "").strip().splitlines()
    return f"failed: {reason[0]}" if reason else "failed"


def render_defender_summary(payload: dict[str, Any], ctx: DisplayContext) -> tuple[list[Text], list[Any]]:
    """Build defender headline lines and optional verbose renderables."""
    defenders = [d for d in (payload.get("defenders") or []) if isinstance(d, dict)]
    lines: list[Text] = []
    for defender in defenders:
        patches = len(defender.get("policy_patches") or [])
        ok = defender.get("ok", True)
        line = Text(f"  {defender.get('agent_name', '<unknown>')}: ")
        if not ok:
            line.append(_failed_because(defender.get("error_message") or defender.get("error")), style="red")
        else:
            line.append(f"{patches} patch{'' if patches == 1 else 'es'}", style="green" if patches else "dim")
        lines.append(line)
    verbose_parts: list[Any] = []
    if ctx.verbose and defenders:
        from agent_hardener.display.renderers.openshell_policy import render_defender_detail  # noqa: PLC0415

        verbose_parts.append(render_defender_detail(defenders, payload.get("policy_patches") or []))
    return lines, verbose_parts


def render_validator_summary(payload: dict[str, Any], ctx: DisplayContext) -> tuple[list[Text], list[Any]]:
    """Build validator headline lines and optional verbose renderables."""
    validators = [v for v in (payload.get("validators") or []) if isinstance(v, dict)]
    lines: list[Text] = []
    for validator in validators:
        ok = validator.get("ok", True)
        line = Text(f"  {validator.get('agent_name', '<unknown>')}: ")
        if not ok:
            line.append(_failed_because(validator.get("error")), style="red")
        else:
            line.append(str(validator.get("summary") or "ok"), style="green")
        lines.append(line)
    verbose_parts = render_outputs_verbose(validators, ctx)
    if ctx.verbose and validators and not verbose_parts:
        generic = GenericRenderer()
        for validator in validators:
            detail = generic.verbose(validator, ctx)
            if detail is not None:
                verbose_parts.append(detail)
    return lines, verbose_parts
