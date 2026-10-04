# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present each question in the batch sequentially and collect answers."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.padding import Padding

from agent_hardener.events import EventType
from agent_hardener.loggers import emit_event

if TYPE_CHECKING:
    from ..schemas import InterviewerQuestion, InterviewOption
    from ..state import InterviewerState

logger = logging.getLogger(__name__)

_SENTINEL_OTHER = "__OTHER__"
_console = Console()

_STYLE = questionary.Style(
    [
        ("qmark", "fg:ansicyan bold"),
        ("question", "bold"),
        ("answer", "fg:ansicyan bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansicyan"),
        ("instruction", "fg:ansibrightblack italic"),
    ]
)


def _sync_ask_with_options(question: str, options: list[InterviewOption]) -> str | None:
    """Blocking stdin interaction with arrow-key option selection."""
    recommended_desc = next((o.description for o in options if o.recommended), options[0].description)

    _console.print()
    _console.rule("[bold blue]Interviewer[/bold blue]", style="blue")
    _console.print(Padding(f"[bold]{question}[/bold]", (1, 2)))

    choices = [
        questionary.Choice(
            title=f"{'★  ' if o.recommended else '   '}{o.description}",
            value=o.description,
        )
        for o in options
    ]
    choices.append(questionary.Choice(title="   Other — enter your own answer", value=_SENTINEL_OTHER))

    selected = questionary.select(
        "",
        choices=choices,
        default=recommended_desc,
        style=_STYLE,
        instruction="(↑↓ move  ·  Enter select)",
    ).ask()

    if selected is None:  # Ctrl+C
        return None
    if selected == _SENTINEL_OTHER:
        _console.print()
        custom = questionary.text("  Your answer:", style=_STYLE).ask()
        return custom or None
    return selected


def _sync_ask_batch(questions: list[InterviewerQuestion]) -> list[tuple[str, str, str]]:
    """Present each question in the batch sequentially, return all collected answers."""
    collected: list[tuple[str, str, str]] = []
    for item in questions:
        if item.question is None:
            continue
        if item.options:
            answer = _sync_ask_with_options(item.question, item.options)
        else:
            _console.print()
            _console.rule("[bold blue]Interviewer[/bold blue]", style="blue")
            _console.print(Padding(f"[bold]{item.question}[/bold]", (1, 2)))
            answer = input("  > ").strip() or None
        if answer is None:
            # Ctrl+C mid-batch — stop and return what we have
            break
        if answer:
            collected.append((item.gap, item.question, answer))
    return collected


async def ask(state: InterviewerState) -> dict[str, object]:
    """Present all questions in the batch and collect answers.

    Questions are shown one at a time with arrow-key option pickers. All
    answers are collected before returning, so the parent re-synthesizes once
    for the whole batch rather than once per question.

    Skips interaction entirely when ``state.interactive`` is False, when the
    batch is empty, or when stdin is not a TTY.
    """
    active = [q for q in state.composed_questions if q.question is not None]
    if not state.interactive:
        return {"source_note": "interviewer skipped (non-interactive)"}
    if not active:
        return {"source_note": "interviewer skipped (no questions to ask)"}
    if not sys.stdin.isatty():
        return {
            "source_note": "interviewer skipped (no TTY)",
            "errors": ["interviewer: interactive=True but stdin is not a TTY"],
        }

    # Bracket the prompt with structured events so any live display (the synth spinner) can suspend
    # itself while questionary owns the terminal and resume after. emit() runs subscribers
    # synchronously, so the spinner is stopped before the prompt draws; the finally guarantees the
    # resume even on Ctrl+C / EOF.
    emit_event(EventType.INTERVIEW_STARTED, {})
    try:
        collected = await asyncio.to_thread(_sync_ask_batch, state.composed_questions)
    except (EOFError, KeyboardInterrupt) as exc:
        logger.info("interview aborted: %s", exc.__class__.__name__)
        return {"source_note": f"interview aborted ({exc.__class__.__name__})"}
    finally:
        emit_event(EventType.INTERVIEW_COMPLETED, {})

    if not collected:
        return {"source_note": "interviewer: no answers collected"}

    return {
        "collected_answers": collected,
        "source_note": f"collected {len(collected)} answer(s)",
    }
