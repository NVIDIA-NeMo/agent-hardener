# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal answer provider for the synth interview (the CLI driver's transport).

The synth graph interrupts with a batch of questions; this prompts the operator for each via ``questionary``
(arrow-key option picker + free-text "Other") and returns the answers. The CLI/orchestrator drivers pass this
to ``drive_synth``; the HTTP service supplies its own provider. Works on plain dicts so it stays decoupled from
the synth package.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import questionary
from rich.console import Console
from rich.padding import Padding

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


def _recommended(options: list[dict[str, Any]]) -> str:
    """The option flagged ``recommended``, or the first when none is."""
    return next((o["description"] for o in options if o.get("recommended")), options[0]["description"])


def _prompt_header(question: str) -> None:
    """Print the interviewer rule followed by the bold question."""
    _console.print()
    _console.rule("[bold blue]Interviewer[/bold blue]", style="blue")
    _console.print(Padding(f"[bold]{question}[/bold]", (1, 2)))


def _ask_with_options(question: str, options: list[dict[str, Any]]) -> str | None:
    """Blocking arrow-key option selection with a free-text fallback."""
    recommended = _recommended(options)
    _prompt_header(question)
    choices = [
        questionary.Choice(
            title=f"{'★  ' if o.get('recommended') else '   '}{o['description']}", value=o["description"]
        )
        for o in options
    ]
    choices.append(questionary.Choice(title="   Other — enter your own answer", value=_SENTINEL_OTHER))
    selected = questionary.select(
        "", choices=choices, default=recommended, style=_STYLE, instruction="(↑↓ move  ·  Enter select)"
    ).ask()
    if selected is None:  # Ctrl+C
        return None
    if selected == _SENTINEL_OTHER:
        _console.print()
        return cast("str | None", questionary.text("  Your answer:", style=_STYLE).ask() or None)
    return cast("str | None", selected)


def _ask_batch(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prompt each question sequentially; return ``{gap, question, answer}`` for every answered one."""
    collected: list[dict[str, Any]] = []
    for item in questions:
        question = item.get("question")
        if not question:
            continue
        if item.get("options"):
            answer = _ask_with_options(question, item["options"])
        else:
            _prompt_header(question)
            answer = input("  > ").strip() or None
        if answer is None:  # Ctrl+C mid-batch — stop with what we have
            break
        if answer:
            collected.append({"gap": item.get("gap", ""), "question": question, "answer": answer})
    return collected


async def tty_answer_provider(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Async provider for ``drive_synth``: collect interview answers from the terminal off the event loop."""
    return await asyncio.to_thread(_ask_batch, questions)


def _default_answer(item: dict[str, Any]) -> dict[str, Any] | None:
    """The recommended option (or the first, if none is flagged); ``None`` for free-text questions."""
    options = item.get("options")
    if not options:  # no suggestion to accept — leave the gap unresolved
        return None
    return {"gap": item.get("gap", ""), "question": item.get("question", ""), "answer": _recommended(options)}


async def auto_answer_provider(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-interactive provider: accept each question's recommended default without prompting (``--yes``)."""
    answers: list[dict[str, Any]] = []
    for item in questions:
        picked = _default_answer(item)
        if picked is None:
            continue
        _console.print(f"[dim]auto-accept[/dim] {picked['question']} [cyan]→[/cyan] {picked['answer']}")
        answers.append(picked)
    return answers
