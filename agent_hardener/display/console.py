# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared terminal presentation primitives for the Agent Hardener CLI.

Presentation only: a cached :class:`rich.console.Console`, a TTY/color gate, small glyph helpers,
a line sink for streamed output, and a spinner-context factory. This module knows nothing about
reports or stats, so any command can reuse it. When output is not a real terminal (CI, piped to a
file), or ``NO_COLOR`` / ``AGENT_HARDENER_PLAIN`` is set, everything degrades to plain unstyled text.
"""

from __future__ import annotations

import contextlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Whole-word cues for line styling in ``emit``. Word boundaries avoid false hits like "already"
# (→ ready) or class names like "AuthenticationError". ``_BENIGN`` keeps lines such as "0 errors"
# or "no errors found" from being painted red.
_RED_CUE = re.compile(r"\b(failed|failure|error|errors|exception|traceback)\b", re.IGNORECASE)
_GREEN_CUE = re.compile(r"\b(ready|succeeded|success|completed)\b", re.IGNORECASE)
_BENIGN = re.compile(r"\b(0|no)\s+errors?\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Return the process-wide :class:`rich.console.Console` (cached)."""
    return Console()


def is_rich() -> bool:
    """Report whether styled output should be emitted.

    True only on a real terminal with color enabled: ``NO_COLOR`` (a cross-tool convention) and the
    explicit ``AGENT_HARDENER_PLAIN`` escape hatch both force plain text, as does a non-TTY stdout.
    """
    if os.environ.get("NO_COLOR") or os.environ.get("AGENT_HARDENER_PLAIN"):
        return False
    return get_console().is_terminal


def _print(message: str, style: str | None) -> None:
    """Print ``message`` styled when rich is active, else as plain text."""
    get_console().print(message, style=style if is_rich() else None, highlight=False)


def step(message: str) -> None:
    """Print an in-progress step line (dim cyan ``▸``)."""
    _print(f"▸ {message}", "cyan")


def ok(message: str) -> None:
    """Print a success line (green ``✓``)."""
    _print(f"✓ {message}", "green")


def fail(message: str) -> None:
    """Print a failure line (red ``✗``)."""
    _print(f"✗ {message}", "red")


def warn(message: str) -> None:
    """Print a non-fatal warning line (yellow ``!``)."""
    _print(f"! {message}", "yellow")


def info(message: str) -> None:
    """Print a neutral, unstyled line."""
    _print(message, None)


def section(title: str) -> None:
    """Print a dim section divider (preceded by a blank line) to group interactive prompts into phases."""
    _print(f"\n── {title} ──", "dim")


def banner(title: str, lines: list[str]) -> None:
    """Print an orienting banner: a bold title followed by dim body lines, then a blank line.

    Used by ``init`` to explain what Agent Hardener is and what the command will create before any
    prompt. Degrades to plain text off a TTY like everything else here.
    """
    _print(title, "bold cyan")
    for line in lines:
        _print(line, "dim")
    info("")


def _path_completer(base: Path) -> Callable[[str, int], str | None]:
    """Build a readline completer that completes filesystem paths relative to ``base``.

    Whole-token (delimiters set to whitespace) so ``../foo/bar`` completes across slashes; the typed
    directory prefix is preserved (so ``../`` stays ``../``), directories get a trailing ``/``, and
    listing failures yield no matches rather than raising into the prompt.
    """

    def complete(text: str, state: int) -> str | None:
        typed_dir, sep, prefix = text.rpartition("/")
        listing = Path(typed_dir).expanduser() if typed_dir else Path()
        if not listing.is_absolute():
            listing = base / listing
        try:
            names = sorted(p.name for p in listing.iterdir())
        except OSError:
            return None
        results: list[str] = []
        for name in names:
            if not name.startswith(prefix):
                continue
            completion = f"{typed_dir}{sep}{name}" if sep else name
            if (listing / name).is_dir():
                completion += "/"
            results.append(completion)
        return results[state] if state < len(results) else None

    return complete


@contextlib.contextmanager
def path_completion(base_dir: Path | None = None) -> Iterator[None]:
    """Enable filesystem Tab-completion for ``input()`` / ``typer.prompt`` within the block.

    Completions are relative to ``base_dir`` (default: cwd) so a prompt for a path inside the agent
    project can complete against that root. No-ops off a TTY (CI/piped) and where ``readline`` is
    unavailable (e.g. some Windows setups); restores the previous completer state on exit.
    """
    if not is_rich():
        yield
        return
    try:
        import readline  # noqa: PLC0415
    except ImportError:
        yield
        return
    base = base_dir if base_dir is not None else Path.cwd()
    prev_completer = readline.get_completer()
    prev_delims = readline.get_completer_delims()
    readline.set_completer(_path_completer(base))
    readline.set_completer_delims(" \t\n")
    # macOS ships libedit, which uses a different bind syntax than GNU readline.
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")
    try:
        yield
    finally:
        readline.set_completer(prev_completer)
        readline.set_completer_delims(prev_delims)


def emit(line: str) -> None:
    """Rich-aware ``on_output`` sink: style a streamed line by simple content cues.

    Multi-line blobs (e.g. ``result.output``) are styled **per line**, not as a whole — so one
    "failed" line inside a long block (e.g. a benign forward-teardown notice) no longer paints the
    entire block red. Each line is red on a genuine failure cue, green on a ready/success cue, and
    plain otherwise. Safe to use as a drop-in for ``print``/``typer.echo``.
    """
    for text in line.splitlines() or [line]:
        if _RED_CUE.search(text) and not _BENIGN.search(text):
            _print(text, "red")
        elif _GREEN_CUE.search(text):
            _print(text, "green")
        else:
            _print(text, None)


def status_factory(label: str) -> contextlib.AbstractContextManager[object]:
    """Return a spinner context for a long-running step, or a no-op when not rich.

    Used as ``run_mission``'s ``status_cm`` seam: ``with status_factory("Bringing up sandbox"): ...``
    shows an animated spinner on a TTY and does nothing observable otherwise.
    """
    if not is_rich():
        return contextlib.nullcontext()
    return get_console().status(f"{label}…", spinner="dots")
