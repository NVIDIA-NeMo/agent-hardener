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


# ``--debug`` switch for the whole process: set once by the CLI callback (``agent_hardener.loggers``), read by
# :func:`detail` and the console log handler. A plain module flag because it is process-wide CLI state.
_debug = False


def set_debug(enabled: bool) -> None:
    """Turn the console's debug tier on or off (``--debug``)."""
    global _debug  # noqa: PLW0603 - one process-wide CLI switch
    _debug = enabled


def is_debug() -> bool:
    """Report whether ``--debug`` is on."""
    return _debug


def _print(message: str, style: str | None, *, markup: bool = True) -> None:
    """Print ``message`` styled when rich is active, else as plain text.

    ``soft_wrap`` keeps a long line (a path, a command) whole instead of hard-wrapping it at the console
    width, which off a terminal is a fixed 80 columns and used to split paths mid-word. ``markup=False``
    is for text that did not come from us (log messages, subprocess output) and may contain ``[...]``.
    """
    get_console().print(message, style=style if is_rich() else None, highlight=False, soft_wrap=True, markup=markup)


def _with_elapsed(message: str, elapsed: float | None) -> str:
    return message if elapsed is None else f"{message} ({format_elapsed(elapsed)})"


def format_elapsed(seconds: float) -> str:
    """Render a duration the way every console line shows it: ``0.4s``, ``12.4s``, ``3m05s``, ``1h02m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# The console vocabulary: level -> (text template, style). One table, so the console, the engine's
# ``output`` events and the log file's echo of them (:func:`line_text`) can never drift apart.
_FORMATS: dict[str, tuple[str, str | None]] = {
    "section": ("── {} ──", "bold"),
    "step": ("▸ {}", "cyan"),
    "ok": ("✓ {}", "green"),
    "warn": ("! {}", "yellow"),
    "fail": ("✗ {}", "red"),
    "hint": ("  → {}", "dim"),
    "detail": ("  · {}", "dim"),
    "info": ("{}", None),
}


def line_text(level: str, message: str, elapsed: float | None = None) -> str:
    """The unstyled text of a console line at ``level`` — e.g. ``✓ Sandbox ready (12.4s)``."""
    template, _style = _FORMATS.get(level, _FORMATS["info"])
    return template.format(_with_elapsed(message, elapsed))


def _say_styled(level: str, message: str, *, elapsed: float | None = None, markup: bool = True) -> None:
    text = line_text(level, message, elapsed)
    _print(f"\n{text}" if level == "section" else text, _FORMATS.get(level, _FORMATS["info"])[1], markup=markup)


def step(message: str, *, markup: bool = True) -> None:
    """Print an in-progress step line (cyan ``▸``)."""
    _say_styled("step", message, markup=markup)


def ok(message: str, *, elapsed: float | None = None, markup: bool = True) -> None:
    """Print a success line (green ``✓``), with how long the step took when ``elapsed`` is given."""
    _say_styled("ok", message, elapsed=elapsed, markup=markup)


def fail(message: str, *, markup: bool = True) -> None:
    """Print a failure line (red ``✗``)."""
    _say_styled("fail", message, markup=markup)


def warn(message: str, *, markup: bool = True) -> None:
    """Print a non-fatal warning line (yellow ``!``)."""
    _say_styled("warn", message, markup=markup)


def hint(message: str, *, markup: bool = True) -> None:
    """Print the next action or a pointer under the line above (dim ``→``)."""
    _say_styled("hint", message, markup=markup)


def detail(message: str, *, markup: bool = True) -> None:
    """Print a supporting detail (dim ``·``) — only with ``--debug``; the log file always has it."""
    if _debug:
        _say_styled("detail", message, markup=markup)


def info(message: str, *, markup: bool = True) -> None:
    """Print a neutral, unstyled line."""
    _say_styled("info", message, markup=markup)


def literal(text: str, style: str | None = None) -> None:
    """Print ``text`` exactly as given (no markup, no glyph), optionally styled — for log records."""
    _print(text, style, markup=False)


def section(title: str, *, markup: bool = True) -> None:
    """Print a section divider (preceded by a blank line) that opens a stage of a command."""
    _say_styled("section", title, markup=markup)


def banner(title: str, lines: list[str]) -> None:
    """Print an orienting banner: a bold title followed by dim body lines, then a blank line.

    Used by ``init`` to explain what Agent Hardener is and what the command will create before any
    prompt. Degrades to plain text off a TTY like everything else here.
    """
    _print(title, "bold cyan")
    for line in lines:
        _print(line, "dim")
    info("")


def say(level: str, message: str, *, elapsed: float | None = None) -> None:
    """Print ``message`` at a named console level (see ``_FORMATS``), or ``output`` for verbatim subprocess output.

    The text is printed literally (no rich markup): engine lines carry paths and model text.
    """
    if level == "output":
        emit(message)
    elif level == "detail":
        detail(message, markup=False)
    else:
        _say_styled(level, message, elapsed=elapsed, markup=False)


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
            _print(text, "red", markup=False)
        elif _GREEN_CUE.search(text):
            _print(text, "green", markup=False)
        else:
            _print(text, None, markup=False)


# A failure excerpt keeps at most this many lines, each cut to ``_EXCERPT_LINE_CHARS`` (a launcher command
# line can run to hundreds of characters and is never the reason).
FAILURE_EXCERPT_LINES = 15
_EXCERPT_LINE_CHARS = 300
_ERROR_BOX = re.compile(r"^\s*(error\b|\u00d7)", re.IGNORECASE)
_STALL_CUE = re.compile(
    r"\b(timed out|refused|denied|unreachable|in use|not found)\b|\bno\b[^.]*\bfound\b", re.IGNORECASE
)
# OpenShell answers an existence check ("is there a sandbox/provider by this name yet?") with an error box;
# before a create that is the expected answer, not the failure.
_EXPECTED_MISS = re.compile(r"requested entity was not found", re.IGNORECASE)


def failure_excerpt(output: str, limit: int = FAILURE_EXCERPT_LINES) -> str:
    """The part of a failed command's output worth showing: its error lines, not its whole transcript.

    Keeps lines that read as errors (``Error:``, OpenShell's error box, ``failed``, ``timed out``…), the ``│``
    lines continuing an error box (OpenShell's explanation of it), and the last line (where a cause usually
    ends up). Existence-check misses are skipped. With no such line, the last ``limit`` lines. A first
    ``… N other line(s) in the log`` line says what was left out.
    """
    lines = [line for line in output.strip().splitlines() if line.strip()]
    keep: list[int] = []
    in_box = False
    for index, line in enumerate(lines):
        if _EXPECTED_MISS.search(line):
            in_box = False
            continue
        in_box = (in_box and line.strip().startswith("│")) or bool(_ERROR_BOX.match(line))
        if in_box or _STALL_CUE.search(line) or (_RED_CUE.search(line) and not _BENIGN.search(line)):
            keep.append(index)
    keep = keep if keep else list(range(len(lines)))
    if lines and keep[-1] != len(lines) - 1:
        keep.append(len(lines) - 1)
    keep = keep[-limit:]
    shown = [
        line if len(line) <= _EXCERPT_LINE_CHARS else f"{line[:_EXCERPT_LINE_CHARS]}…"
        for line in (lines[i] for i in keep)
    ]
    left_out = len(lines) - len(shown)
    return "\n".join([f"… {left_out} other line(s) in the log", *shown] if left_out else shown)


def status_factory(label: str) -> contextlib.AbstractContextManager[object]:
    """Return a spinner context for a long-running step, or a no-op when not rich.

    Used as ``run_mission``'s ``status_cm`` seam: ``with status_factory("Bringing up sandbox"): ...``
    shows an animated spinner on a TTY and does nothing observable otherwise.
    """
    if not is_rich():
        return contextlib.nullcontext()
    return get_console().status(f"{label}…", spinner="dots")
