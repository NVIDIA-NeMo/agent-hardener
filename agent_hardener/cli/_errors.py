# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI error boundary: turn any unhandled failure into a clean exit + a structured dump.

Command bodies wrap their work in :func:`cli_error_boundary`. It keeps the runtime taxonomy
(:mod:`agent_hardener.errors`) free of a typer dependency while giving every command one place to
serialize the failure (for the calling process) and print a clean, actionable message (for an
interactive operator) instead of leaking a raw traceback.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import typer

from agent_hardener.display.console import fail, info
from agent_hardener.errors import AgentHardenerError, emit_run_error


@contextlib.contextmanager
def cli_error_boundary() -> Iterator[None]:
    """Serialize any unhandled failure in a CLI command body, then exit non-zero with a clean message.

    ``typer.Exit``/``typer.Abort`` are control flow (the normal exit path, including already-reported
    ``fail()`` cases) and pass through untouched. An :class:`AgentHardenerError` is dumped via
    :func:`emit_run_error`, its message + remediation are printed, and the command exits 1. Any other
    exception is dumped as ``unexpected`` (its traceback goes to the error file, not the console) and
    the command exits 1 — no raw traceback reaches the operator.
    """
    try:
        yield
    except (typer.Exit, typer.Abort):
        raise
    except AgentHardenerError as exc:
        emit_run_error(exc)
        fail(str(exc))
        if exc.remediation:
            info(f"  → {exc.remediation}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        emit_run_error(exc)
        fail(f"Unexpected error: {exc}")
        raise typer.Exit(1) from exc
