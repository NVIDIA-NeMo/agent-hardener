# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw OpenShell command execution — the collaborator :class:`OpenShellLifecycle` composes.

This owns *how a single command runs* (subprocess plumbing, output capture, per-command logging),
not *which* commands run or how failures are retried/reconciled — that orchestration stays in
:mod:`agent_hardener.openshell.lifecycle`. Under a real :class:`SubprocessCommandRunner` the executor
captures output via a file/tempfile to avoid inherited-pipe hangs; under an injected test runner it
just forwards to it, so command assertions stay deterministic.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

from agent_hardener.openshell.config import OpenShellCommandResult, SubprocessCommandRunner
from agent_hardener.openshell.support import safe_path_token

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.openshell.config import CommandRunner, OpenShellConfig

logger = logging.getLogger("agent_hardener.session")


class OpenShellCommandExecutor:
    """Runs individual OpenShell commands and normalizes them to :class:`OpenShellCommandResult`."""

    def __init__(self, config: OpenShellConfig, runner: CommandRunner, log_dir: Path | None) -> None:
        self.config = config
        self.runner = runner
        self.log_dir = log_dir

    def run(self, args: list[str]) -> OpenShellCommandResult:
        """Run a command, capturing output (via a tempfile under the subprocess runner)."""
        command = [self.config.openshell_bin, *args]
        if isinstance(self.runner, SubprocessCommandRunner):
            completed, captured, duration, error = self._run_to_file(command, log_file=None)
            if error is not None:
                logger.info(
                    "openshell command failed op=%s duration_seconds=%.3f error=%s",
                    _command_label(args),
                    duration,
                    error,
                )
                return OpenShellCommandResult(ok=False, command=command, output=str(error))
            logger.info(
                "openshell command completed op=%s ok=%s exit_code=%s duration_seconds=%.3f",
                _command_label(args),
                completed.returncode == 0,
                completed.returncode,
                duration,
            )
            return OpenShellCommandResult(ok=completed.returncode == 0, command=command, output=captured)
        try:
            completed = self.runner.run(command, cwd=self.config.cwd)
        except OSError as exc:
            return OpenShellCommandResult(ok=False, command=command, output=str(exc))
        return OpenShellCommandResult.from_completed(command, completed)

    def run_logged(self, args: list[str]) -> OpenShellCommandResult:
        """Run a command to completion while writing full output to a stable log file."""
        command = [self.config.openshell_bin, *args]
        if isinstance(self.runner, SubprocessCommandRunner):
            log_path = self.log_path_for(args)
            completed, _captured, duration, error = self._run_to_file(command, log_file=log_path)
            if error is not None:
                logger.info(
                    "openshell logged command failed op=%s duration_seconds=%.3f error=%s",
                    _command_label(args),
                    duration,
                    error,
                )
                return OpenShellCommandResult(ok=False, command=command, output=str(error))
            logger.info(
                "openshell logged command completed op=%s ok=%s exit_code=%s duration_seconds=%.3f log=%s",
                _command_label(args),
                completed.returncode == 0,
                completed.returncode,
                duration,
                log_path,
            )
            tail = self._tail_file(log_path)
            status = "command completed" if completed.returncode == 0 else "command failed"
            output = f"{status} exit_code={completed.returncode}; duration_seconds={duration:.3f}; log={log_path}"
            if tail:
                output = f"{output}\n{tail}"
            return OpenShellCommandResult(ok=completed.returncode == 0, command=command, output=output)

        completed = self.runner.run(command, cwd=self.config.cwd)
        return OpenShellCommandResult.from_completed(command, completed)

    def log_path_for(self, args: list[str]) -> Path:
        """Return a log file location for background/logged OpenShell commands under the run's log dir."""
        from pathlib import Path  # noqa: PLC0415 - keep the module import surface minimal.

        log_dir = self.log_dir or (self.config.cwd or Path.cwd()) / ".agent-hardener" / "openshell-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        command_name = args[0] if args else "command"
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        return log_dir / f"{safe_path_token(self.config.sandbox)}-{safe_path_token(command_name)}-{timestamp}.log"

    def _run_to_file(
        self, command: list[str], *, log_file: Path | None
    ) -> tuple[subprocess.CompletedProcess[str] | None, str, float, OSError | None]:
        """Run ``command`` capturing stdout+stderr to a stream, avoiding inherited-pipe hangs.

        With ``log_file`` the output is written to that stable path (with a command header) and the
        returned captured text is empty (callers tail the file); without it, output is captured in a
        tempfile and returned. Returns ``(completed, captured_text, duration_seconds, error)`` where
        exactly one of ``completed`` / ``error`` is set.
        """
        started_at = time.monotonic()
        try:
            if log_file is not None:
                with log_file.open("w", encoding="utf-8") as stream:
                    stream.write(f"command: {shlex.join(command)}\n")
                    stream.flush()
                    completed = subprocess.run(  # noqa: S603 - explicit arg lists built by lifecycle callers.
                        command,
                        cwd=self.config.cwd,
                        check=False,
                        text=True,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                captured = ""
            else:
                with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stream:
                    completed = subprocess.run(  # noqa: S603 - explicit arg lists built by lifecycle callers.
                        command,
                        cwd=self.config.cwd,
                        check=False,
                        text=True,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    stream.seek(0)
                    captured = stream.read()
        except OSError as exc:
            return None, "", time.monotonic() - started_at, exc
        return completed, captured, time.monotonic() - started_at, None

    def _tail_file(self, path: Path, line_count: int = 80, max_chars: int = 12000) -> str:
        """Return the end of a log file without flooding mission reports."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"failed to read log {path}: {exc}"
        tail = "\n".join(text.splitlines()[-line_count:])
        if len(tail) > max_chars:
            tail = tail[-max_chars:]
        return tail


def _command_label(args: list[str]) -> str:
    """Return a short non-secret command label for timing logs."""
    return " ".join(args[:3]) if args else "<empty>"
