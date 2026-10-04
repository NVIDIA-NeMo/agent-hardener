# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port-forward management for the OpenShell lifecycle (mixed into :class:`OpenShellLifecycle`).

Start/stop the configured forward — keeping the working foreground forward alive from our own
detached process under the real subprocess runner — and track its pid. Composed as a mixin because
these methods share the lifecycle's runner, command executor, ``config``, and health polling.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent_hardener.openshell.command import _command_label
from agent_hardener.openshell.config import OpenShellCommandResult, SubprocessCommandRunner
from agent_hardener.openshell.support import _poll_until, combine_results, safe_path_token

if TYPE_CHECKING:
    from agent_hardener.openshell.command import OpenShellCommandExecutor
    from agent_hardener.openshell.config import CommandRunner, OpenShellConfig

logger = logging.getLogger("agent_hardener.session")


class ForwardMixin:
    """Start/stop the configured port-forward and track its detached process."""

    # Provided by OpenShellLifecycle (the concrete class this mixin is composed into).
    config: OpenShellConfig
    runner: CommandRunner
    log_dir: Path | None
    _exec: OpenShellCommandExecutor

    def start_forward(self) -> OpenShellCommandResult:
        """Start the configured forward.

        OpenShell's built-in --background mode can report success while the
        helper process immediately dies on some gateways. For the real
        subprocess runner, keep the working foreground forward alive from our
        own detached process and verify it through the victim health endpoint.
        Tests and custom runners keep using the OpenShell-managed background
        path so command assertions remain deterministic.
        """
        if not self.config.forward:
            return OpenShellCommandResult(ok=True, command=[], output="")

        args = [
            "forward",
            "start",
            "--gateway",
            self.config.gateway,
            self.config.forward,
            self.config.sandbox,
        ]
        if isinstance(self.runner, SubprocessCommandRunner):
            result = self._start_managed_forward(args)
            if not result.ok:
                return result
            if self.config.health_url:
                health = self.wait_for_health()
                return combine_results([result, health], ok=health.ok)
            return result

        result = self._exec.run([*args[:4], "--background", *args[4:]])
        if result.ok:
            return combine_results([result, self.wait_for_forward_active()])
        return result

    def wait_for_forward_active(self) -> OpenShellCommandResult:
        """Wait until the configured forward is listed as active."""
        if not self.config.forward:
            return OpenShellCommandResult(ok=True, command=[], output="")
        command = ["wait-for-forward", self.config.forward]
        last = {"output": ""}

        def _active() -> bool:
            forward_list = self._exec.run(["forward", "list", "--gateway", self.config.gateway])
            last["output"] = forward_list.output
            return forward_list.ok and self._forward_is_active(forward_list.output)

        if _poll_until(_active, timeout=min(self.config.health_timeout, 30.0), interval=1.0):
            return OpenShellCommandResult(
                ok=True,
                command=command,
                output=f"forward ready: {self.config.forward} -> {self.config.sandbox}",
            )
        return OpenShellCommandResult(
            ok=False,
            command=command,
            output=(
                f"timed out waiting for forward {self.config.forward} on sandbox {self.config.sandbox}\n{last['output']}"
            ).strip(),
        )

    def stop_managed_forward(self) -> OpenShellCommandResult:
        """Stop a forward process started by start_forward."""
        pid_path = self._managed_forward_pid_path()
        command = ["managed-forward-stop", str(pid_path)]
        try:
            raw_pid = pid_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return OpenShellCommandResult(ok=True, command=command, output="")
        except OSError as exc:
            return OpenShellCommandResult(ok=False, command=command, output=f"failed to read {pid_path}: {exc}")

        try:
            pid = int(raw_pid)
        except ValueError:
            pid_path.unlink(missing_ok=True)
            return OpenShellCommandResult(
                ok=True, command=command, output=f"removed invalid forward pid file {pid_path}"
            )

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            output = f"managed forward process {pid} was not running"
        except OSError as exc:
            return OpenShellCommandResult(
                ok=False, command=command, output=f"failed to stop managed forward {pid}: {exc}"
            )
        else:
            output = f"stopped managed forward process {pid}"
        pid_path.unlink(missing_ok=True)
        return OpenShellCommandResult(ok=True, command=command, output=output)

    def _forward_is_active(self, output: str) -> bool:
        if not self.config.forward:
            return False
        if self.config.forward in output and self.config.sandbox in output:
            return True
        bind, _, port = self.config.forward.rpartition(":")
        if not bind or not port:
            return False
        return self.config.sandbox in output and bind in output and port in output

    def _start_managed_forward(self, args: list[str]) -> OpenShellCommandResult:
        command = [self.config.openshell_bin, *args]
        started_at = time.monotonic()
        stop_result = self.stop_managed_forward()
        if not stop_result.ok:
            return stop_result
        try:
            log_path = self._exec.log_path_for(args)
            pid_path = self._managed_forward_pid_path()
            with log_path.open("w", encoding="utf-8") as stream:
                stream.write(f"command: {shlex.join(command)}\n")
                stream.flush()
                process = subprocess.Popen(  # noqa: S603 - command arguments are explicit lists built by lifecycle callers.
                    command,
                    cwd=self.config.cwd,
                    text=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            # OpenShell's forward helper can report launch success then immediately die without binding the
            # host port. Give it a short grace period; if it already exited, fail fast with the log instead
            # of leaving a dead forward for the caller's health poll to time out against.
            try:
                exit_code = process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                exit_code = None  # still running after the grace period — a healthy launch
            if exit_code is not None:
                log_tail = log_path.read_text(encoding="utf-8")[-2000:] if log_path.exists() else ""
                return OpenShellCommandResult(
                    ok=False,
                    command=command,
                    output=f"managed forward exited immediately (code {exit_code}); log:\n{log_tail}",
                )
        except OSError as exc:
            logger.info(
                "openshell background command failed op=%s duration_seconds=%.3f error=%s",
                _command_label(args),
                time.monotonic() - started_at,
                exc,
            )
            return OpenShellCommandResult(ok=False, command=command, output=str(exc))
        logger.info(
            "openshell background command started op=%s pid=%s duration_seconds=%.3f",
            _command_label(args),
            process.pid,
            time.monotonic() - started_at,
        )
        output = f"managed forward started pid={process.pid}; log={log_path}"
        if stop_result.output:
            output = f"{stop_result.output}\n{output}"
        return OpenShellCommandResult(ok=True, command=command, output=output)

    def _managed_forward_pid_path(self) -> Path:
        base_dir = self.log_dir or (self.config.cwd or Path.cwd()) / ".agent-hardener" / "openshell-forwards"
        base_dir.mkdir(parents=True, exist_ok=True)
        safe_sandbox = safe_path_token(self.config.sandbox)
        safe_forward = safe_path_token(self.config.forward or "forward")
        return base_dir / f"{safe_sandbox}-{safe_forward}.pid"
