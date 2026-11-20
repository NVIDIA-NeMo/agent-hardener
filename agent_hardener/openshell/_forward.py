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
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent_hardener.errors import VictimPortInUseError
from agent_hardener.openshell.command import _command_label
from agent_hardener.openshell.config import OpenShellCommandResult, SubprocessCommandRunner
from agent_hardener.openshell.support import _poll_until, combine_results, safe_path_token

if TYPE_CHECKING:
    from agent_hardener.openshell.command import OpenShellCommandExecutor
    from agent_hardener.openshell.config import CommandRunner, OpenShellConfig

logger = logging.getLogger("agent_hardener.session")


#: How OpenShell says a forward could not take its host port: another process holds it, or another
#: sandbox's forward does.
_PORT_TAKEN = re.compile(r"Port (?P<port>\d+) is already (?:in use by another process|forwarded to sandbox '[^']+')")


def port_in_use_error(output: str) -> VictimPortInUseError | None:
    """The classified error for a forward that failed because its host port was taken; None for any other failure."""
    match = _PORT_TAKEN.search(output)
    if match is None:
        return None
    port = match["port"]
    return VictimPortInUseError(
        f"{match[0]}, so the victim cannot be reached on it",
        remediation=(
            f"Free port {port} (find the process with `lsof -i :{port} -sTCP:LISTEN`; for another run's forward, "
            f"`openshell forward list` names the sandbox to `openshell forward stop {port} <sandbox>`), "
            "or give this agent another agent.port."
        ),
    )


#: Where Linux exposes each process's command line. A module constant so tests can point it elsewhere.
PROC_ROOT = Path("/proc")


class Unverifiable:
    """:func:`process_command_line`'s answer when it cannot look a process up at all."""

    def __repr__(self) -> str:
        return "UNVERIFIABLE"


#: Neither ``/proc`` nor ``ps`` could say what the process is — not the same as "no such process".
UNVERIFIABLE = Unverifiable()


def process_command_line(pid: int) -> str | Unverifiable | None:
    """The command line of process *pid*; None when there is no such process; UNVERIFIABLE when we can't tell.

    ``/proc/<pid>/cmdline`` first: it is always there on Linux and needs no binary — Helix's distroless task
    image has no ``ps``. ``ps`` only where there is no ``/proc``, as on macOS.
    """
    if PROC_ROOT.is_dir():
        try:
            raw = (PROC_ROOT / str(pid) / "cmdline").read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return UNVERIFIABLE
        # Arguments are NUL-separated. A zombie or kernel thread has none: present, but nothing to check.
        args = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        return args or UNVERIFIABLE
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, the pid is an int.
            ["ps", "-p", str(pid), "-o", "args="],  # noqa: S607 - ps is resolved from PATH on purpose.
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNVERIFIABLE
    args = completed.stdout.strip()
    return args if completed.returncode == 0 and args else None


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

    def ensure_forward(self) -> OpenShellCommandResult:
        """Make sure the configured forward is this sandbox's, starting it when it is not listed.

        A reused sandbox skips ``up``, so without this nothing checks who serves the victim's port: when
        another run's forward holds it, the probes reach that run's victim and blame this one.
        """
        if not self.config.forward:
            return OpenShellCommandResult(ok=True, command=[], output="")
        forward_list = self._exec.run(["forward", "list", "--gateway", self.config.gateway])
        if forward_list.ok and self._forward_is_active(forward_list.output):
            return forward_list
        started = self.start_forward()
        # A failed listing is not fatal: starting the forward is what decides.
        return combine_results([forward_list, started], ok=started.ok)

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

        # The pid can outlive its process — a pid file left by an earlier run, the pid since reused — and
        # signalling it would then take down an unrelated process group. Only stop what is still our forward.
        owned = self._is_managed_forward(pid)
        if owned is False:
            pid_path.unlink(missing_ok=True)
            return OpenShellCommandResult(
                ok=True, command=command, output=f"managed forward process {pid} was not running"
            )
        if owned is None:
            # Can't tell is not "gone": skipping the stop would leave every forward running, our own included.
            # Signal it as before the check existed; the reuse risk stays only where nothing can rule it out.
            logger.warning(
                "cannot verify that pid %s is still the forward for sandbox %s (no /proc or ps); stopping it anyway",
                pid,
                self.config.sandbox,
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

    def _is_managed_forward(self, pid: int) -> bool | None:
        """Whether *pid* is still this lifecycle's ``openshell forward start`` for the configured sandbox.

        None when that cannot be checked here (see :func:`process_command_line`).
        """
        args = process_command_line(pid)
        if args is None:
            return False
        if isinstance(args, Unverifiable):
            return None
        return "forward start" in args and self.config.sandbox in args.split()

    def _forward_is_active(self, output: str) -> bool:
        """Whether one line of ``openshell forward list`` shows this sandbox forwarding the configured port.

        Per line: across the whole listing, this sandbox's forward of one port and another sandbox's forward
        of the configured port would read as this sandbox holding it.
        """
        if not self.config.forward:
            return False
        bind, _, port = self.config.forward.rpartition(":")
        for line in output.splitlines():
            if self.config.sandbox not in line:
                continue
            if self.config.forward in line:
                return True
            if bind and port and bind in line and port in line:
                return True
        return False

    def _start_managed_forward(self, args: list[str]) -> OpenShellCommandResult:
        command = [self.config.openshell_bin, *args]
        started_at = time.monotonic()
        stop_result = self.stop_managed_forward()
        if not stop_result.ok:
            return stop_result
        try:
            log_path = self._exec.log_path_for(args)
            pid_path = self._managed_forward_pid_path()
            pid_path.parent.mkdir(parents=True, exist_ok=True)
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
        # Keyed by sandbox and forward, never by run: a run killed before its teardown leaves the forward
        # running and holding the port, and only a pid file the *next* run can find lets it stop that forward.
        # Under the per-run log dir, every run looked in its own empty folder.
        base_dir = (self.config.cwd or Path.cwd()) / ".agent-hardener" / "openshell-forwards"
        safe_sandbox = safe_path_token(self.config.sandbox)
        safe_forward = safe_path_token(self.config.forward or "forward")
        return base_dir / f"{safe_sandbox}-{safe_forward}.pid"
