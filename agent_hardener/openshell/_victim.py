# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Victim-process lifecycle inside the sandbox (mixed into :class:`OpenShellLifecycle`).

Start (with a detached launcher + transient-failure retry), stop, and restart the victim. Composed
as a mixin because these methods share the lifecycle's command executor, ``config``, exec helper, and
health polling.
"""

from __future__ import annotations

import logging
import shlex
import time
from typing import TYPE_CHECKING

from agent_hardener.openshell.config import OpenShellCommandResult
from agent_hardener.openshell.support import combine_results, is_transient_start_failure

if TYPE_CHECKING:
    from agent_hardener.openshell.command import OpenShellCommandExecutor
    from agent_hardener.openshell.config import OpenShellConfig

logger = logging.getLogger("agent_hardener.session")


class VictimMixin:
    """Start/stop/restart the configured victim process inside the sandbox."""

    # Provided by OpenShellLifecycle (the concrete class this mixin is composed into).
    config: OpenShellConfig
    _exec: OpenShellCommandExecutor

    def start_victim(self) -> OpenShellCommandResult:
        """Start the victim process inside the sandbox, retrying transient supervisor relay failures."""
        command = self._start_command_args()
        command_text = shlex.join([self.config.openshell_bin, *command])
        prefix = f"victim start command: {command_text}"
        logger.info("victim start command sandbox=%s command=%s", self.config.sandbox, command_text)
        deadline = time.monotonic() + min(self.config.health_timeout, 45.0)
        last_result = OpenShellCommandResult(ok=False, command=[self.config.openshell_bin, *command], output=prefix)
        while True:
            raw_result = self._exec.run(command)
            output = f"{prefix}\n{raw_result.output}".strip()
            last_result = OpenShellCommandResult(ok=raw_result.ok, command=raw_result.command, output=output)
            if raw_result.ok:
                return last_result
            if not is_transient_start_failure(raw_result.output):
                return last_result
            if time.monotonic() >= deadline:
                return last_result
            time.sleep(2)

    def stop_victim(self) -> OpenShellCommandResult:
        """Stop the victim process started by start_victim."""
        if not self.config.start_command:
            return OpenShellCommandResult(ok=True, command=[], output="no victim start command configured")
        stop_script = (
            "pid_file=/tmp/agent.pid; "
            'if [ ! -s "$pid_file" ]; then '
            'echo "victim pid file missing: $pid_file"; exit 1; '
            "fi; "
            'pid="$(cat "$pid_file")"; '
            'if kill "$pid" 2>/dev/null; then '
            'echo "stopped victim process $pid"; '
            "else "
            'echo "victim process $pid was not running"; '
            "fi; "
            'rm -f "$pid_file"; '
            "sleep 1"
        )
        return self._exec.run(self._sandbox_exec_args(stop_script))

    def restart_victim(self, wait_for_health: bool = True) -> OpenShellCommandResult:
        """Restart the configured victim process inside an existing sandbox."""
        results = [self.stop_victim()]
        if not results[-1].ok:
            return combine_results(results)

        if self.config.start_command:
            results.append(self.start_victim())
        if results[-1].ok and wait_for_health and self.config.health_url:
            results.append(self.wait_for_health())
        return combine_results(results)

    def _start_command_args(self) -> list[str]:
        """Return a sandbox-exec command that leaves the victim running after exec exits."""
        launcher = (
            "startup_log=/tmp/agent.startup.log; "
            "app_log=/tmp/agent.log; "
            "pid_file=/tmp/agent.pid; "
            '{ echo "victim launcher started"; date; echo "command: $1"; } > "$startup_log" 2>&1; '
            'nohup sh -lc "exec $1" >> "$app_log" 2>&1 </dev/null & '
            'pid=$!; echo "$pid" > "$pid_file"; echo "pid=$pid" >> "$startup_log"; '
            "sleep 1; "
            'if [ -f "$app_log" ]; then '
            'tail -80 "$app_log" >> "$startup_log" 2>&1; '
            "else "
            'echo "app log missing after launch" >> "$startup_log"; '
            "fi"
        )
        return self._sandbox_exec_args(launcher, "sh", self.config.start_command or "")
