# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness/health polling for the OpenShell lifecycle (mixed into :class:`OpenShellLifecycle`).

Groups the "wait until X" operations and their in-sandbox startup diagnostics. Composed as a mixin
because these methods share the lifecycle's ``config``, command executor, and ``status``/exec helpers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib import error, request
from urllib.parse import urlsplit

from agent_hardener.openshell.config import OpenShellCommandResult
from agent_hardener.openshell.support import _poll_until

if TYPE_CHECKING:
    from agent_hardener.openshell.command import OpenShellCommandExecutor
    from agent_hardener.openshell.config import OpenShellConfig

logger = logging.getLogger("agent_hardener.session")


class PollingMixin:
    """Sandbox-ready / victim-health polling and startup diagnostics."""

    # Provided by OpenShellLifecycle (the concrete class this mixin is composed into).
    config: OpenShellConfig
    _exec: OpenShellCommandExecutor

    def wait_for_sandbox_ready(self) -> OpenShellCommandResult:
        """Wait until the configured sandbox reaches Ready state."""
        command = ["wait-for-sandbox", self.config.sandbox]
        logger.info("sandbox setup: sandbox ready polling sandbox=%s", self.config.sandbox)
        last = {"output": ""}

        def _ready() -> bool:
            status = self.status()
            last["output"] = status.output
            return status.ok and "Ready" in status.output

        def _tick(elapsed: int) -> None:
            logger.debug("sandbox setup: sandbox ready tick elapsed=%ss sandbox=%s", elapsed, self.config.sandbox)

        if _poll_until(_ready, timeout=self.config.sandbox_timeout, interval=2.0, on_tick=_tick):
            return OpenShellCommandResult(ok=True, command=command, output=f"sandbox ready: {self.config.sandbox}")
        return OpenShellCommandResult(
            ok=False,
            command=command,
            output=f"timed out waiting for sandbox {self.config.sandbox} to become ready\n{last['output']}".strip(),
        )

    def wait_for_health(self) -> OpenShellCommandResult:
        """Wait until the configured health URL responds successfully."""
        if not self.config.health_url:
            return OpenShellCommandResult(ok=True, command=[], output="")
        command = ["wait-for-health", self.config.health_url]
        try:
            wait_for_http_health(self.config.health_url, self.config.health_timeout)
        except Exception as exc:
            diagnostics = self.collect_victim_diagnostics()
            output = str(exc)
            if diagnostics.output:
                output = f"{output}\n\nVictim startup diagnostics:\n{diagnostics.output}"
            full_command = [*command, "&&", *diagnostics.command] if diagnostics.command else command
            return OpenShellCommandResult(ok=False, command=full_command, output=output)
        return OpenShellCommandResult(ok=True, command=command, output=f"health ready: {self.config.health_url}")

    def collect_victim_diagnostics(self) -> OpenShellCommandResult:
        """Collect best-effort startup diagnostics from inside the sandbox."""
        diagnostics_script = "; ".join(
            [
                "echo '=== /tmp victim files ==='",
                "ls -lh /tmp/agent.log /tmp/agent.startup.log /tmp/relay-victim.log 2>&1 || true",
                "echo '=== process snapshot ==='",
                "ps -ef 2>/dev/null || tr '\\0' ' ' </proc/1/cmdline 2>/dev/null || true",
                "echo '=== startup log ==='",
                "tail -120 /tmp/agent.startup.log 2>&1 || true",
                "echo '=== app log (launcher-captured) ==='",
                "tail -200 /tmp/agent.log 2>&1 || true",
                "echo '=== relay-victim log (if start_command redirected) ==='",
                "tail -200 /tmp/relay-victim.log 2>&1 || true",
            ]
        )
        return self._exec.run(self._sandbox_exec_args(diagnostics_script))


def wait_for_http_health(url: str, timeout_seconds: float) -> None:
    """Wait until an HTTP health endpoint returns a 2xx response."""
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported health URL scheme: {url}")

    def _healthy() -> bool:
        try:
            with request.urlopen(url, timeout=5) as response:  # noqa: S310 - scheme is validated above.
                return 200 <= response.status < 300
        except (OSError, error.URLError):
            return False

    def _tick(elapsed: int) -> None:
        logger.debug("sandbox setup: health check tick elapsed=%ss url=%s", elapsed, url)

    if not _poll_until(_healthy, timeout=timeout_seconds, interval=2.0, on_tick=_tick):
        raise TimeoutError(f"timed out waiting for victim health at {url}")
