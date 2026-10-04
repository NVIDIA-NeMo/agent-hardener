# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable OpenShell lifecycle controls.

:class:`OpenShellLifecycle` orchestrates the gateway/sandbox: bring it up and down, ensure the
provider, and apply policies. The victim, forward, and readiness/health-polling operations live in
focused mixins (:mod:`._victim`, :mod:`._forward`, :mod:`._polling`); the raw "run one command"
plumbing lives in :class:`~agent_hardener.openshell.command.OpenShellCommandExecutor` (composed as
``self._exec``); shared helpers/predicates live in :mod:`.support`; config/result value types in
:mod:`.config`. The public surface is re-exported here for backward-compatible imports.
"""

from __future__ import annotations

import logging
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent_hardener.openshell._forward import ForwardMixin
from agent_hardener.openshell._polling import PollingMixin, wait_for_http_health
from agent_hardener.openshell._victim import VictimMixin
from agent_hardener.openshell.command import OpenShellCommandExecutor
from agent_hardener.openshell.config import (
    CommandRunner,
    OpenShellCommandResult,
    OpenShellConfig,
    SubprocessCommandRunner,
)
from agent_hardener.openshell.support import (
    combine_results,
    is_sandbox_not_found,
    is_supervisor_relay_unavailable,
    is_transient_start_failure,
    is_transient_transport_failure,
    read_env_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("agent_hardener.session")

# Re-exported so ``from agent_hardener.openshell.lifecycle import X`` keeps working for existing importers
# after the split into config/command/support/mixin modules.
__all__ = [
    "CommandRunner",
    "OpenShellCommandResult",
    "OpenShellConfig",
    "OpenShellLifecycle",
    "SubprocessCommandRunner",
    "combine_results",
    "is_sandbox_not_found",
    "is_supervisor_relay_unavailable",
    "is_transient_start_failure",
    "is_transient_transport_failure",
    "read_env_file",
    "wait_for_http_health",
]


class OpenShellLifecycle(PollingMixin, VictimMixin, ForwardMixin):
    """Thin reusable wrapper around OpenShell gateway and sandbox commands."""

    def __init__(
        self,
        config: OpenShellConfig,
        runner: CommandRunner | None = None,
        log_dir: Path | None = None,
        active_state_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()
        # Where background command logs are written. The composition root (runner/UI) injects the run's
        # ``run-logs/<run_id>/openshell-logs`` here; when unset (standalone/CLI-direct) we fall back to the
        # shared ``.agent-hardener/openshell-logs`` under cwd.
        self.log_dir = log_dir
        # The run's ``victim-active-state`` dir, holding the files the defenders have hardened. A sandbox
        # create uploads those in place of the configured seeds, so a recreate keeps deployed guardrails.
        self.active_state_dir = active_state_dir
        self._exec = OpenShellCommandExecutor(self.config, self.runner, self.log_dir)

    def status(self) -> OpenShellCommandResult:
        """Return sandbox status."""
        return self._run_with_transport_retries(
            self._exec.run, ["sandbox", "get", self.config.sandbox, "--gateway", self.config.gateway]
        )

    def up(self, policy_path: Path | None = None) -> OpenShellCommandResult:
        """Create the sandbox when a build context is configured."""
        effective_policy_path = policy_path or self.config.policy_path
        results = self._ensure_gateway_started()
        if not results[-1].ok:
            return combine_results(results)

        sandbox_status = self.status()
        results.append(sandbox_status)
        if self.config.build_context is None:
            return combine_results(results, ok=results[-1].ok)

        if not sandbox_status.ok:
            if is_transient_transport_failure(sandbox_status.output):
                return combine_results(results)
            if self.config.provider:
                results.append(self.ensure_provider())
                if not results[-1].ok:
                    return combine_results(results)
            results.extend(self._create_sandbox(effective_policy_path))

        self._start_configured_services(results)
        return combine_results(results, ok=results[-1].ok)

    def _ensure_gateway_started(self) -> list[OpenShellCommandResult]:
        results = [self._exec.run(["gateway", "info", "--gateway", self.config.gateway])]
        if not results[-1].ok:
            start_result = self._exec.run(["gateway", "start", "--name", self.config.gateway])
            if "unrecognized subcommand" in start_result.output:
                # gateway start was removed in OpenShell >=0.0.45; the gateway is
                # managed by the system service. Record a synthetic success so up()
                # does not bail on the failed `gateway info`, and let subsequent
                # sandbox commands surface the real error if the gateway is down.
                logger.debug("openshell gateway start not supported; skipping (system service manages gateway)")
                results.append(
                    OpenShellCommandResult(
                        ok=True,
                        command=start_result.command,
                        output=(
                            "openshell gateway start not supported; "
                            "continuing with sandbox commands\n"
                            f"{start_result.output}"
                        ).strip(),
                    )
                )
            else:
                results.append(start_result)
        return results

    def _create_sandbox(self, effective_policy_path: Path) -> list[OpenShellCommandResult]:
        args = self._sandbox_create_args(effective_policy_path)
        uploads = self._effective_uploads()
        if uploads:
            logger.info("sandbox create uploads sandbox=%s uploads=%s", self.config.sandbox, uploads)
        logger.info(
            "sandbox create command sandbox=%s command=%s",
            self.config.sandbox,
            shlex.join([self.config.openshell_bin, *args]),
        )
        logger.info("sandbox setup: sandbox create starting sandbox=%s", self.config.sandbox)
        results = [
            self._run_with_transport_retries(
                self._exec.run_logged, args, delay_seconds=10.0, reconcile=self.status, warn=True
            )
        ]
        if results[-1].ok:
            results.append(self.wait_for_sandbox_ready())
        return results

    def _sandbox_create_args(self, effective_policy_path: Path) -> list[str]:
        args = [
            "sandbox",
            "create",
            "--gateway",
            self.config.gateway,
            "--name",
            self.config.sandbox,
            "--from",
            str(self.config.build_context),
        ]
        uploads = self._effective_uploads()
        for upload in uploads:
            args.extend(["--upload", upload])
        if uploads:
            args.append("--no-git-ignore")
        args.extend(["--policy", str(effective_policy_path), "--no-tty"])
        if self.config.provider:
            args.extend(["--provider", self.config.provider, "--no-auto-providers"])
        args.extend(["--", *self.config.create_command])
        return args

    def _effective_uploads(self) -> list[str]:
        """``config.uploads`` with each source swapped for its same-named active-state copy, if one exists.

        The configured sources are the run's seeds. A policy patch that forces a recreate would otherwise
        re-upload the pristine seed over the hardened ``plugins.toml`` and silently drop every guardrail
        deployed so far. Before the run seeds its active state (initial bring-up) the seeds are used as-is.
        """
        if self.active_state_dir is None:
            return list(self.config.uploads)
        uploads = []
        for upload in self.config.uploads:
            local, separator, destination = upload.partition(":")
            active = self.active_state_dir / Path(local).name
            uploads.append(f"{active}:{destination}" if separator and active.is_file() else upload)
        return uploads

    def _start_configured_services(self, results: list[OpenShellCommandResult]) -> None:
        if results[-1].ok and self.config.start_command:
            results.append(self.start_victim())
        if results[-1].ok and self.config.forward:
            forward_list = self._exec.run(["forward", "list", "--gateway", self.config.gateway])
            results.append(forward_list)
            if not self._forward_is_active(forward_list.output):
                results.append(self.start_forward())

    def ensure_provider(self) -> OpenShellCommandResult:
        """Ensure the configured provider exists for sandbox credential injection."""
        if not self.config.provider:
            return OpenShellCommandResult(ok=True, command=[], output="")

        results = [
            self._run_with_transport_retries(
                self._exec.run, ["provider", "get", self.config.provider, "--gateway", self.config.gateway]
            )
        ]
        if results[-1].ok:
            return combine_results(results)
        if is_transient_transport_failure(results[-1].output):
            return combine_results(results)

        args = [
            "provider",
            "create",
            "--name",
            self.config.provider,
            "--gateway",
            self.config.gateway,
            "--type",
            self.config.provider_type,
        ]
        if self.config.provider_env_file:
            try:
                credentials = self._provider_credentials_from_env_file()
            except OSError as exc:
                return OpenShellCommandResult(
                    ok=False,
                    command=[self.config.openshell_bin, *args],
                    output=f"failed to read provider_env_file {self.config.provider_env_file}: {exc}",
                )
            if not credentials:
                return OpenShellCommandResult(
                    ok=False,
                    command=[self.config.openshell_bin, *args],
                    output=f"no configured provider credentials found in {self.config.provider_env_file}",
                )
            for name, value in credentials.items():
                args.extend(["--credential", f"{name}={value}"])
        else:
            args.append("--from-existing")
        results.append(self._exec.run(args))
        return combine_results(results, ok=results[-1].ok)

    def down(self) -> OpenShellCommandResult:
        """Delete the sandbox."""
        results: list[OpenShellCommandResult] = []
        if self.config.forward:
            results.append(self.stop_managed_forward())
        results.append(self._exec.run(["sandbox", "delete", self.config.sandbox, "--gateway", self.config.gateway]))
        return combine_results(results, ok=results[-1].ok)

    def restart(self, policy_path: Path | None = None, wait_for_health: bool = True) -> OpenShellCommandResult:
        """Recreate the sandbox using delete followed by create."""
        down = self.down()
        if not down.ok and is_sandbox_not_found(down.output):
            down = OpenShellCommandResult(ok=True, command=down.command, output=down.output)
        up = self.up(policy_path=policy_path)
        results = [down, up]
        if up.ok and wait_for_health and self.config.health_url:
            results.append(self.wait_for_health())
        return combine_results(results)

    def apply_policy(self, policy_path: Path) -> OpenShellCommandResult:
        """Apply a policy to a running sandbox."""
        return self._exec.run(
            [
                "policy",
                "set",
                self.config.sandbox,
                "--gateway",
                self.config.gateway,
                "--policy",
                str(policy_path),
                "--wait",
                "--timeout",
                str(int(self.config.policy_wait_timeout)),
            ],
        )

    def active_policy(self) -> OpenShellCommandResult:
        """Return the current active sandbox policy as YAML."""
        return self._exec.run(
            [
                "sandbox",
                "get",
                self.config.sandbox,
                "--gateway",
                self.config.gateway,
                "--policy-only",
            ]
        )

    def upload_file(self, local_path: Path, destination: str) -> OpenShellCommandResult:
        """Upload a local file into the running sandbox."""
        return self._exec.run(
            [
                "sandbox",
                "upload",
                "--gateway",
                self.config.gateway,
                "--no-git-ignore",
                self.config.sandbox,
                str(local_path),
                destination,
            ]
        )

    def fetch_file(self, remote_path: str, local_path: Path) -> bool:
        """Copy a file out of the sandbox to *local_path*; return whether it arrived.

        The sandbox shares no filesystem with the host, so a file the victim writes — its ATOF
        stream above all — is invisible until it is pulled. ``cat`` over ``sandbox exec`` rather than
        a download API because OpenShell exposes upload but no matching download.

        Falls back to same-named files one level down, concatenated, because a Fabric victim does not
        write where it was told to: ``normalize_relay_output_dirs`` appends the runtime id, so the
        stream lands at ``<output_dir>/<runtime-id>/events.atof.jsonl`` — and Fabric opens a new
        runtime per session, so one victim's telemetry is spread across a directory per invocation.
        Agent Hardener attacks concurrently with a session id each, so taking only the newest would keep
        one attack's events and discard the rest.

        Concatenating is safe: ATOF is JSONL, and every consumer here attributes by the record's own
        fields rather than by position in the file.

        Missing-file is a normal outcome, not an error: callers poll this while the victim is still
        starting up, before it has emitted anything.
        """
        script = (
            'if [ -f "$1" ]; then cat "$1"; else cat "$(dirname "$1")"/*/"$(basename "$1")"; fi 2>/dev/null || true'
        )
        result = self._exec.run(self._sandbox_exec_args(script, "sh", remote_path))
        if not result.ok or not result.output:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(result.output, encoding="utf-8")
        return True

    def _sandbox_exec_args(self, script: str, *tail: str) -> list[str]:
        """Build a ``sandbox exec ... -- sh -lc <script> [tail...]`` command for the configured sandbox."""
        return [
            "sandbox",
            "exec",
            "--gateway",
            self.config.gateway,
            "--name",
            self.config.sandbox,
            "--no-tty",
            "--",
            "sh",
            "-lc",
            script,
            *tail,
        ]

    def _run_with_transport_retries(
        self,
        run_fn: Callable[[list[str]], OpenShellCommandResult],
        args: list[str],
        *,
        attempts: int = 3,
        delay_seconds: float = 5.0,
        reconcile: Callable[[], OpenShellCommandResult] | None = None,
        warn: bool = False,
    ) -> OpenShellCommandResult:
        """Run an OpenShell command via ``run_fn``, retrying transient gateway transport resets.

        When ``reconcile`` is given, after a transient failure + delay it is called (e.g. ``status``):
        if it reports the sandbox already exists, the retry short-circuits as success — this covers the
        case where a create command's gRPC stream fails after the sandbox was created server-side.
        ``warn`` logs a warning per transient attempt (used for the noisy sandbox-create path).
        """
        results: list[OpenShellCommandResult] = []
        for attempt in range(1, attempts + 1):
            result = run_fn(args)
            if result.ok or not is_transient_transport_failure(result.output) or attempt == attempts:
                if results:
                    results.append(result)
                    return combine_results(results, ok=result.ok)
                return result
            if warn:
                logger.warning(
                    "transient OpenShell transport failure command=%s attempt=%s/%s",
                    shlex.join([self.config.openshell_bin, *args]),
                    attempt,
                    attempts,
                )
            results.append(
                OpenShellCommandResult(
                    ok=False,
                    command=result.command,
                    output=f"transient OpenShell transport failure on attempt {attempt}/{attempts}: {result.output}",
                )
            )
            time.sleep(delay_seconds)
            if reconcile is not None:
                status = reconcile()
                if status.ok:
                    results.append(
                        OpenShellCommandResult(
                            ok=True,
                            command=status.command,
                            output=f"sandbox exists after transient create failure; continuing\n{status.output}",
                        )
                    )
                    return combine_results(results, ok=True)
                if is_transient_transport_failure(status.output):
                    results.append(
                        OpenShellCommandResult(
                            ok=False,
                            command=status.command,
                            output=f"status check after transient create failure still failed: {status.output}",
                        )
                    )
        return results[-1]

    def _provider_credentials_from_env_file(self) -> dict[str, str]:
        env_values = read_env_file(self.config.provider_env_file)
        credential_names = self.config.provider_credentials or sorted(env_values)
        return {name: env_values[name] for name in credential_names if env_values.get(name)}
