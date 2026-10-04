# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, ``self``-free helpers for the OpenShell lifecycle.

Result combination, a poll loop, dotenv reading, and the transient/absent-failure predicates — used
by the lifecycle core and each of its mixins (polling / victim / forward), so they live in one
dependency-free module (imports only the value types) to avoid cycles.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agent_hardener.openshell.config import OpenShellCommandResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def safe_path_token(value: str) -> str:
    """Filesystem-safe token for a log/pid filename: keep alnum and ``-._``, replace anything else with ``_``."""
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in value)


def _poll_until(
    check: Callable[[], bool],
    *,
    timeout: float,
    interval: float,
    on_tick: Callable[[int], None] | None = None,
) -> bool:
    """Poll ``check`` until it returns True or ``timeout`` elapses.

    Returns True when ``check`` succeeded, False on timeout. ``on_tick`` (if given) is called with the
    integer elapsed seconds before each sleep — used for the per-tick debug logs.
    """
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    while True:
        if check():
            return True
        if time.monotonic() >= deadline:
            return False
        if on_tick is not None:
            on_tick(int(time.monotonic() - started))
        time.sleep(interval)


def combine_results(results: list[OpenShellCommandResult], ok: bool | None = None) -> OpenShellCommandResult:
    """Combine multiple lifecycle command results."""
    return OpenShellCommandResult(
        ok=all(result.ok for result in results) if ok is None else ok,
        command=[part for result in results for part in [*result.command, "&&"]][:-1],
        output="\n".join(result.output for result in results if result.output).strip(),
    )


def read_env_file(path: Path | None) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from a dotenv-style file."""
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def is_sandbox_not_found(output: str) -> bool:
    """Return whether an OpenShell delete/get failure only means the sandbox is absent."""
    normalized = output.lower()
    return "notfound" in normalized and "sandbox not found" in normalized


def is_supervisor_relay_unavailable(output: str) -> bool:
    """Return whether sandbox exec failed before the supervisor relay connected."""
    normalized = output.lower()
    return "supervisor relay failed" in normalized or "supervisor session not connected" in normalized


def is_transient_start_failure(output: str) -> bool:
    """Return whether a victim start failure looks like a transient sandbox-readiness race."""
    normalized = output.lower()
    return is_supervisor_relay_unavailable(output) or (
        "sandbox" in normalized and "is not ready" in normalized and "provisioning" in normalized
    )


def is_transient_transport_failure(output: str) -> bool:
    """Return whether an OpenShell command failed due to a transient gateway transport reset."""
    normalized = output.lower()
    return (
        "transport error" in normalized
        or "connection reset by peer" in normalized
        or "status: unavailable" in normalized
        or "gateway connect failed" in normalized
        or "ssh exited with status" in normalized
        # OpenShell gRPC stream reader sometimes fails to consume trailing bytes after
        # a successful Docker build. The sandbox is usually created server-side; the
        # retry loop will detect this via sandbox get and continue rather than failing.
        or "bytes remaining on stream" in normalized
    )
