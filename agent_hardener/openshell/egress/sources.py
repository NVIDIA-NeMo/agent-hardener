# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete egress sources: workflow YAML, project Python code, manual hints, and a log-trace stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.openshell.egress.base import DiscoveredEndpoint, EgressSource, parse_urls

if TYPE_CHECKING:
    from pathlib import Path

# Directories not worth scanning for source URLs (huge / irrelevant).
_CODE_IGNORE = frozenset({".git", ".venv", "venv", "__pycache__", "node_modules", ".agent-hardener", "dist", "build"})
_MAX_CODE_FILES = 2000


class CodeEgressSource(EgressSource):
    """Hosts hardcoded in the project's source and config files.

    Scans configuration as well as Python. Under NAT there was a single known workflow YAML to read
    ``base_url`` out of; a Relay-connected agent's config is whatever format its own framework uses,
    so the only generic option is to scan the text config files a project is likely to keep its
    endpoints in. Missing one is not cosmetic: the sandbox is default-deny, so an undiscovered host
    means the agent's outbound call is dropped mid-run.
    """

    #: Extensions scanned for URLs. Text formats only — a binary would just add noise.
    SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".json")

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir

    def _candidates(self) -> list[Path]:
        return sorted(path for suffix in self.SCANNED_SUFFIXES for path in self.project_dir.rglob(f"*{suffix}"))

    def discover(self) -> set[DiscoveredEndpoint]:
        endpoints: set[DiscoveredEndpoint] = set()
        scanned = 0
        for path in self._candidates():
            if any(part in _CODE_IGNORE for part in path.parts):
                continue
            if scanned >= _MAX_CODE_FILES:
                break
            scanned += 1
            try:
                endpoints |= parse_urls(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
        return endpoints


class ManualEgressSource(EgressSource):
    """User-supplied egress hints (manifest ``agent.egress``): bare hosts or full URLs."""

    def __init__(self, entries: list[str]) -> None:
        self.entries = entries

    def discover(self) -> set[DiscoveredEndpoint]:
        endpoints: set[DiscoveredEndpoint] = set()
        for raw in self.entries:
            entry = raw.strip()
            if not entry:
                continue
            if "://" in entry:
                endpoints |= parse_urls(entry)
            else:
                # Bare "host" or "host:port" → default to 443 when no port is given.
                host, _, port = entry.partition(":")
                endpoints.add(DiscoveredEndpoint(host=host.rstrip(".").lower(), port=int(port) if port else 443))
        return endpoints


class DockerfileEnvEgressSource(EgressSource):
    """Hosts named by the victim image's own ``ENV`` declarations.

    An agent told where its backend lives — ``BACKEND_URL=https://ledger.internal`` — needs that host
    reachable, and nothing else in the run knows about it: it is not in the code (it is configuration)
    and not in the manifest's ``egress`` (the author already said it once). Reading the composed
    Dockerfile keeps one source of truth for the victim's environment.
    """

    def __init__(self, dockerfile: Path) -> None:
        self.dockerfile = dockerfile

    def discover(self) -> set[DiscoveredEndpoint]:
        from agent_hardener.openshell.relay_victim import dockerfile_env  # noqa: PLC0415

        if not self.dockerfile.is_file():
            return set()
        endpoints: set[DiscoveredEndpoint] = set()
        for value in dockerfile_env(self.dockerfile).values():
            if "://" in value:
                endpoints |= parse_urls(value)
        return endpoints


class OpenShellLogEgressSource(EgressSource):
    """Seam for dynamic discovery from OpenShell deny-traces (``openshell logs --source sandbox``).

    Not implemented yet: the blocked-egress log line format is unconfirmed (pending #openshell-dev).
    Kept as a first-class source so it can be added later without touching the discoverer or callers.
    """

    def __init__(self, sandbox: str, gateway: str) -> None:
        self.sandbox = sandbox
        self.gateway = gateway

    def discover(self) -> set[DiscoveredEndpoint]:  # pragma: no cover - stub until log format is confirmed
        return set()
