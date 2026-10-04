# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project discovery and secret-file heuristics used by ``agent-hardener init``.

Pure filesystem/heuristic helpers (no prompting, no display): locate the installable project
root, discover workflow/Dockerfile/dotenv files, derive a default agent name, scan for egress
endpoints, and inspect the agent's secrets file. The interactive layer lives in ``cli_prompts``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_INSTALL_MARKERS = ("pyproject.toml", "uv.lock", "requirements.txt")

# Dirs whose contents are dependencies/VCS/build noise, not the user's agent — never discover inside
# them (mirrors the egress CodeEgressSource ignore list so a vendored Dockerfile/workflow isn't picked).
_DISCOVERY_IGNORE = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", ".agent-hardener", "dist", "build"}
)


def _discover(project_dir: Path, *patterns: str) -> list[Path]:
    """Recursively glob ``patterns`` under ``project_dir``, skipping vendored/build dirs (sorted, deduped)."""
    if not project_dir.is_dir():
        return []
    found = {path for pattern in patterns for path in project_dir.rglob(pattern)}
    return sorted(p for p in found if not any(part in _DISCOVERY_IGNORE for part in p.parts))


def discover_dockerfiles(project_dir: Path) -> list[Path]:
    """Return every Dockerfile under ``project_dir`` (absolute, sorted).

    Matches ``Dockerfile``, ``Dockerfile.<suffix>`` (e.g. ``Dockerfile.prod``) and ``<name>.Dockerfile``,
    so init can offer a detected image. Skips
    vendored/build dirs (``.venv``/``.git``/etc.) so a dependency's Dockerfile isn't picked.
    """
    return _discover(project_dir, "Dockerfile", "Dockerfile.*", "*.Dockerfile")


def discover_env_files(*dirs: Path) -> list[Path]:
    """Return existing ``.env`` / ``.env.*`` files in ``dirs`` (deduped, in the order given).

    So init can offer a detected secrets file instead of always
    defaulting to ``.env``. Plain ``.env`` sorts before its variants within each directory.
    """
    found: list[Path] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in [directory / ".env", *sorted(directory.glob(".env.*"))]:
            if path.is_file() and path not in found:
                found.append(path)
    return found


def default_agent_name(project_dir_value: str, workflow_value: str | None) -> str:
    """Derive a default agent name from the chosen workflow (else the project dir).

    Uses the workflow's parent folder (e.g. ``codereview``) and appends any filename variant, so
    ``agents/codereview/workflow.hardened.yaml`` -> ``codereview-hardened`` — keeping each agent
    variant distinct (separate sandbox, artifacts, and benign-profile cache).
    """
    if workflow_value:
        wf = Path(workflow_value)
        base = wf.parent.name or Path(project_dir_value).name
        parts = wf.name.split(".")  # workflow[.<variant>].yaml
        if len(parts) >= 3 and parts[0] == "workflow":
            base = f"{base}-{'-'.join(parts[1:-1])}"
        if base:
            return base
    return Path(project_dir_value).name or "agent"


def find_project_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the nearest dir with an installable-project marker.

    Lets a user point ``init`` at an agent *subfolder* and still get the real project root
    (the dir with pyproject.toml/uv.lock) — which is what must be built and served.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _INSTALL_MARKERS):
            return candidate
    return None


def discover_egress_entries(project_root: Path) -> list[str]:
    """Scan the project's code and config for external hosts the agent reaches (offline, no DNS).

    Returns manifest-style entries (``host`` or ``host:port`` when not 443), sorted and de-duplicated.
    Reuses the same source the sandbox build uses (:mod:`agent_hardener.openshell.egress`).
    """
    from agent_hardener.openshell.egress import CodeEgressSource, EgressDiscoverer  # noqa: PLC0415

    discovered = EgressDiscoverer([CodeEgressSource(project_root)]).endpoints()
    return sorted({ep.host if ep.port == 443 else f"{ep.host}:{ep.port}" for ep in discovered})


def discover_backend_ports(project_root: Path) -> list[int]:
    """Scan the project for host-backend ports (``localhost``/``127.0.0.1:<port>`` its tools call).

    A pure text scan (no execution): agents that hardcode a local service (a DB/API) reference it as
    ``localhost:<port>``, which the sandbox can't reach as-is. Returned ports seed the ``init``/wizard
    backend prompt so those services get the sandbox->host route.

    Scans the same files as :class:`CodeEgressSource` rather than one known config, because a
    Relay-connected agent keeps its endpoints in whatever format its own framework uses.
    """
    from agent_hardener.openshell.egress import CodeEgressSource  # noqa: PLC0415

    ports: set[int] = set()
    for path in CodeEgressSource(project_root)._candidates():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ports |= {int(match) for match in re.findall(r"(?:localhost|127\.0\.0\.1):(\d+)", text)}
    return sorted(ports)


def secret_names_from_file(env_path: Path) -> list[str]:
    """Return the env-var names present in ``env_path`` (empty if it doesn't exist)."""
    from agent_hardener.openshell.lifecycle import read_env_file  # noqa: PLC0415

    return list(read_env_file(env_path).keys()) if env_path.exists() else []


def missing_secrets(secret_names: list[str], env_path: Path) -> list[str]:
    """Return the secret names not already set in the environment or ``env_path``."""
    from agent_hardener.openshell.lifecycle import read_env_file  # noqa: PLC0415

    existing = read_env_file(env_path) if env_path.exists() else {}
    return [name for name in secret_names if not os.environ.get(name) and not existing.get(name)]


def append_env(env_path: Path, values: dict[str, str]) -> None:
    """Append ``KEY=value`` lines to ``env_path`` (creating it if needed)."""
    if not values:
        return
    block = "".join(f"{key}={value}\n" for key, value in values.items())
    prefix = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    env_path.write_text(prefix + block, encoding="utf-8")
