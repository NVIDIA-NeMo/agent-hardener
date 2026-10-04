# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clone the repo (shallow), scan NAT YAMLs, capture HEAD SHA + README text.

The tmp directory lifetime is bounded by this single node: the
``TemporaryDirectory`` context exits before the function returns, so even
if downstream nodes fail there is no directory to leak. All data the next
node needs (``declared_tools``, ``readme_text``, ``head_sha``) is captured in
the state before cleanup.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from agent_hardener.agents.validators.smart_benign.models import ToolSpec

if TYPE_CHECKING:
    from ..state import GitHubAnalyzerState

logger = logging.getLogger(__name__)

README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README")
MAX_README_BYTES = 50_000
MAX_YAML_BYTES = 200_000
#: A declared tool is near-certain — it is read from config, not inferred from prose.
DECLARED_TOOL_CONFIDENCE = 0.95


async def clone_and_scan(state: GitHubAnalyzerState) -> dict[str, object]:
    """Run git clone in a tmp dir, scan YAMLs + README, then tear the dir down."""
    if not state.repo_url:
        return {"source_note": "github_analyzer skipped (no github_url configured)"}

    try:
        return await asyncio.to_thread(_clone_and_scan_sync, state)
    except Exception as exc:
        logger.exception("github_analyzer clone_and_scan failed")
        return {
            "source_note": f"github_analyzer failed: {exc}",
            "errors": [f"github_analyzer.clone_and_scan: {exc}"],
        }


def _clone_and_scan_sync(state: GitHubAnalyzerState) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="github_analyzer_") as tmp:
        clone_dir = Path(tmp) / "repo"
        _git_clone(state, clone_dir)
        head_sha = _head_sha(clone_dir)
        declared_tools, yaml_files = _scan_declared_tools(clone_dir)
        readme_text = _read_readme(clone_dir)
    logger.info(
        "github_analyzer scanned HEAD=%s files=%d declared_tools=%d readme=%s",
        head_sha or "?",
        yaml_files,
        len(declared_tools),
        "yes" if readme_text else "no",
    )
    return {
        "head_sha": head_sha,
        "declared_tools": declared_tools,
        "yaml_files_scanned": yaml_files,
        "readme_text": readme_text,
    }


def _git_clone(state: GitHubAnalyzerState, target: Path) -> None:
    result = subprocess.run(  # noqa: S603 - git binary on PATH, args are not from untrusted source
        ["git", "clone", "--depth", "1", "--single-branch", state.repo_url, str(target)],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=state.clone_timeout_seconds,
        check=False,
        env=_clone_env(state.gh_token),
    )
    if result.returncode != 0:
        # Never surface raw git stderr: on auth failure git can echo the credentialed URL back.
        msg = f"git clone failed (exit={result.returncode})"
        raise RuntimeError(msg)


def _clone_env(token: str | None) -> dict[str, str]:
    """Pass the token as an HTTP auth header via env-based git config.

    Keeps the credential out of the process argv and the remote URL, so it never
    lands in a ``ps`` listing or in git's error output.
    """
    env = os.environ.copy()
    if token:
        auth = base64.b64encode(f"oauth2:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {auth}"
    return env


def _head_sha(repo: Path) -> str | None:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _scan_declared_tools(repo: Path) -> tuple[list[ToolSpec], int]:
    """Parse the repo's config files for tools the agent declares.

    Static discovery, and the only kind available here: the benign suite has to be generated before
    the victim is ever invoked, so there is no trajectory to read tools from yet.

    Under NAT there was one shape to look for — a ``functions:`` block in the workflow YAML. A
    Relay-connected agent declares tools in whatever format its own stack uses, so the shapes that
    can be recognised generically are the portable ones:

    * MCP servers — ``mcp.servers.<name>.allowed_tools``
    * Fabric/Platform tool config — ``tools.definitions.<name>`` and ``tools.enabled``
    """
    seen: dict[str, ToolSpec] = {}
    files_scanned = 0
    for path in (*repo.rglob("*.yaml"), *repo.rglob("*.yml")):
        # Skip symlinks for the same reason _read_readme does: a cloned repo is untrusted, and
        # stat() follows the link, so a symlink to a special file reports a passing size and then
        # reads without end.
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_YAML_BYTES:
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, yaml.YAMLError):
            continue
        files_scanned += 1
        for tool in _declared_tools(data):
            seen.setdefault(tool.name, tool)
    return list(seen.values()), files_scanned


def _declared_tools(data: Any) -> list[ToolSpec]:
    """Every tool declaration recognised in one parsed config document."""
    if not isinstance(data, dict):
        return []
    tools: list[ToolSpec] = []

    definitions = _nested(data, "tools", "definitions")
    for name, spec in definitions.items():
        kind = str(spec.get("kind") or spec.get("ref") or name) if isinstance(spec, dict) else name
        tools.append(_declared_tool(name, f"{kind} tool"))

    for name in _as_str_list(_nested_value(data, "tools", "enabled")):
        tools.append(_declared_tool(name, f"{name} tool"))

    for server, spec in _nested(data, "mcp", "servers").items():
        if not isinstance(spec, dict):
            continue
        for name in _as_str_list(spec.get("allowed_tools")):
            tools.append(_declared_tool(name, f"{name} tool exposed by the {server} MCP server"))
    return tools


def _declared_tool(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        source="github",
        confidence=DECLARED_TOOL_CONFIDENCE,
    )


def _nested(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    value = _nested_value(data, *keys)
    return {k: v for k, v in value.items() if isinstance(k, str)} if isinstance(value, dict) else {}


def _nested_value(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def _read_readme(repo: Path) -> str | None:
    for candidate in README_CANDIDATES:
        path = repo / candidate
        # Skip symlinks: a cloned repo is untrusted, so a symlinked README could point at a host
        # file outside the clone. Bound the read so an oversized README can't exhaust memory.
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_README_BYTES:
                with path.open("rb") as handle:
                    blob = handle.read(MAX_README_BYTES)
                return blob.decode("utf-8", errors="replace") + "\n\n...<truncated>"
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return None
