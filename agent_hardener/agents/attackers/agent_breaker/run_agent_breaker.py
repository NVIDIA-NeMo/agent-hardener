# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run garak's AgentBreaker probe directly via the garak CLI (no REST app).

The attacker renders a per-run garak config from the manifest's target endpoint (overlaying
an optional user-edited ``garak-scan.yaml`` scaffold), spawns ``garak --config`` as a
subprocess, then reads the hitlog garak writes under ``report_dir``. The victim endpoint can be
overridden via ``target_uri`` (whole URL) or ``target_port`` (swap the port only).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from agent_hardener.agents.attackers.agent_breaker.config import (
    OVERRIDE_KEYS,
    apply_runtime_fields,
    build_agent_breaker_config,
)
from agent_hardener.agents.attackers.hits import normalize_hit_record
from agent_hardener.env import garak_command
from agent_hardener.garak_venv import GARAK_PYTHON_ENVVAR, garak_subprocess_env, resolve_garak_python
from agent_hardener.models import ARTIFACT_DIR_KEY, AgentConfig, AgentRunInput, Artifact, AttackRecord

DEFAULT_CONFIG_PATH = "garak-scan.yaml"
DEFAULT_REPORT_DIR = ".agent-hardener/garak_runs"
DEFAULT_REPORT_PREFIX = "agent-breaker"
DEFAULT_TARGET_URI = "http://127.0.0.1:8000/v1/chat/completions"

GarakRunner = Callable[[list[str], Path, float | None, dict[str, str] | None], Awaitable[int]]


def _garak_command() -> list[str]:
    """The base garak invocation.

    ``GARAK_COMMAND`` (via ``agent_hardener.env.garak_command``) is an explicit full override
    (e.g. ``garak`` or ``python -m garak``). Otherwise garak is run from its dedicated venv
    interpreter (see :func:`agent_hardener.garak_venv.resolve_garak_python`); a missing interpreter
    raises with instructions to provision it.
    """
    raw = garak_command()
    if raw:
        return raw.split()
    garak_python = resolve_garak_python()
    if not Path(garak_python).exists():
        raise FileNotFoundError(
            f"garak interpreter not found at {garak_python}. Provision the dedicated garak venv "
            f"(`agent-hardener setup`) or set ${GARAK_PYTHON_ENVVAR} to an existing interpreter that "
            "has garak installed."
        )
    return [garak_python, "-m", "garak"]


def resolve_target_uri(
    base_url: str | None,
    *,
    target_uri: str | None = None,
    target_port: int | None = None,
) -> str:
    """Resolve the victim endpoint garak attacks.

    A full ``target_uri`` override wins; otherwise ``target_port`` swaps the port on the
    manifest-derived ``base_url``; otherwise the ``base_url`` is used as-is.
    """
    if target_uri:
        return target_uri
    url = base_url or DEFAULT_TARGET_URI
    if target_port is None:
        return url
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    netloc = f"{host}:{target_port}"
    if parts.username:
        auth = parts.username + (f":{parts.password}" if parts.password else "")
        netloc = f"{auth}@{netloc}"
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment))


async def _spawn_garak(
    command: list[str], log_path: Path, timeout_s: float | None, env: dict[str, str] | None = None
) -> int:
    """Run garak as a subprocess, streaming output to ``log_path``; return its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=log_file, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
    return process.returncode or 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _latest(report_dir: Path, prefix: str, suffix: str) -> Path | None:
    """Newest ``<prefix>*<suffix>`` file garak produced under ``report_dir``."""
    candidates = sorted(report_dir.glob(f"{prefix}*{suffix}"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


@dataclass
class ScanResult:
    """Outcome of one AgentBreaker garak scan."""

    return_code: int
    hits: list[dict[str, Any]]
    config_path: str
    report_path: str | None = None
    hitlog_path: str | None = None


async def run_agent_breaker_scan(
    *,
    target_uri: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    report_dir: str = DEFAULT_REPORT_DIR,
    report_prefix: str = DEFAULT_REPORT_PREFIX,
    overrides: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    garak_command: list[str] | None = None,
    runner: GarakRunner = _spawn_garak,
) -> ScanResult:
    """Render the garak config, run the AgentBreaker probe, and collect its hits.

    Args:
        target_uri: Victim chat-completions endpoint the probe attacks.
        config_path: Optional scaffold YAML to overlay the resolved fields onto; built from
            defaults when absent.
        report_dir: Directory garak writes its report/hitlog into (created if missing).
        report_prefix: Filename prefix for this run's garak artifacts.
        overrides: Per-setting overrides forwarded to the config builder.
        timeout_s: Max seconds to wait for garak; ``None`` waits indefinitely.
        garak_command: Base garak invocation; defaults to ``GARAK_COMMAND`` or the dedicated
            garak venv interpreter (see ``_garak_command`` / ``AGENT_HARDENER_GARAK_PYTHON``).
        runner: Injectable subprocess runner (return code) for testing.

    Returns:
        A :class:`ScanResult` with the exit code, hitlog rows, and artifact paths.
    """
    report_dir_path = Path(report_dir).resolve()
    report_dir_path.mkdir(parents=True, exist_ok=True)

    scaffold = Path(config_path)
    if scaffold.is_file():
        base = yaml.safe_load(scaffold.read_text(encoding="utf-8")) or {}
    else:
        base = build_agent_breaker_config(
            target_uri=target_uri, report_dir=report_dir, report_prefix=report_prefix, overrides=overrides
        )
    resolved = apply_runtime_fields(
        base, target_uri=target_uri, report_dir=str(report_dir_path), report_prefix=report_prefix
    )

    # Co-locate the resolved config + garak log with the report/hitlog under report_dir. When the
    # orchestrator anchors report_dir to the run's iteration dir these become run-scoped; using a
    # fixed name is safe because each run gets its own dir (no cross-run overwrite).
    resolved_path = report_dir_path / "garak-agent-breaker.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    command = [*(garak_command or _garak_command()), "--config", str(resolved_path)]
    return_code = await runner(command, report_dir_path / "garak-agent-breaker.log", timeout_s, garak_subprocess_env())

    hitlog = _latest(report_dir_path, report_prefix, ".hitlog.jsonl")
    report = _latest(report_dir_path, report_prefix, ".report.jsonl")
    return ScanResult(
        return_code=return_code,
        hits=_read_jsonl(hitlog) if hitlog else [],
        config_path=str(resolved_path),
        report_path=str(report) if report else None,
        hitlog_path=str(hitlog) if hitlog else None,
    )


async def run(request: AgentRunInput, agent: AgentConfig) -> AttackRecord:
    """Run AgentBreaker as an Agent Hardener attacker."""
    config = agent.config
    overrides = {key: config[key] for key in OVERRIDE_KEYS if config.get(key) is not None}
    # Precedence: an explicit report_dir override in the agent config wins; otherwise write to the ready
    # run-scoped dir the stage injected; fall back to the shared default only for standalone/CLI use where
    # no run dir is threaded in. No subdir math here — the authority already resolved the directory.
    report_dir = config.get("report_dir") or request.context.get(ARTIFACT_DIR_KEY) or DEFAULT_REPORT_DIR
    target_uri = resolve_target_uri(
        request.target.base_url,
        target_uri=str(config["target_uri"]) if config.get("target_uri") else None,
        target_port=int(config["target_port"]) if config.get("target_port") is not None else None,
    )
    result = await run_agent_breaker_scan(
        target_uri=target_uri,
        config_path=str(config.get("config_path", DEFAULT_CONFIG_PATH)),
        report_dir=str(report_dir),
        report_prefix=str(config.get("report_prefix", DEFAULT_REPORT_PREFIX)),
        overrides=overrides,
        timeout_s=float(config["timeout_s"]) if config.get("timeout_s") is not None else None,
    )
    hits = _coerce_hits(result.hits)
    hit_count = len(hits)
    # Treat a non-zero exit as ok if garak still produced hits — a mid-run crash
    # (e.g. victim returning 422 on a later attempt) should not discard earlier findings.
    ok = bool(hits) or result.return_code == 0
    artifacts = []
    if result.report_path:
        artifacts.append(Artifact(type="garak_report", path=result.report_path))
    if result.hitlog_path:
        artifacts.append(Artifact(type="garak_hitlog", path=result.hitlog_path))
    return AttackRecord(
        agent_id=agent.agent_id,
        agent_name=agent.name,
        ok=ok,
        summary=_summary(hit_count, partial=result.return_code != 0),
        records=hits,
        artifacts=artifacts,
        metadata={
            "garak_return_code": result.return_code,
            "garak_report_path": result.report_path,
            "garak_config_path": result.config_path,
            "target": request.target.name,
            "target_uri": target_uri,
        },
    )


def _summary(hit_count: int, *, partial: bool = False) -> str:
    suffix = " (garak exited early — partial run)" if partial else ""
    if hit_count == 0:
        return f"AgentBreaker completed without reported hits.{suffix}"
    return (
        f"AgentBreaker reported {hit_count} attack hit(s), "
        f"indicating a vulnerability or exploit path in the victim agent behavior.{suffix}"
    )


def _coerce_hits(raw_hits: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_hits, list):
        return [{"source": "garak-agent-breaker", "raw_hits": raw_hits}]
    return [normalize_hit_record(hit, index, "garak-agent-breaker") for index, hit in enumerate(raw_hits)]
