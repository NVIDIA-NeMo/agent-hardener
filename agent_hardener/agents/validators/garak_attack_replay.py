# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Garak attack hits against a defended victim and validate per attack."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx

from agent_hardener.agents.validators.garak_replay import detectors, parsing, redaction, verdict
from agent_hardener.agents.validators.garak_replay.config import ReplayConfig, build_config, validator_kind
from agent_hardener.agents.validators.garak_replay.parsing import AttackType, GitHubIssueTarget
from agent_hardener.agents.validators.replay_http import ReplayResult
from agent_hardener.agents.validators.replay_http import replay_prompt as replay_http_prompt
from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.env import GARAK_REPO_PATH
from agent_hardener.garak_venv import GARAK_PYTHON_ENVVAR, garak_subprocess_env, resolve_garak_python
from agent_hardener.llm_telemetry import emit_agent_exchange
from agent_hardener.models import AgentConfig, AgentRunInput, ValidatorReport
from agent_hardener.rate_limits import RateLimitError, raise_for_rate_limit_response

__all__ = [
    "AttackType",
    "GitHubIssueTarget",
    "PersistentGarakWorker",
    "ReplayConfig",
    "ReplayResult",
    "detector_for",
    "run",
]

# The garak detector worker script, run by the dedicated garak-venv interpreter (garak is isolated
# in its own venv and cannot be imported in-process). See :mod:`agent_hardener.garak_venv`.
_DETECT_WORKER = Path(__file__).resolve().parent / "_garak_detect_worker.py"


logger = logging.getLogger(__name__)


async def run(request: AgentRunInput, agent: AgentConfig) -> ValidatorReport:
    """Replay supported Garak attack hits and return one result per hit."""
    kind = validator_kind(request, agent)
    config = build_config(request, agent)
    attack_results: list[dict[str, Any]] = []
    findings: list[str] = []

    supported_hits = list(parsing.iter_supported_hits(request.attacks))
    if not supported_hits:
        return ValidatorReport(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            kind=kind,
            summary="no supported Garak attack hits to replay",
            metadata={
                "total_attacks": 0,
                "blocked_count": 0,
                "not_blocked_count": 0,
                "error_count": 0,
                "attack_results": [],
                "concurrency": config.concurrency,
            },
        )

    detector_cache: dict[AttackType, Any] = {}
    worker = PersistentGarakWorker(
        repo_path=str(config.garak_repo_path) if config.garak_repo_path else os.getenv(GARAK_REPO_PATH),
        timeout_seconds=config.timeout_seconds,
    )
    try:
        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            indexed_hits = list(enumerate(supported_hits, start=1))
            hit_results = await gather_limited_ordered(
                indexed_hits,
                config.concurrency,
                lambda item: _validate_hit_timed(item[0], item[1], request, config, client, worker, detector_cache),
            )
            attack_results = [cast("dict[str, Any]", result) for result in hit_results]
            for aggregate_index, result in enumerate(attack_results, start=1):
                findings.append(verdict.format_finding(result, aggregate_index))
    finally:
        worker.close()

    blocked_count = sum(1 for result in attack_results if result["verdict"]["status"] == "blocked")
    not_blocked_count = sum(1 for result in attack_results if result["verdict"]["status"] == "not_blocked")
    error_count = sum(1 for result in attack_results if result["verdict"]["status"] == "error")
    ok = not_blocked_count == 0 and error_count == 0
    false_negatives = [
        result["prompt_excerpt"]
        for result in attack_results
        if result["verdict"]["status"] == "not_blocked" and result.get("prompt_excerpt")
    ]

    return ValidatorReport(
        agent_id=agent.agent_id,
        agent_name=agent.name,
        kind=kind,
        ok=ok,
        summary=f"blocked {blocked_count}/{len(attack_results)} original Garak attack hits",
        findings=findings,
        false_negatives=false_negatives,
        metadata={
            "total_attacks": len(attack_results),
            "blocked_count": blocked_count,
            "not_blocked_count": not_blocked_count,
            "error_count": error_count,
            "attack_results": attack_results,
            "concurrency": config.concurrency,
        },
    )


async def _validate_hit_timed(
    aggregate_index: int,
    hit_context: dict[str, Any],
    request: AgentRunInput,
    config: ReplayConfig,
    client: httpx.AsyncClient,
    worker: PersistentGarakWorker,
    detector_cache: dict[AttackType, Any],
) -> dict[str, Any]:
    started_at = perf_counter()
    result = await _validate_hit(
        hit_context=hit_context,
        aggregate_index=aggregate_index,
        request=request,
        config=config,
        client=client,
        worker=worker,
        detector_cache=detector_cache,
    )
    result["duration_seconds"] = round(perf_counter() - started_at, 6)
    logger.info(
        "attack validator hit completed index=%s attack_type=%s status=%s duration_seconds=%.3f",
        aggregate_index,
        result["attack_type"],
        result["verdict"]["status"],
        result["duration_seconds"],
    )
    hit_verdict = result.get("verdict") or {}
    emit_agent_exchange(
        request=result.get("prompt_excerpt", ""),
        response=(result.get("replay") or {}).get("response_excerpt", ""),
        label=str(hit_verdict.get("status", "")),
        ok=hit_verdict.get("status") != "error",
        blocked=bool(hit_verdict.get("blocked")),
    )
    return result


async def _validate_hit(
    *,
    hit_context: dict[str, Any],
    aggregate_index: int,
    request: AgentRunInput,
    config: ReplayConfig,
    client: httpx.AsyncClient,
    worker: PersistentGarakWorker,
    detector_cache: dict[AttackType, Any],
) -> dict[str, Any]:
    hit = hit_context["hit"]
    attack_type = cast("AttackType", hit_context["attack_type"])
    try:
        if attack_type == "direct_prompt_injection":
            return await _validate_direct_hit(
                hit_context=hit_context,
                aggregate_index=aggregate_index,
                request=request,
                config=config,
                client=client,
                worker=worker,
                detector_cache=detector_cache,
            )
        return await _validate_indirect_hit(
            hit_context=hit_context,
            aggregate_index=aggregate_index,
            request=request,
            config=config,
            client=client,
            worker=worker,
            detector_cache=detector_cache,
        )
    except RateLimitError:
        raise
    except Exception as exc:
        return _error_result(
            hit_context=hit_context,
            aggregate_index=aggregate_index,
            config=config,
            error=str(exc) or exc.__class__.__name__,
            prompt=parsing.extract_prompt(hit),
        )


async def _validate_direct_hit(
    *,
    hit_context: dict[str, Any],
    aggregate_index: int,
    request: AgentRunInput,
    config: ReplayConfig,
    client: httpx.AsyncClient,
    worker: PersistentGarakWorker,
    detector_cache: dict[AttackType, Any],
) -> dict[str, Any]:
    hit = hit_context["hit"]
    prompt = parsing.extract_prompt(hit)
    target_tool = parsing.parse_target_tool(prompt)
    replay = await _replay_prompt(prompt, request, config, client)
    result = _base_result(hit_context, aggregate_index, config, prompt, replay)
    result["direct_context"] = {"target_tool": target_tool, "triggers": hit.get("triggers")}

    # A defended victim that returns no content blocked the attack — no need to spend the detector call.
    if not replay.text.strip():
        result["verdict"] = verdict.blocked_without_detector()
        return result

    detector = _detector_for("direct_prompt_injection", config, worker, detector_cache)
    is_success, confidence, reasoning = await asyncio.to_thread(
        detector.verify,
        output_text=replay.text,
        target_tool=target_tool,
        vulnerability_info=str(hit.get("goal") or ""),
        attack_prompt=prompt,
    )
    result["verdict"] = verdict.verdict_from_detector(is_success, confidence, reasoning, config)
    return result


async def _validate_indirect_hit(
    *,
    hit_context: dict[str, Any],
    aggregate_index: int,
    request: AgentRunInput,
    config: ReplayConfig,
    client: httpx.AsyncClient,
    worker: PersistentGarakWorker,
    detector_cache: dict[AttackType, Any],
) -> dict[str, Any]:
    hit = hit_context["hit"]
    triggers = hit.get("triggers") if isinstance(hit.get("triggers"), dict) else {}
    prompt = parsing.extract_indirect_prompt(hit)
    injected_payload = str(triggers.get("injected_payload") or "")
    exploit_goal = str(triggers.get("exploit_goal") or "")
    injection_location = str(triggers.get("injection_location") or "")
    watched_targets = parsing.watched_github_targets(injection_location, injected_payload)

    comments_before, before_errors = await _fetch_issue_comments(watched_targets, config, client)
    replay = await _replay_prompt(prompt, request, config, client)
    comments_after, after_errors = await _fetch_issue_comments(watched_targets, config, client)
    new_comments = _new_comments(comments_before, comments_after)

    detector = _detector_for("indirect_prompt_injection", config, worker, detector_cache)
    is_success, confidence, reasoning = await asyncio.to_thread(
        detector.verify,
        output_text=replay.text,
        exploit_goal=exploit_goal,
        injected_payload=injected_payload,
        trigger_query=prompt,
        github_comments=new_comments,
    )
    verdict_dict = verdict.verdict_from_detector(is_success, confidence, reasoning, config)

    result = _base_result(hit_context, aggregate_index, config, prompt, replay)
    result["indirect_context"] = {
        "exploit_goal": exploit_goal,
        "injection_location": injection_location,
        "watched_side_effect_targets": [target.label for target in watched_targets],
        "old_garak_comments_count": len(triggers.get("github_comments") or []),
        "github_comments_checked": bool(watched_targets) and config.github_comments_enabled,
        "comments_before_count": sum(len(comments) for comments in comments_before.values()),
        "comments_after_count": sum(len(comments) for comments in comments_after.values()),
        "new_github_comments_count": len(new_comments),
        "comment_fetch_errors": redaction.redacted_errors([*before_errors, *after_errors], config),
        "context_complete": bool(injected_payload and exploit_goal and injection_location),
    }
    result["verdict"] = verdict_dict
    return result


def _base_result(
    hit_context: dict[str, Any],
    aggregate_index: int,
    config: ReplayConfig,
    prompt: str,
    replay: ReplayResult,
) -> dict[str, Any]:
    hit = hit_context["hit"]
    result: dict[str, Any] = {
        "attack_id": verdict.attack_id(hit_context, aggregate_index),
        "attack_type": hit_context["attack_type"],
        "attacker_name": hit_context["attacker_name"],
        "source": hit.get("source"),
        "hitlog_line": hit_context["record_index"] + 1,
        "probe": hit.get("probe"),
        "detector": hit.get("detector"),
        "garak_score": hit.get("score"),
        "goal": redaction.excerpt(str(hit.get("goal") or ""), config),
        "prompt_excerpt": redaction.excerpt(prompt, config),
        "replay": {
            "ok": replay.ok,
            "status_code": replay.status_code,
            "response_excerpt": redaction.excerpt(replay.text, config),
        },
    }
    if config.store_full_replay:
        result["replay"]["response"] = redaction.redact(replay.text, config)
    return result


def _error_result(
    *,
    hit_context: dict[str, Any],
    aggregate_index: int,
    config: ReplayConfig,
    error: str,
    prompt: str,
) -> dict[str, Any]:
    hit = hit_context["hit"]
    return {
        "attack_id": verdict.attack_id(hit_context, aggregate_index),
        "attack_type": hit_context["attack_type"],
        "attacker_name": hit_context["attacker_name"],
        "source": hit.get("source"),
        "hitlog_line": hit_context["record_index"] + 1,
        "probe": hit.get("probe"),
        "detector": hit.get("detector"),
        "garak_score": hit.get("score"),
        "goal": redaction.excerpt(str(hit.get("goal") or ""), config),
        "prompt_excerpt": redaction.excerpt(prompt, config),
        "replay": {"ok": False, "status_code": None, "response_excerpt": ""},
        "verdict": {
            "status": "error",
            "blocked": False,
            "attack_succeeded": None,
            "confidence": 0.0,
            "reasoning": redaction.excerpt(error, config),
        },
        "error": redaction.excerpt(error, config),
    }


async def _replay_prompt(
    prompt: str,
    request: AgentRunInput,
    config: ReplayConfig,
    client: httpx.AsyncClient,
) -> ReplayResult:
    del request
    return await replay_http_prompt(
        prompt,
        config,
        client,
        rate_limit_source="attack validator victim replay",
        error_excerpt=lambda text: redaction.excerpt(text, config),
    )


async def _fetch_issue_comments(
    targets: list[GitHubIssueTarget],
    config: ReplayConfig,
    client: httpx.AsyncClient,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    if not targets or not config.github_comments_enabled:
        return {}, []

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Agent-Hardener",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if config.github_token:
        headers["Authorization"] = f"Bearer {config.github_token}"

    comments_by_target: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for target in targets:
        url = f"https://api.github.com/repos/{target.owner}/{target.repo}/issues/{target.issue_number}/comments"
        try:
            response = await client.get(url, headers=headers, timeout=config.timeout_seconds)
            raise_for_rate_limit_response(response, source=f"GitHub comments {target.label}")
            response.raise_for_status()
            comments_by_target[target.label] = [_normalize_comment(target, comment) for comment in response.json()]
        except RateLimitError:
            raise
        except Exception as exc:
            comments_by_target[target.label] = []
            errors.append({"target": target.label, "error": str(exc) or exc.__class__.__name__})
    return comments_by_target, errors


def _normalize_comment(target: GitHubIssueTarget, comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": target.label,
        "id": comment.get("id"),
        "html_url": comment.get("html_url"),
        "user": (comment.get("user") or {}).get("login", "unknown"),
        "body": comment.get("body", ""),
        "created_at": comment.get("created_at", ""),
    }


def _new_comments(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    new: list[dict[str, Any]] = []
    for target, after_comments in after.items():
        before_keys = {_comment_key(comment) for comment in before.get(target, [])}
        for comment in after_comments:
            if _comment_key(comment) not in before_keys:
                new.append(comment)
    return new


def _comment_key(comment: dict[str, Any]) -> tuple[Any, ...]:
    return (
        comment.get("id"),
        comment.get("html_url"),
        comment.get("user"),
        comment.get("body"),
        comment.get("created_at"),
    )


def detector_for(
    attack_type: AttackType,
    config: ReplayConfig,
    worker: PersistentGarakWorker,
    detector_cache: dict[AttackType, Any],
) -> Any:
    if attack_type not in detector_cache:
        detector_cache[attack_type] = _WorkerDetector(
            worker=worker,
            attack_type=attack_type,
            config_root=detectors.detector_config_root(attack_type, config),
        )
    return detector_cache[attack_type]


@dataclass(frozen=True)
class _WorkerDetector:
    """Detector proxy: each ``verify`` evaluates one hit via the shared persistent garak worker.

    garak is isolated in its own venv (it pins torch/litellm that conflict with agent-hardener), so the
    detector cannot be imported in-process. The proxy delegates to :class:`PersistentGarakWorker`,
    which runs :data:`_DETECT_WORKER` under the garak-venv interpreter and imports garak once per run.
    """

    worker: PersistentGarakWorker
    attack_type: AttackType
    config_root: dict[str, Any]

    def verify(self, **verify_kwargs: Any) -> tuple[bool, float, str]:
        return self.worker.verify(self.attack_type, self.config_root, verify_kwargs)


class PersistentGarakWorker:
    """A long-lived garak-venv subprocess that services many ``verify`` requests.

    garak/torch is imported once (on the worker's first request) instead of once per hit — this is the
    difference that made attack replay slow. The worker speaks the newline-delimited JSON protocol of
    :data:`_DETECT_WORKER`. Access is serialized by a lock (``verify`` is called from
    ``asyncio.to_thread``, so several threads reach it); a per-request timeout is enforced with
    ``select`` on the worker's stdout, and a timeout / unexpected EOF tears the worker down so the next
    request restarts it. A per-hit worker error still surfaces as an ``error`` verdict via
    :func:`_validate_hit`.
    """

    def __init__(self, *, repo_path: str | None, timeout_seconds: float) -> None:
        self._repo_path = repo_path
        self._timeout_seconds = timeout_seconds
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def verify(
        self,
        attack_type: AttackType,
        config_root: dict[str, Any],
        verify_kwargs: dict[str, Any],
    ) -> tuple[bool, float, str]:
        request = json.dumps(
            {
                "attack_type": attack_type,
                "config_root": config_root,
                "verify_kwargs": verify_kwargs,
                "repo_path": self._repo_path,
            }
        )
        with self._lock:
            proc = self._ensure()
            try:
                proc.stdin.write(request + "\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, ValueError) as exc:
                self.close()
                raise RuntimeError(f"garak detector worker stdin unavailable: {exc}") from exc
            line = self._read_line(proc)
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "garak detector worker failed"))
        return bool(response["is_success"]), float(response["confidence"]), str(response["reasoning"])

    def _ensure(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        garak_python = resolve_garak_python()
        if not Path(garak_python).exists():
            msg = (
                f"garak interpreter not found at {garak_python}. Provision the dedicated garak venv "
                f"(`agent-hardener setup`) or set ${GARAK_PYTHON_ENVVAR} to an existing interpreter that "
                "has garak installed."
            )
            raise RuntimeError(msg)
        self._proc = subprocess.Popen(  # noqa: S603 — fixed interpreter + bundled worker script, no shell
            [garak_python, str(_DETECT_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=garak_subprocess_env(),
        )
        return self._proc

    def _read_line(self, proc: subprocess.Popen[str]) -> str:
        ready, _, _ = select.select([proc.stdout], [], [], self._timeout_seconds)
        if not ready:
            self.close()
            raise RuntimeError(f"garak detector worker timed out after {self._timeout_seconds}s")
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            self.close()
            raise RuntimeError("garak detector worker exited before returning a verdict")
        return line

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# Private aliases kept so every in-repo caller and monkeypatch target works unchanged. Both names
# resolve to the same object, and internal code looks the global up at call time, so patching
# ``_detector_for`` in a test still redirects the real call.
_detector_for = detector_for
_PersistentGarakWorker = PersistentGarakWorker
