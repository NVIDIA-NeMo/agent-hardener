# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator: replay each generated request and judge refusal in parallel."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import httpx

from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.llm_telemetry import emit_agent_exchange
from agent_hardener.rate_limits import RateLimitError

from .judge import JudgeVerdict, judge_one
from .replay import replay_one
from .reporting import VerdictStatus, excerpt

if TYPE_CHECKING:
    from agent_hardener.agents.validators.replay_http import ReplayResult

    from ..models import GeneratedRequest
    from .config import ReplayJudgeConfig

logger = logging.getLogger(__name__)


async def run_all(
    requests: list[GeneratedRequest],
    config: ReplayJudgeConfig,
) -> list[dict[str, Any]]:
    """Replay every request and judge each in parallel (bounded by ``concurrency``).

    Returns one result dict per input request, in input order. Each result
    propagates ``label`` / ``persona`` from its :class:`GeneratedRequest`
    so the parent ``ValidatorReport`` can slice findings along those axes.
    """
    if not requests:
        return []

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        indexed = list(enumerate(requests, start=1))
        results = await gather_limited_ordered(
            indexed,
            config.concurrency,
            lambda item: _process_one(item[0], item[1], config, client),
        )
    return [cast("dict[str, Any]", result) for result in results]


async def _process_one(
    index: int,
    request: GeneratedRequest,
    config: ReplayJudgeConfig,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    started = perf_counter()
    base: dict[str, Any] = {
        "index": index,
        "tool": request.tool,
        "label": request.label,
        "persona": request.persona,
        "payload_excerpt": excerpt(request.payload, config.excerpt_chars),
    }

    try:
        replay_result = await replay_one(request.payload, config, client)
    except RateLimitError:
        raise
    except Exception as exc:
        return _finalize(base, _replay_failed_marker(), _error_verdict(f"replay failed: {_msg(exc)}", config), started)

    try:
        verdict = await judge_one(
            model_name=config.judge_model,
            base_url=config.judge_base_url,
            api_key=config.judge_api_key,
            tool=request.tool,
            payload=request.payload,
            response=excerpt(replay_result.text, config.judge_input_chars),
            max_tokens=config.judge_max_tokens,
            timeout=config.judge_timeout_seconds,
            temperature=config.judge_temperature,
        )
    except RateLimitError:
        raise
    except Exception as exc:
        return _finalize(
            base,
            _replay_marker(replay_result, config),
            _error_verdict(f"judge failed: {_msg(exc)}", config),
            started,
        )

    return _finalize(base, _replay_marker(replay_result, config), _verdict_dict(verdict, config), started)


def _finalize(
    base: dict[str, Any],
    replay_marker: dict[str, Any],
    verdict: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    result = {**base, "replay": replay_marker, "verdict": verdict}
    result["duration_seconds"] = round(perf_counter() - started, 6)
    logger.info(
        "smart benign row completed index=%s tool=%s label=%s status=%s duration_seconds=%.3f",
        result["index"],
        result["tool"],
        result["label"],
        verdict["status"],
        result["duration_seconds"],
    )
    # A refused benign request means the victim blocked a legitimate call — surface it as blocked.
    emit_agent_exchange(
        request=base.get("payload_excerpt", ""),
        response=(replay_marker or {}).get("response_excerpt", ""),
        label=str(verdict.get("status", "")),
        ok=verdict.get("status") != "error",
        blocked=bool(verdict.get("is_refused")),
    )
    return result


def _verdict_dict(verdict: JudgeVerdict, config: ReplayJudgeConfig) -> dict[str, Any]:
    decisive = verdict.confidence >= config.confidence_cutoff
    refused = verdict.is_refused and decisive
    status: VerdictStatus = "refused" if refused else "complied"
    return {
        "status": status,
        "is_refused": refused,
        "complied": not refused,
        "confidence": verdict.confidence,
        "reasoning": excerpt(verdict.reasoning, config.excerpt_chars),
    }


def _error_verdict(reason: str, config: ReplayJudgeConfig) -> dict[str, Any]:
    return {
        "status": "error",
        "is_refused": None,
        "complied": False,
        "confidence": 0.0,
        "reasoning": excerpt(reason, config.excerpt_chars),
    }


def _replay_marker(result: ReplayResult, config: ReplayJudgeConfig) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "status_code": result.status_code,
        "response_excerpt": excerpt(result.text, config.excerpt_chars),
    }


def _replay_failed_marker() -> dict[str, Any]:
    return {"ok": False, "status_code": None, "response_excerpt": ""}


def _msg(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
