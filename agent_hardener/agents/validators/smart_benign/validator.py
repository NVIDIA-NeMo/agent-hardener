# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level entries: ``synthesize`` (pre-flight DAG) and ``run`` (in-loop replay).

Synthesis and replay are split so the expensive synth DAG runs once per target
(pre-flight, with input-hash cache reuse) while the orchestrator's per-attempt
``run`` only replays the pre-generated ``requests.csv`` against the (mitigated)
victim. See ``CLAUDE.md`` in this package for the full contract.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from agent_hardener.concurrency import agent_concurrency
from agent_hardener.env import INFERENCE_API_KEY, inference_api_key
from agent_hardener.events import EventType
from agent_hardener.loggers import emit_event
from agent_hardener.models import AgentConfig, AgentRunInput, ValidatorReport

from .config import defaults
from .driver import drive_synth
from .graph import quiet_warnings
from .models import GeneratedRequest
from .replay_judge import ReplayJudgeConfig, format_finding, run_all, stats_key
from .state import SynthInputs, SynthState
from .subgraphs.profile_synthesizer import compute_input_hash
from .tracing import make_langfuse_callbacks

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .driver import AnswerProvider
    from .replay_judge.config import ReplayMode

REQUESTS_CSV = "requests.csv"
INPUT_HASH_FILE = "input_hash.txt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthResult:
    """Outcome of a pre-flight :func:`synthesize` call."""

    artifact_dir: Path
    request_count: int
    cached: bool
    errors: list[str] = field(default_factory=list)

    @property
    def viable(self) -> bool:
        """True when synthesis produced at least one replayable request."""
        return self.request_count > 0


async def run(request: AgentRunInput, agent: AgentConfig) -> ValidatorReport:
    """Replay the pre-synthesized benign suite against the victim and judge refusal.

    Synthesis happens out-of-loop (see :func:`synthesize`); this loads the cached
    ``requests.csv`` and replays it. If no suite exists yet, it hard-fails with
    guidance rather than running the expensive synth DAG inside the retry loop.
    """
    started = perf_counter()
    log_fields: dict[str, Any] = {
        "agent_name": agent.name,
        "agent_id": agent.agent_id,
        "round_id": request.round_id,
        "target": request.target.name,
        "iteration": request.iteration,
    }
    logger.info("smart_benign run starting", extra=log_fields)

    artifact_dir = _resolve_artifact_dir(request, agent)
    csv_path = artifact_dir / REQUESTS_CSV
    if not csv_path.is_file():
        return _failure_report(
            agent,
            started,
            f"no synthesized benign suite at {csv_path}; run synthesis first "
            "('agent-hardener synth-benign -c <config>' or the 'agent-hardener run' pre-flight phase)",
        )

    requests = load_requests(csv_path)
    if not requests:
        return _empty_report(agent, started, artifact_dir, reason=f"{csv_path} is empty")

    cfg = _build_config(request, agent)
    with quiet_warnings():
        row_results = await run_all(requests, cfg)
    return _build_report(agent, artifact_dir, row_results, started)


def cached_suite_count(request: AgentRunInput, agent: AgentConfig) -> int | None:
    """Request count of a reusable cached suite for these inputs, else ``None``.

    Lets callers detect an existing suite and offer to re-synthesize before
    :func:`synthesize` silently reuses it (pass ``force=True`` to override).
    """
    inputs = _build_inputs(request, agent)
    artifact_dir = _resolve_artifact_dir(request, agent)
    if _cache_hit(artifact_dir, inputs):
        return len(load_requests(artifact_dir / REQUESTS_CSV))
    return None


async def synthesize(
    request: AgentRunInput,
    agent: AgentConfig,
    *,
    answer_provider: AnswerProvider,
    force: bool = False,
    trust_cached_suite: bool = False,
    source_suite: Path | None = None,
) -> SynthResult:
    """Run the synth DAG once and persist the benign suite; reuse on input-hash match.

    Called by the orchestrator pre-flight phase (and reusable elsewhere); the
    in-loop :func:`run` only consumes its output. An empty result (no tools
    discovered — e.g. the victim was unreachable) is **not cached**: the input
    hash is invalidated so the next run re-synthesizes instead of freezing the
    failure. Returns a :class:`SynthResult` describing what happened.

    Args:
        request: The validator run input (target + context).
        agent: The benign-validator agent config.
        answer_provider: Supplies interview answers when the synth graph interrupts (terminal or HTTP).
        force: Re-run the synth DAG even when a cached suite matches the inputs.
        trust_cached_suite: Reuse a non-empty on-disk ``requests.csv`` as-is, skipping the DAG even on an
            input-hash miss — lets an out-of-band (e.g. manifest-owned) suite replay without re-synthesizing.
        source_suite: An explicit ``requests.csv`` to use as-is. Seeded into the target's artifact dir and
            replayed without synthesis (``run --benign-suite <path>``). Takes precedence over everything.
    """
    inputs = _build_inputs(request, agent)
    artifact_dir = _resolve_artifact_dir(request, agent)
    csv_file = artifact_dir / REQUESTS_CSV
    # Explicit suite file wins: agent-hardener owns the destination path, so callers just hand over a CSV.
    if source_suite is not None:
        return _seed_source_suite(artifact_dir, csv_file, source_suite)
    reused = _reuse_cached(artifact_dir, csv_file, inputs, force=force, trust_cached_suite=trust_cached_suite)
    if reused is not None:
        return reused

    # No reusable suite → run the synth DAG.
    d = defaults()
    max_questions = int(agent.config.get("max_interview_questions") or d.pipeline.max_interview_questions)
    max_rounds = int(agent.config.get("max_interview_rounds") or d.pipeline.max_interview_rounds)
    final = await _run_synth_dag(
        inputs,
        artifact_dir,
        answer_provider=answer_provider,
        interactive=_is_interactive(),
        max_interview_questions=max_questions,
        max_interview_rounds=max_rounds,
        round_id=request.round_id,
        target_name=request.target.name,
    )
    count = len(final.requests)
    if count == 0:
        reason = "; ".join(final.errors[:3]) or "no tools discovered"
        logger.warning(
            "smart_benign synth produced an EMPTY suite for %s (%s); not caching — will re-synthesize next run",
            request.target.name,
            reason,
        )
        _invalidate_cache(artifact_dir)
    return SynthResult(artifact_dir=artifact_dir, request_count=count, cached=False, errors=list(final.errors))


def _seed_source_suite(artifact_dir: Path, csv_file: Path, source_suite: Path) -> SynthResult:
    """Seed an explicitly supplied suite into the target's artifact dir and use it as-is (no synthesis)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if source_suite.resolve() != csv_file.resolve():  # a suite already at the destination needs no copy
        shutil.copyfile(source_suite, csv_file)
    requests = load_requests(csv_file)
    logger.info("smart_benign using supplied suite %s (%d request(s))", source_suite, len(requests))
    return SynthResult(artifact_dir=artifact_dir, request_count=len(requests), cached=True)


def _reuse_cached(
    artifact_dir: Path, csv_file: Path, inputs: SynthInputs, *, force: bool, trust_cached_suite: bool
) -> SynthResult | None:
    """Reuse an on-disk suite without running the DAG, or ``None`` when the DAG must run.

    Two reuse paths: trust any non-empty on-disk suite (``trust_cached_suite``), or reuse only on an
    input-hash match. ``force`` bypasses both.
    """
    if force:
        return None
    if trust_cached_suite and csv_file.is_file() and (requests := load_requests(csv_file)):
        logger.info("smart_benign trusting on-disk suite; skipping synthesis (%d request(s))", len(requests))
        return SynthResult(artifact_dir=artifact_dir, request_count=len(requests), cached=True)
    if _cache_hit(artifact_dir, inputs):
        count = len(load_requests(artifact_dir / REQUESTS_CSV))
        logger.info("smart_benign synth cache hit; reusing %s (%d request(s))", artifact_dir, count)
        return SynthResult(artifact_dir=artifact_dir, request_count=count, cached=True)
    return None


def _invalidate_cache(artifact_dir: Path) -> None:
    """Drop the cache key so an empty/failed synth is retried on the next run."""
    (artifact_dir / INPUT_HASH_FILE).unlink(missing_ok=True)


# Human-readable progress line per synth-DAG node, emitted as each node completes so an
# operator can see the recon advancing rather than staring at a silent prompt. Keys are the
# node names registered in ``graph.py``; nodes absent here (none currently) stream nothing.
_PROGRESS_LABELS: dict[str, str] = {
    "nl_parser": "recon: parsed capabilities from the description",
    "github_analyzer": "recon: analyzed the GitHub repository",
    "api_prober": "recon: probed the victim endpoint for tools",
    "profile_synthesizer": "recon: merged the victim capability profile",
    "gap_detector": "recon: checked the profile for gaps",
    "interviewer": "recon: recorded an interview answer",
    "request_generator": "recon: generated the benign request suite",
    "critic": "recon: filtered and deduplicated requests",
    "profile_writer": "recon: wrote profile.json + requests.csv",
}


async def _run_synth_dag(
    inputs: SynthInputs,
    artifact_dir: Path,
    *,
    answer_provider: AnswerProvider,
    interactive: bool,
    max_interview_questions: int,
    max_interview_rounds: int,
    round_id: str = "",
    target_name: str = "",
) -> SynthState:
    """Run the compiled synth graph and return the validated final state.

    Streams per-node updates so a human-readable recon progress line is logged as each DAG node
    completes — so an operator watching the run sees the recon advancing rather than a silent
    prompt. Progress goes through the module logger, so it reaches the run-log bus (and any
    subscriber) like every other ``agent_hardener`` log line.
    """
    state = SynthState.from_inputs(
        inputs,
        interactive=interactive,
        artifact_dir=artifact_dir,
        max_interview_questions=max_interview_questions,
        max_interview_rounds=max_interview_rounds,
    )
    tracing = make_langfuse_callbacks(round_id=round_id, target_name=target_name)
    lf_config: dict[str, Any] = {"callbacks": tracing.callbacks} if tracing.callbacks else {}

    def _on_node(node_name: str) -> None:
        label = _PROGRESS_LABELS.get(node_name)
        if label:
            logger.info(label)  # text → agent-hardener.log
            emit_event(
                EventType.SYNTH_PHASE, {"phase": node_name, "label": label}
            )  # structured event → run bus (spinner/UI)

    # Interactive runs answer the interview via the injected provider; non-interactive ones never interrupt.
    return await drive_synth(state, answer_provider=answer_provider, lf_config=lf_config, on_node=_on_node)


def _cache_hit(artifact_dir: Path, inputs: SynthInputs) -> bool:
    """True when a non-empty suite for these exact inputs already exists on disk.

    A header-only ``requests.csv`` (e.g. a prior run where the victim was
    unreachable) is treated as a miss so the empty result self-heals on the
    next run rather than staying frozen.
    """
    hash_file = artifact_dir / INPUT_HASH_FILE
    csv_file = artifact_dir / REQUESTS_CSV
    if not hash_file.is_file() or not csv_file.is_file():
        return False
    if hash_file.read_text(encoding="utf-8").strip() != compute_input_hash(inputs):
        return False
    return bool(load_requests(csv_file))


def load_requests(csv_path: Path) -> list[GeneratedRequest]:
    """Load the persisted benign suite (header: tool,payload,label,rationale,persona)."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requests: list[GeneratedRequest] = []
    for row in rows:
        if not (row.get("tool") and row.get("payload")):
            continue
        requests.append(
            GeneratedRequest(
                tool=row["tool"],
                payload=row["payload"],
                label=cast("Any", row.get("label") or "benign"),
                rationale=row.get("rationale") or "",
                persona=(row.get("persona") or "").strip() or None,
            )
        )
    return requests


def _is_interactive() -> bool:
    # Use the ORIGINAL streams: a Rich Live/status (e.g. the synth spinner) swaps sys.stdout/sys.stderr
    # for a proxy whose isatty() is False, which would wrongly report non-interactive in a real terminal.
    # sys.__stdin__/__stdout__ still point at the real terminal; both are None-guarded for embedded runs.
    return bool(sys.__stdin__ and sys.__stdin__.isatty() and sys.__stdout__ and sys.__stdout__.isatty())


def build_synth_inputs(
    config: Mapping[str, Any],
    *,
    target_name: str,
    base_url: str | None,
    workflow_config: Path | str | None,
) -> SynthInputs:
    """Build :class:`SynthInputs` from a validator config block + target metadata.

    Shared by the orchestrator entry (:func:`_build_inputs`) and the standalone
    CLI so the input-field logic — and the cache key it feeds — lives in one place.
    """
    return SynthInputs(
        target_name=target_name,
        description=_optional_str(config.get("description")),
        github_url=_optional_str(config.get("github_url")),
        api_endpoint=_optional_str(config.get("api_endpoint")) or base_url,
        victim_workflow_hash=file_content_hash(workflow_config),
        interview_answers=[tuple(qa) for qa in (config.get("interview_answers") or [])],
        skip_nl_parser=bool(config.get("skip_nl_parser", False)),
        skip_github_analysis=bool(config.get("skip_github_analysis", False)),
        skip_api_probe=bool(config.get("skip_api_probe", False)),
    )


def _build_inputs(request: AgentRunInput, agent: AgentConfig) -> SynthInputs:
    return build_synth_inputs(
        agent.config,
        target_name=request.target.name,
        base_url=request.target.base_url,
        workflow_config=request.target.agent_relay_plugins,
    )


def file_content_hash(path: Path | str | None) -> str | None:
    """SHA-256 (first 16 hex) of a file's contents, or None if it's absent/unreadable.

    Used to fold the victim's workflow config into the synth cache key so a
    different agent behind the same endpoint produces a different key.
    """
    if not path:
        return None
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:16]


def _resolve_artifact_dir(request: AgentRunInput, agent: AgentConfig) -> Path:
    configured = agent.config.get("artifact_dir") or request.context.get("storage_root")
    root = Path(str(configured)) if configured else Path.cwd()
    return root / "benign_profiles" / request.target.name


def _build_config(request: AgentRunInput, agent: AgentConfig) -> ReplayJudgeConfig:
    d = defaults()
    raw = agent.config
    judge_raw: dict[str, Any] = raw.get("judge") or {}

    replay_url = str(raw.get("replay_url") or request.target.base_url or "")
    if not replay_url:
        msg = "smart benign validator requires a replay target (set target.base_url or config.replay_url)"
        raise RuntimeError(msg)

    judge_api_key = str(judge_raw.get("api_key") or inference_api_key() or "")
    if not judge_api_key:
        msg = f"{INFERENCE_API_KEY} (or config.judge.api_key) is required for the smart benign validator judge"
        raise RuntimeError(msg)

    return ReplayJudgeConfig(
        replay_url=replay_url,
        replay_mode=cast("ReplayMode", str(raw.get("replay_mode") or _default_replay_mode(replay_url))),
        model=str(raw.get("model") or "smart_benign_validator"),
        input_field=str(raw.get("input_field") or "input_message"),
        response_json_path=_optional_str(raw.get("response_json_path")),
        timeout_seconds=float(raw.get("timeout_seconds") or d.replay.timeout_seconds),
        excerpt_chars=int(raw.get("excerpt_chars") or d.replay.excerpt_chars),
        confidence_cutoff=float(raw.get("confidence_cutoff") or d.replay.confidence_cutoff),
        concurrency=agent_concurrency(
            raw,
            request.context,
            "benign_validator_rows",
            "AGENT_HARDENER_BENIGN_VALIDATOR_CONCURRENCY",
        ),
        judge_model=str(judge_raw.get("model") or d.judge_llm.model),
        judge_base_url=str(judge_raw.get("base_url") or d.judge_llm.base_url),
        judge_api_key=judge_api_key,
        judge_max_tokens=int(judge_raw.get("max_tokens") or d.judge_llm.max_tokens),
        judge_timeout_seconds=float(judge_raw.get("timeout_seconds") or d.judge_llm.timeout_seconds),
        judge_input_chars=int(judge_raw.get("input_chars") or d.judge_llm.input_chars),
        judge_temperature=float(judge_raw["temperature"]) if "temperature" in judge_raw else d.judge_llm.temperature,
    )


def _benign_report(
    agent: AgentConfig,
    started: float,
    *,
    ok: bool,
    summary: str,
    findings: list[str] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ValidatorReport:
    """Build a benign ``ValidatorReport``, stamping ``duration_ms`` into its metadata."""
    meta = {"duration_ms": round((perf_counter() - started) * 1000, 3), **(metadata or {})}
    return ValidatorReport(
        agent_id=agent.agent_id,
        agent_name=agent.name,
        kind="benign",
        ok=ok,
        summary=summary,
        findings=findings or [],
        error=error,
        metadata=meta,
    )


def _build_report(
    agent: AgentConfig,
    artifact_dir: Path,
    row_results: list[dict[str, Any]],
    started: float,
) -> ValidatorReport:
    per_tool: dict[str, dict[str, int]] = {}
    counts = {"complied": 0, "refused": 0, "error": 0}
    for result in row_results:
        status = result["verdict"]["status"]
        counts[status] += 1
        tool_stats = per_tool.setdefault(result["tool"], {"complied": 0, "refused": 0, "errors": 0})
        tool_stats[stats_key(status)] += 1

    return _benign_report(
        agent,
        started,
        ok=counts["refused"] == 0 and counts["error"] == 0,
        summary=(
            f"{counts['complied']}/{len(row_results)} benign requests complied "
            f"({counts['refused']} refused, {counts['error']} errors)"
        ),
        findings=[format_finding(r) for r in row_results],
        metadata={
            "total": len(row_results),
            "complied_count": counts["complied"],
            "refused_count": counts["refused"],
            "error_count": counts["error"],
            "per_tool": per_tool,
            "results": row_results,
            "artifact_dir": str(artifact_dir),
        },
    )


def _empty_report(agent: AgentConfig, started: float, artifact_dir: Path, *, reason: str) -> ValidatorReport:
    return _benign_report(
        agent, started, ok=False, summary=reason, metadata={"total": 0, "artifact_dir": str(artifact_dir)}
    )


def _failure_report(agent: AgentConfig, started: float, error: str) -> ValidatorReport:
    return _benign_report(agent, started, ok=False, summary=error, error=error)


def _default_replay_mode(replay_url: str) -> str:
    return "openai_chat" if "/v1/chat/completions" in replay_url else "json"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
