# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth wizard backing the ``agent-hardener synth-benign`` command.

Runs the smart benign validator's synth DAG interactively (when stdin is a TTY)
or non-interactively, writes ``profile.json`` + ``requests.csv`` + per-source
notes under ``<storage.root_dir>/benign_profiles/<target>/``, and prints a one-
screen summary. With the victim's traces it then verifies the suite's tools against
the live victim (:mod:`.tool_verification`); the in-loop replay + judge runs from
the orchestrator.

This is the synth-only path: iterate on / inspect a profile (and optionally
upload it to a Langfuse dataset) without spinning up a full attack/defend run.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings
from typing import TYPE_CHECKING

from agent_hardener.agents.validators.args_judge import build_args_judge
from agent_hardener.config import load_config
from agent_hardener.display.console import fail, hint, info, ok, section, warn
from agent_hardener.display.progress import build_console_renderer
from agent_hardener.events import EventType
from agent_hardener.loggers import EventBus, emit_event, using_event_sink
from agent_hardener.models import AgentRunInput

from .config import defaults
from .driver import drive_synth
from .progress import PhaseTracker
from .state import SynthState
from .tool_verification import (
    ToolVerificationOutcome,
    default_drop_decider,
    run_tool_verification,
    tty_drop_decider,
    write_verification,
)
from .tracing import TracingContext, make_langfuse_callbacks
from .validator import build_replay_config, build_synth_inputs

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import AgentConfig, SessionConfig
    from agent_hardener.traces import TraceProvider

    from .driver import AnswerProvider
    from .models import GeneratedRequest
    from .tool_verification import DropDecider

logger = logging.getLogger(__name__)

SMART_BENIGN_IMPL = "agent_hardener.agents.validators.smart_benign:run"


def run_synth_wizard(
    config: Path,
    *,
    answer_provider: AnswerProvider,
    no_interactive: bool = False,
    yes: bool = False,
    validator: str | None = None,
    upload_dataset: bool = False,
    verify_tools: bool = True,
    trace_provider: TraceProvider | None = None,
) -> int:
    """Synthesize the benign profile + request suite from a session config.

    Returns a process exit code (0 = no synth errors). Backs ``agent-hardener synth-benign``.

    Answer modes: default prompts per question over the TTY; ``yes`` runs the same interview but takes
    each question's recommended default (no TTY needed); ``no_interactive`` skips the interview entirely
    (leaner, rules-only suite). The same modes govern tool verification's drop prompt: only a real TTY
    without ``yes`` / ``no_interactive`` is asked; otherwise missing tools are dropped and refused rows kept.

    ``trace_provider`` reads the live victim's traces; with it (and ``verify_tools``), the synthesized suite
    is replayed and its tools verified before the summary.
    """
    # Logging is wired by the CLI (``configure_cli_logging``): console warnings + the command's log file.
    warnings.filterwarnings("ignore")

    session = load_config(config)
    validator_agent = _resolve_validator(session, validator)
    inputs = build_synth_inputs(
        validator_agent.config,
        target_name=session.target.name,
        base_url=session.target.base_url,
        workflow_config=session.target.agent_relay_plugins,
    )
    artifact_dir = session.storage.root_dir / "benign_profiles" / session.target.name
    interactive = yes or ((not no_interactive) and sys.stdin.isatty())

    state = SynthState.from_inputs(inputs, interactive=interactive, artifact_dir=artifact_dir)
    tracing = make_langfuse_callbacks(target_name=session.target.name)
    lf_config: dict[str, object] = {"callbacks": tracing.callbacks} if tracing.callbacks else {}
    final = _synthesize_with_progress(state, lf_config, answer_provider)
    verification = None
    if verify_tools and defaults().verification.enabled and final.requests:
        if trace_provider is None:
            warn("tool verification skipped: the victim has no relay-victim block, so there are no traces to read")
        else:
            prompt = (not yes) and (not no_interactive) and sys.stdin.isatty()
            verification = _verify_with_progress(
                final,
                session,
                validator_agent,
                trace_provider,
                decide=tty_drop_decider if prompt else default_drop_decider,
            )
            write_verification(artifact_dir, verification)
            if verification.passes:
                final = final.model_copy(update={"requests": verification.requests, "profile": verification.profile})

    if yes:
        answers_mode = "auto-accepted defaults"
    elif interactive:
        answers_mode = "prompted (TTY)"
    else:
        answers_mode = "skipped (rules-only)"
    _print_summary(
        final,
        artifact_dir,
        answers_mode=answers_mode,
        preseeded=len(inputs.interview_answers),
        verification=verification,
    )

    if upload_dataset:
        _upload_to_langfuse_dataset(final.requests, session.target.name, tracing)

    return 0 if not final.errors else 1


def _synthesize_with_progress(
    state: SynthState, lf_config: dict[str, object], answer_provider: AnswerProvider
) -> SynthState:
    """Run synth with the console renderer showing each finished step, so a long synthesis never looks stuck.

    The steps reach the console as ``synth_phase`` events, the same ones Studio lists, through an event bus so
    a rendering failure can never abort synthesis.
    """
    tracker = PhaseTracker(state)
    bus = EventBus()
    bus.subscribe(build_console_renderer(verbose=False))
    started = time.monotonic()
    with using_event_sink(bus.emit):
        emit_event(EventType.STATUS_STARTED, {"label": "Synthesizing benign suite", "detail": tracker.first_step()})
        try:
            final = asyncio.run(
                drive_synth(state, answer_provider=answer_provider, lf_config=lf_config, on_node=tracker.on_node)
            )
        finally:
            emit_event(EventType.STATUS_COMPLETED, {})
    ok("Benign suite synthesized", elapsed=time.monotonic() - started)
    return final


def _verify_with_progress(
    final: SynthState,
    session: SessionConfig,
    validator_agent: AgentConfig,
    provider: TraceProvider,
    *,
    decide: DropDecider,
) -> ToolVerificationOutcome:
    """Replay the suite and verify its tools, with each pass printed as it finishes (see ``tool_verification``)."""
    request = AgentRunInput(
        round_id="synth-verify",
        target=session.target,
        context={"storage_root": str(session.storage.root_dir)},
        validator_kind="benign",
    )
    cfg = build_replay_config(request, validator_agent)
    bus = EventBus()
    bus.subscribe(build_console_renderer(verbose=False))
    started = time.monotonic()
    with using_event_sink(bus.emit):
        emit_event(
            EventType.STATUS_STARTED,
            {"label": "Verifying the suite's tools", "detail": f"replaying {len(final.requests)} request(s)"},
        )
        try:
            outcome = asyncio.run(
                run_tool_verification(
                    final,
                    cfg=cfg,
                    provider=provider,
                    decide=decide,
                    args_judge=build_args_judge(validator_agent, "benign"),
                )
            )
        finally:
            emit_event(EventType.STATUS_COMPLETED, {})
    ok("Tools verified against the victim", elapsed=time.monotonic() - started)
    return outcome


def _upload_to_langfuse_dataset(requests: list[GeneratedRequest], target_name: str, tracing: TracingContext) -> None:
    if not tracing.enabled or tracing.client is None:
        warn("Langfuse dataset upload skipped: LANGFUSE_ENABLED is not set or langfuse is not installed")
        return
    lf = tracing.client
    dataset_name = f"{target_name}-benign"
    lf.create_dataset(name=dataset_name, description=f"Benign requests for {target_name}")
    uploaded = 0
    for req in requests:
        try:
            lf.create_dataset_item(
                dataset_name=dataset_name,
                input={"payload": req.payload, "tool": req.tool, "persona": req.persona},
                expected_output={"label": req.label},
                metadata={"rationale": req.rationale},
            )
            uploaded += 1
        except Exception:
            logger.warning("failed to upload dataset item to Langfuse", exc_info=True)
    lf.flush()
    (ok if uploaded == len(requests) else warn)(
        f"Langfuse dataset: uploaded {uploaded} / {len(requests)} item(s) to '{dataset_name}'", markup=False
    )


def _resolve_validator(session: SessionConfig, name: str | None) -> AgentConfig:
    candidates = [v for v in session.benign_validators if v.implementation == SMART_BENIGN_IMPL]
    if not candidates:
        msg = f"no benign_validator entry with implementation {SMART_BENIGN_IMPL!r} found in {session.target.name}"
        raise SystemExit(msg)
    if name is not None:
        for candidate in candidates:
            if candidate.name == name:
                return candidate
        msg = f"no benign_validator named {name!r}; available: {[c.name for c in candidates]}"
        raise SystemExit(msg)
    if len(candidates) > 1:
        msg = (
            f"multiple smart benign validators found ({[c.name for c in candidates]}); "
            "pass --validator <name> to choose"
        )
        raise SystemExit(msg)
    return candidates[0]


def _print_summary(
    final: SynthState,
    artifact_dir: Path,
    *,
    answers_mode: str,
    preseeded: int = 0,
    verification: ToolVerificationOutcome | None = None,
) -> None:
    """The synth outcome: one ``✓``/``✗`` headline, an aligned field table, then what to do next."""
    tools = final.profile.tools if final.profile else []
    answered_gap_strs = {gap for gap, _q, _a in final.interviewer_answers if gap}
    unresolved_gaps = [g for g in final.gaps if g not in answered_gap_strs]
    section("Result")
    headline = f"Benign suite: {len(final.requests)} request(s) for {final.inputs.target_name}"
    if final.errors:
        fail(f"{headline}, {len(final.errors)} error(s)", markup=False)
    else:
        ok(headline, markup=False)
    rows = [
        ("artifact dir", str(artifact_dir)),
        ("answers", answers_mode),
        ("tools", str(len(tools))),
        ("personas", str(len(final.profile.personas) if final.profile else 0)),
        ("out-of-scope", str(len(final.profile.out_of_scope) if final.profile else 0)),
        ("gaps", f"{len(unresolved_gaps)} / {len(final.gaps)} unresolved"),
        ("Q&A", f"{len(final.interviewer_answers)} collected, {final.interview_questions_asked} question(s) asked"),
    ]
    rows.extend(_verification_rows(verification))
    for label, value in rows:
        info(f"  {label:<14}{value}", markup=False)
    if verification is not None and verification.skipped:
        warn(f"tool verification: {verification.skipped}", markup=False)
    for err in final.errors[:5]:
        warn(err, markup=False)
    if len(final.errors) > 5:
        hint(f"{len(final.errors) - 5} more error(s) in the log", markup=False)
    newly_collected = final.interviewer_answers[preseeded:]
    if newly_collected:
        hint("add these to your YAML's `interview_answers:` to skip re-asking next run:", markup=False)
        for _gap, q, a in newly_collected:
            info(f'      - ["{q}", "{a}"]', markup=False)
    if verification is not None:
        hint(f"tool verification: {artifact_dir / 'tool_verification.json'}", markup=False)
    # The suite is a plain CSV: edit rows here, then feed it back to `run` verbatim.
    suite_csv = artifact_dir / "requests.csv"
    hint(f"suite: {suite_csv} — edit rows to add/remove/tweak requests", markup=False)
    hint(f"next: uv run agent-hardener run --benign-suite {suite_csv}", markup=False)


def _verification_rows(verification: ToolVerificationOutcome | None) -> list[tuple[str, str]]:
    """The summary's tool-verification lines; every count carries what it was counted over."""
    if verification is None or not verification.passes:
        return []
    first = verification.passes[0].result
    tools = verification.profile.tools if verification.profile else []
    statuses = [str(t.verification) for t in tools]
    verdicts = ", ".join(f"{statuses.count(s)} {s}" for s in ("verified", "missing", "unknown") if s in statuses)
    dropped = sum(len(p.dropped_rows) for p in verification.passes)
    added = [name for p in verification.passes for name in p.added_tools]
    refused = sum(len(p.result.baseline_refused) for p in verification.passes)
    rows = [
        ("tool check", f"{verdicts} of {len(tools)} tool(s); {first.traced_rows}/{first.rows} row(s) traced"),
        ("passes", str(len(verification.passes))),
        ("dropped", f"{dropped} row(s)"),
        ("refused", f"{refused} row(s) refused by the unhardened victim"),
    ]
    if added:
        added_rows = sum(p.added_rows for p in verification.passes)
        rows.append(("added", f"{', '.join(added)} ({added_rows} row(s))"))
    if verification.unhandled_tools:
        rows.append(("not added", f"{', '.join(verification.unhandled_tools)} (pass cap reached)"))
    return rows
