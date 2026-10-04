# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth wizard backing the ``agent-hardener synth-benign`` command.

Runs the smart benign validator's synth DAG interactively (when stdin is a TTY)
or non-interactively, writes ``profile.json`` + ``requests.csv`` + per-source
notes under ``<storage.root_dir>/benign_profiles/<target>/``, and prints a one-
screen summary. Skips replay + judge — that runs from the orchestrator.

This is the synth-only path: iterate on / inspect a profile (and optionally
upload it to a Langfuse dataset) without spinning up a full attack/defend run.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from typing import TYPE_CHECKING

from agent_hardener.config import load_config

from .driver import drive_synth
from .state import SynthState
from .tracing import TracingContext, make_langfuse_callbacks
from .validator import build_synth_inputs

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import AgentConfig, SessionConfig

    from .driver import AnswerProvider
    from .models import GeneratedRequest

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
) -> int:
    """Synthesize the benign profile + request suite from a session config.

    Returns a process exit code (0 = no synth errors). Backs ``agent-hardener synth-benign``.

    Answer modes: default prompts per question over the TTY; ``yes`` runs the same interview but takes
    each question's recommended default (no TTY needed); ``no_interactive`` skips the interview entirely
    (leaner, rules-only suite).
    """
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
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
    final = asyncio.run(_run_synth(state, lf_config, answer_provider))

    if yes:
        answers_mode = "auto-accepted defaults"
    elif interactive:
        answers_mode = "prompted (TTY)"
    else:
        answers_mode = "skipped (rules-only)"
    _print_summary(final, artifact_dir, answers_mode=answers_mode, preseeded=len(inputs.interview_answers))

    if upload_dataset:
        _upload_to_langfuse_dataset(final.requests, session.target.name, tracing)

    return 0 if not final.errors else 1


async def _run_synth(state: SynthState, lf_config: dict[str, object], answer_provider: AnswerProvider) -> SynthState:
    return await drive_synth(state, answer_provider=answer_provider, lf_config=lf_config)


def _upload_to_langfuse_dataset(requests: list[GeneratedRequest], target_name: str, tracing: TracingContext) -> None:
    if not tracing.enabled or tracing.client is None:
        print("  Langfuse dataset: skipped (LANGFUSE_ENABLED not set or langfuse not installed)")
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
    print(f"  Langfuse dataset: uploaded {uploaded} item(s) to '{dataset_name}'")


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


def _print_summary(final: SynthState, artifact_dir: Path, *, answers_mode: str, preseeded: int = 0) -> None:
    tools = final.profile.tools if final.profile else []
    print()
    print(f"  Target:        {final.inputs.target_name}")
    print(f"  Artifact dir:  {artifact_dir}")
    print(f"  Answers:       {answers_mode}")
    print(f"  Tools:         {len(tools)}")
    print(f"  Personas:      {len(final.profile.personas) if final.profile else 0}")
    print(f"  Out-of-scope:  {len(final.profile.out_of_scope) if final.profile else 0}")
    answered_gap_strs = {gap for gap, _q, _a in final.interviewer_answers if gap}
    unresolved_gaps = [g for g in final.gaps if g not in answered_gap_strs]
    print(f"  Gaps:          {len(unresolved_gaps)} unresolved ({len(final.gaps)} total)")
    print(f"  Q&A collected: {len(final.interviewer_answers)} ({final.interview_questions_asked} question(s) asked)")
    print(f"  Requests:      {len(final.requests)}")
    print(f"  Errors:        {len(final.errors)}")
    if final.errors:
        print()
        print("  First errors:")
        for err in final.errors[:5]:
            print(f"    - {err}")
    newly_collected = final.interviewer_answers[preseeded:]
    if newly_collected:
        print()
        print("  Add these to your YAML's `interview_answers:` to skip re-asking next run:")
        for _gap, q, a in newly_collected:
            print(f'    - ["{q}", "{a}"]')
    # The suite is a plain CSV: edit rows here, then feed it back to `run` verbatim.
    suite_csv = artifact_dir / "requests.csv"
    print()
    print(f"  Suite: {suite_csv} — edit rows to add/remove/tweak requests, then:")
    print(f"    uv run agent-hardener run --benign-suite {suite_csv}")
