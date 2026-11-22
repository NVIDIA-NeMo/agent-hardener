# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The post-synth tool verification run: verify, let the operator drop, add observed tools, persist.

Each pass replays a batch of suite rows (:func:`.verify.verify_suite`): the whole suite first, then only the
rows generated for tools the previous pass saw the victim call but the profile did not list
(:mod:`.regenerate`). ``missing`` tools and rows the unhardened victim refused are offered for dropping
through a :data:`DropDecider`: a TTY prompt, or the defaults (drop missing tools, keep refused rows). The
run stops when a pass finds no new tools, or after ``verification.max_passes``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import questionary
from pydantic import Field

from agent_hardener.agents.validators.trace_step import open_trace_window
from agent_hardener.models.base import AgentHardenerModel
from agent_hardener.runtime.run_context import RunContext
from agent_hardener.traces.inspect import same_tool

from .config import defaults
from .models import GeneratedRequest, ToolSpec, VictimCapabilityProfile
from .regenerate import describe_observed_tools, generate_for_tools
from .subgraphs.profile_writer.nodes.write import _write_requests_csv
from .verify import ToolVerdict, VerificationResult, verify_suite

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.traces import TraceProvider

    from .replay_judge import ReplayJudgeConfig
    from .state import SynthState

logger = logging.getLogger(__name__)

REQUESTS_CSV = "requests.csv"
VERIFICATION_JSON = "tool_verification.json"
PRE_VERIFICATION_CSV = "requests.pre_verification.csv"
PROFILE_JSON = "profile.json"
_TITLE_CHARS = 100


class DropDecision(AgentHardenerModel):
    """What to drop after one pass: whole ``missing`` tools, and refused rows by their index in the batch."""

    tools: list[str] = Field(default_factory=list)
    rows: list[int] = Field(default_factory=list)
    decided_by: Literal["operator", "default", "nothing_to_decide"]


DropDecider: TypeAlias = Callable[
    [list[ToolVerdict], list[tuple[int, GeneratedRequest]], list[GeneratedRequest]], Awaitable[DropDecision]
]
"""``(missing tools, refused (batch index, row) pairs, the batch) -> DropDecision``."""


class PassRecord(AgentHardenerModel):
    """One pass: what the replay said, what was dropped, and what was added for the next pass."""

    number: int
    result: VerificationResult
    decision: DropDecision
    dropped_rows: list[GeneratedRequest] = Field(default_factory=list)
    added_tools: list[str] = Field(default_factory=list)
    added_rows: int = 0
    errors: list[str] = Field(default_factory=list)


class ToolVerificationOutcome(AgentHardenerModel):
    """The verified suite and profile, with every pass that produced them."""

    requests: list[GeneratedRequest]
    profile: VictimCapabilityProfile | None
    passes: list[PassRecord] = Field(default_factory=list)
    #: Tools the last allowed pass saw but that got no rows (the pass cap was reached).
    unhandled_tools: list[str] = Field(default_factory=list)
    #: Why verification did not run, or stopped early without traces; ``None`` when it ran.
    skipped: str | None = None


async def run_tool_verification(
    state: SynthState,
    *,
    cfg: ReplayJudgeConfig,
    provider: TraceProvider,
    decide: DropDecider,
    args_judge: Callable[[str, str, Any], Awaitable[bool]] | None = None,
) -> ToolVerificationOutcome:
    """Verify the synthesized suite against the live victim; see the module docstring."""
    requests, profile = list(state.requests), state.profile
    if profile is None or not requests:
        return ToolVerificationOutcome(requests=requests, profile=profile, skipped="the suite is empty")
    max_passes = defaults().verification.max_passes
    passes: list[PassRecord] = []
    batch = requests
    for number in range(1, max_passes + 1):
        window = await open_trace_window(RunContext(trace_provider=provider))
        if window is None:
            return _outcome(requests, profile, passes, skipped="could not read the victim's traces")
        result = await verify_suite(
            batch, cfg, window, known_tools=[t.name for t in profile.tools], args_judge=args_judge
        )
        if not result.traced:
            passes.append(PassRecord(number=number, result=result, decision=_nothing()))
            return _outcome(requests, profile, passes, skipped="the trace step failed; the suite is unchanged")
        already_verified = [t.name for t in profile.tools if t.verification == "verified"]
        profile = _mark(profile, result.tools)
        missing = [v for v in result.tools if v.status == "missing" and not _named(v.tool, already_verified)]
        refused = [(i, batch[i]) for i in result.baseline_refused]
        decision = await decide(missing, refused, batch) if missing or refused else _nothing()
        dropped = [r for i, r in enumerate(batch) if r.tool in decision.tools or i in decision.rows]
        requests = [r for r in requests if not any(r is d for d in dropped)]
        record = PassRecord(number=number, result=result, decision=decision, dropped_rows=dropped)
        passes.append(record)
        new = result.observed_unlisted
        if not new:
            break
        if number == max_passes:
            return _outcome(requests, profile, passes, unhandled=list(new))
        specs, describe_errors = await describe_observed_tools(new, profile.system_role)
        rows, generate_errors = await generate_for_tools(state.model_copy(update={"profile": profile}), specs)
        profile = profile.model_copy(update={"tools": [*profile.tools, *specs]})
        record.added_tools, record.added_rows = [s.name for s in specs], len(rows)
        record.errors = [*describe_errors, *generate_errors]
        if not rows:
            break
        requests, batch = [*requests, *rows], rows
    return _outcome(requests, profile, passes)


def write_verification(artifact_dir: Path, outcome: ToolVerificationOutcome) -> Path:
    """Persist the outcome next to the synth artifacts; returns the ``tool_verification.json`` path.

    The suite and profile are rewritten only when a pass ran. The suite synthesis wrote is kept beside them
    as ``requests.pre_verification.csv``, so any drop can be undone by hand.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if outcome.passes:
        synthesized = artifact_dir / REQUESTS_CSV
        if synthesized.is_file():
            shutil.copyfile(synthesized, artifact_dir / PRE_VERIFICATION_CSV)
        _write_requests_csv(artifact_dir, outcome.requests)
        if outcome.profile is not None:
            (artifact_dir / PROFILE_JSON).write_text(outcome.profile.model_dump_json(indent=2), encoding="utf-8")
    path = artifact_dir / VERIFICATION_JSON
    path.write_text(outcome.model_dump_json(indent=2, exclude={"requests", "profile"}), encoding="utf-8")
    return path


async def default_drop_decider(
    missing: list[ToolVerdict], _refused: list[tuple[int, GeneratedRequest]], _batch: list[GeneratedRequest]
) -> DropDecision:
    """No one to ask: drop the tools the victim never called, keep refused rows (they are only reported)."""
    return DropDecision(tools=[v.tool for v in missing], decided_by="default")


async def tty_drop_decider(
    missing: list[ToolVerdict], refused: list[tuple[int, GeneratedRequest]], batch: list[GeneratedRequest]
) -> DropDecision:
    """Ask the operator over the TTY; Ctrl+C on a question takes that question's default."""
    return await asyncio.to_thread(_ask, missing, refused, batch)


def _ask(
    missing: list[ToolVerdict], refused: list[tuple[int, GeneratedRequest]], batch: list[GeneratedRequest]
) -> DropDecision:
    default = DropDecision(tools=[v.tool for v in missing], decided_by="default")
    tools = rows = None
    if missing:
        tools = questionary.checkbox(
            "The victim never called these tools during the replay. Drop their rows?",
            choices=[questionary.Choice(title=_tool_title(v, batch), value=v.tool, checked=True) for v in missing],
        ).ask()
    if refused:
        rows = questionary.checkbox(
            "The unhardened victim refused these benign rows. Drop them?",
            choices=[
                questionary.Choice(title=f"[{r.tool}] {_clip(r.payload)}", value=i, checked=False) for i, r in refused
            ],
        ).ask()
    return DropDecision(
        tools=default.tools if tools is None else tools,
        rows=[] if rows is None else rows,
        decided_by="operator" if tools is not None or rows is not None else "default",
    )


def _tool_title(verdict: ToolVerdict, batch: list[GeneratedRequest]) -> str:
    sample = next(r.payload for r in batch if r.tool == verdict.tool)
    return f"{verdict.tool} — {verdict.traced_rows}/{verdict.rows} row(s) traced, e.g. {_clip(sample)}"


def _clip(text: str) -> str:
    return text if len(text) <= _TITLE_CHARS else f"{text[: _TITLE_CHARS - 1]}…"


def _mark(profile: VictimCapabilityProfile, verdicts: list[ToolVerdict]) -> VictimCapabilityProfile:
    """Stamp each profile tool with this pass's verdict; a tool once verified stays verified."""
    tools: list[ToolSpec] = []
    for tool in profile.tools:
        verdict = next((v for v in verdicts if same_tool(v.tool, tool.name)), None)
        stamp = verdict is not None and tool.verification != "verified"
        tools.append(tool.model_copy(update={"verification": verdict.status}) if stamp and verdict else tool)
    return profile.model_copy(update={"tools": tools})


def _named(tool: str, names: list[str]) -> bool:
    return any(same_tool(tool, name) for name in names)


def _nothing() -> DropDecision:
    return DropDecision(decided_by="nothing_to_decide")


def _outcome(
    requests: list[GeneratedRequest],
    profile: VictimCapabilityProfile,
    passes: list[PassRecord],
    *,
    unhandled: list[str] | None = None,
    skipped: str | None = None,
) -> ToolVerificationOutcome:
    if skipped:
        logger.warning("tool verification: %s", skipped)
    return ToolVerificationOutcome(
        requests=requests, profile=profile, passes=passes, unhandled_tools=unhandled or [], skipped=skipped
    )
