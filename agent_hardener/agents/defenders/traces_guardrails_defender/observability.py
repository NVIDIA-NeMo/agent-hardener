# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every phase produced, readable after the run: JSON artifacts and a summary.

Layout::

    <run_dir>/traces_guardrails_defender/<round_id>/<attack>/
        00_input.json  01_prepare.json  02_vulnerability_analysis.attempt1.json
        03_benign_args.json  04_rail.json  outcome.json  summary.md

Each node's state update is written as it happens, so a run that dies midway still shows how far it
got. Recording never fails the defender: an unwritable directory is logged and ignored.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_hardener.storage import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_hardener.models import DefenderInput

    from .schemas import Outcome, RailDraft, RailGrade, VulnerabilityAnalysis

logger = logging.getLogger("agent_hardener.session")

ARTIFACTS_DIRNAME = "traces_guardrails_defender"
OUTCOME_FILENAME = "outcome.json"
SUMMARY_FILENAME = "summary.md"
#: ``DefenderInput.context`` key that overrides where artifacts go (tests, local smoke runs).
ARTIFACTS_DIR_KEY = "traces_guardrails_defender_artifacts_dir"

#: The file each phase writes. ``analyze`` can run twice (a retry when the answer did not match the
#: trace), so its files always carry an attempt number.
PHASE_FILES = {
    "input": "00_input",
    "prepare": "01_prepare",
    "analyze": "02_vulnerability_analysis",
    "benign_args": "03_benign_args",
    "generate": "04_rail",
}
_REPEATABLE = frozenset({"analyze"})


class Recorder:
    """Writes one attack's phase artifacts. ``directory=None`` records nothing."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = {}

    def write(self, phase: str, payload: Any) -> None:
        """Write ``payload`` as the phase's file (``NN_<phase>[.attemptK].json``)."""
        if phase not in PHASE_FILES:
            msg = f"unknown traces guardrails defender phase {phase!r}"
            raise ValueError(msg)
        with self._lock:
            attempt = self._attempts.get(phase, 0) + 1
            self._attempts[phase] = attempt
        suffix = f".attempt{attempt}" if phase in _REPEATABLE else ""
        self.write_file(f"{PHASE_FILES[phase]}{suffix}.json", json.dumps(to_jsonable(payload), indent=2, default=str))

    def write_file(self, filename: str, text: str) -> None:
        """Write ``text`` as ``filename`` in the attack's directory."""
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / filename).write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning("traces guardrails defender could not write artifact %s: %s", filename, exc)

    def node(self, phase: str, fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        """Wrap a LangGraph node so its state update is recorded as ``phase``."""

        @wraps(fn)
        def recorded(*args: Any, **kwargs: Any) -> dict[str, Any]:
            update = fn(*args, **kwargs)
            self.write(phase, update)
            return update

        return recorded


def artifacts_dir(defender_input: DefenderInput, attack: str) -> Path | None:
    """Where this attack's artifacts go, or ``None`` when the run has no directory to put them in.

    The run directory is found from the plugin config the defender extends, which lives at
    ``<run_dir>/victim-active-state/plugins.toml``.
    """
    override = defender_input.context.get(ARTIFACTS_DIR_KEY)
    if override:
        root = Path(override)
    elif defender_input.relay_plugins_path is not None:
        root = defender_input.relay_plugins_path.resolve().parent.parent / ARTIFACTS_DIRNAME
    else:
        return None
    round_id = str(defender_input.context.get("round_id") or "round")
    return root / round_id / attack


def render_summary(
    outcome: Outcome,
    analysis: VulnerabilityAnalysis | None,
    drafts: list[RailDraft],
    grades: list[RailGrade],
) -> str:
    """A human-readable walkthrough of one attack: the analysis, every draft and grade, the outcome."""
    lines = [f"# Traces guardrails defender: {outcome.attack}", "", f"**Outcome:** {outcome.status}"]
    if outcome.guardrail_name:
        lines.append(f"**Guardrail:** `{outcome.guardrail_name}`")
    if outcome.reason:
        lines.append(f"**Reason:** {outcome.reason}")
    lines += ["", "## Vulnerability analysis", ""]
    if analysis is None:
        lines.append("_No analysis._")
    else:
        lines.append(f"- Target: **{analysis.target}**")
        if analysis.target == "tool":
            lines.append(f"- Call: step {analysis.call_index}, `{analysis.tool_name}`")
            lines.append(f"- Malicious args: `{json.dumps(analysis.malicious_args, default=str)}`")
        if analysis.fallback_reason:
            lines.append(f"- Fallback reason: {analysis.fallback_reason}")
        for label, value in (
            ("Attack pattern", analysis.attack_pattern),
            ("Legitimate use", analysis.benign_contrast),
            ("Evidence", analysis.evidence),
        ):
            if value:
                lines.append(f"- {label}: {value}")
    grade_by_iteration = {grade.iteration: grade for grade in grades}
    lines += ["", "## Drafts", ""]
    for draft in drafts:
        grade = grade_by_iteration.get(draft.iteration)
        verdict = f"score {grade.score:.1f}/10" if grade else "not graded"
        lines += [f"### Iteration {draft.iteration} ({verdict})", "", "```text", draft.system_instructions, "```"]
        if grade:
            lines.append("Criteria: " + ", ".join(f"{name}={value}" for name, value in grade.criteria.items()))
            lines += [f"- Fix: {fix}" for fix in grade.required_fixes]
        if draft.lint_warnings:
            lines.append("Copied attack literals: " + ", ".join(f"`{literal}`" for literal in draft.lint_warnings))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
