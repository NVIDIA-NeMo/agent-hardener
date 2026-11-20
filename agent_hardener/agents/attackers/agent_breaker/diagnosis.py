# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Say why a garak AgentBreaker scan produced nothing, from the files garak left behind.

garak exits 0 when the probe gives up before its first attack: it logs a warning and records zero
attempts. Read only the exit code and the hitlog, and that run looks like a clean scan of a victim that
resisted. Its absence of tool traffic is then blamed on the victim's Relay wiring. The evidence for
the real cause is in two files: the report (``*.report.jsonl``), which says how many attacks were sent
and answered, and garak's own log, which says why the probe stopped.

The log is matched on garak's message text, which is not an API. :data:`ZERO_ATTEMPT_RULES` pins the
markers for garak 0.17, and a message that matches none of them is reported as ``zero_attempts`` with the
log tail, so a reworded garak gives a vaguer error rather than a wrong one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_hardener.errors import ATTACKER_LLM_UNAVAILABLE

#: garak prints where it writes its detailed log; stdout carries none of the probe's warnings.
_LOG_PATH_LINE = re.compile(r"logging to (\S+\.log)\s*$")
#: How the OpenAI client garak's models use logs each response.
_HTTP_RESPONSE = re.compile(r'HTTP Response: \w+ (\S+) "(\d{3})')
_AUTH_STATUSES = frozenset({"401", "403"})
#: Lines kept from the end of a log when no rule explains it.
_TAIL_LINES = 20
#: The first line every garak process writes; a log appended to by several runs is cut at the last one.
_RUN_START = re.compile(r"\bINFO\s+invoked\s*$")


@dataclass(frozen=True, slots=True)
class ScanFailure:
    """Why a scan yielded no usable attacks: a stable ``code``, an operator-facing message and fix."""

    code: str
    message: str
    remediation: str
    #: Whether the cause is random, so running the same scan again is likely to succeed.
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class ScanEvidence:
    """What the report says the probe did."""

    #: Distinct attacks sent. garak writes each attempt twice (sent, then scored), so this is per uuid.
    attempts: int = 0
    #: Attacks the victim answered with any text at all.
    answered: int = 0
    #: The tools garak aimed its attacks at, as it named them.
    targets: tuple[str, ...] = ()


def read_scan_evidence(report: Path | None) -> ScanEvidence:
    """Count the attempts in garak's report. A missing report means nothing was sent."""
    if report is None or not report.exists():
        return ScanEvidence()
    sent: set[str] = set()
    answered: set[str] = set()
    targets: set[str] = set()
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("entry_type") != "attempt":
            continue
        uuid = str(entry.get("uuid"))
        sent.add(uuid)
        if any(_output_text(output) for output in entry.get("outputs") or []):
            answered.add(uuid)
        notes = entry.get("notes")
        if isinstance(notes, dict) and isinstance(notes.get("current_target"), str):
            targets.add(notes["current_target"])
    return ScanEvidence(attempts=len(sent), answered=len(answered), targets=tuple(sorted(targets)))


def _output_text(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("text") or "").strip()
    return str(output or "").strip()


def detail_log_path(stdout_log: Path) -> Path | None:
    """The detailed log garak announced on stdout, when it exists."""
    if not stdout_log.exists():
        return None
    for line in stdout_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := _LOG_PATH_LINE.search(line):
            path = Path(match.group(1))
            return path if path.exists() else None
    return None


@dataclass(frozen=True, slots=True)
class _Rule:
    """``terminal`` is where the probe stopped; ``preceding``, when set, must appear before it."""

    terminal: str
    preceding: str | None
    failure: ScanFailure


_RERUN = "Re-run the scan; this is intermittent."

#: Most specific first; the first rule whose markers are present explains the run.
ZERO_ATTEMPT_RULES: tuple[_Rule, ...] = (
    # Discovery is garak's first request to the victim, so a dead victim stops the probe here, before
    # any attempt exists to show it unanswered.
    _Rule(
        "No tools found",
        "Discovery call failed",
        ScanFailure(
            "victim_unreachable",
            "garak's first request to the victim (asking which tools it has) failed",
            "Inspect the victim agent log; check the endpoint garak attacks (target_uri / target_port).",
        ),
    ),
    _Rule(
        "No tools found",
        "Agent returned empty response during discovery",
        ScanFailure(
            "victim_unreachable",
            "garak asked the victim which tools it has and got no response",
            "Inspect the victim agent log; check the endpoint garak attacks (target_uri / target_port) and "
            "its request timeout (garak.request_timeout).",
        ),
    ),
    _Rule(
        "No tools found",
        "Parse model failed to parse discovery response",
        ScanFailure(
            "discovery_unparsed",
            "garak asked the victim which tools it has, but its parse model returned nothing usable for the answer",
            _RERUN,
            retryable=True,
        ),
    ),
    _Rule(
        "No tools found",
        "Failed to parse discovery JSON",
        ScanFailure(
            "discovery_unparsed",
            "garak asked the victim which tools it has, but its parse model's answer was not valid JSON",
            _RERUN,
            retryable=True,
        ),
    ),
    _Rule(
        "No tools found",
        None,
        ScanFailure(
            "victim_disclosed_no_tools",
            "garak asked the victim which tools it has and the victim named none",
            "Check that the victim's tools are bound to its model and that its system prompt does not forbid "
            "describing them.",
        ),
    ),
    _Rule(
        "No tools to attack",
        "Failed to parse JSON response",
        ScanFailure(
            "tool_analysis_unparsed",
            "garak's attacker model returned malformed JSON while analysing the victim's tools",
            "Re-run the scan. If it recurs, keep JSON mode on (garak.json_mode) or switch the attacker model "
            "(garak.red_team_model_name).",
            retryable=True,
        ),
    ),
    _Rule(
        "No tools to attack",
        None,
        ScanFailure(
            "no_priority_targets",
            "garak's attacker model analysed the victim's tools and chose none to attack",
            "Check the tool descriptions the victim gives: a tool described too vaguely gives the attacker "
            "nothing to aim at.",
        ),
    ),
)

_NO_ATTACKS_GENERATED = ScanFailure(
    "no_attacks_generated",
    "garak chose {targets} to attack but produced no attack for any of them",
    "Check garak.max_attempts_per_tool (it must be at least 1) and garak's log for why the attacker model's "
    "exploit prompts were empty.",
)
#: Logged once the analysis has picked its targets, just before the first attack is built.
_TARGETS_CHOSEN = re.compile(r"AgentBreaker # Attacking (\d+) tools")
#: Lines that say what garak did; DEBUG transport noise is left out of an unexplained failure's excerpt.
_NOTEWORTHY = re.compile(r"\b(WARNING|ERROR)\b|AgentBreaker #")

_ATTEMPTS_NOT_RECORDED = ScanFailure(
    "attempts_not_recorded",
    "garak started attacking but recorded no attempt in its report",
    _RERUN,
    retryable=True,
)
#: Logged once per attack conversation, so its presence means attacks were under way.
_ATTACK_STARTED = "Starting NEW conversation"


def diagnose_scan(evidence: ScanEvidence, log_text: str, *, target_uri: str) -> ScanFailure | None:
    """Explain a scan that produced no usable attacks, or ``None`` when it did.

    Checked in causal order, so the most upstream cause wins: rejected model credentials explain an
    empty scan better than the "no tools" warning they lead to.

    Args:
        evidence: What the report says was sent and answered.
        log_text: garak's detailed log (stdout's when the detailed one is missing).
        target_uri: The victim's endpoint, so its own responses are not read as the attacker's.
    """
    lines = _current_run(log_text.splitlines())
    if (auth := _model_auth_failure(lines, target_uri)) is not None:
        return auth
    if evidence.attempts == 0:
        return _zero_attempts(lines)
    if evidence.answered == 0:
        return ScanFailure(
            "victim_unreachable",
            f"garak sent {evidence.attempts} attack(s) and the victim answered none of them",
            "Inspect the victim agent log; check the endpoint garak attacks (target_uri) and the victim's "
            "request timeout (garak.request_timeout).",
        )
    return None


def _model_auth_failure(lines: list[str], target_uri: str) -> ScanFailure | None:
    """Every call to some model host was refused for credentials.

    The backstop to the pre-scan model check (:mod:`agent_hardener.agents.attackers.agent_breaker.preflight`),
    which only covers NIM models and only before the scan: a key revoked or out of quota mid-scan, or a
    non-NIM model, still ends here, under the same failure code.

    Per host, because garak's attacker and judge can sit on different endpoints: a judge whose key is
    rejected leaves every attack unscored even while the attacker works. A host that answered even once
    is not reported; an occasional 403 there is a quota blip, not a dead key.
    """
    victim_host = urlsplit(target_uri).netloc
    refused: dict[str, str] = {}
    served: set[str] = set()
    for line in lines:
        if not (match := _HTTP_RESPONSE.search(line)):
            continue
        host, status = urlsplit(match.group(1)).netloc, match.group(2)
        if host == victim_host:
            continue
        if status in _AUTH_STATUSES:
            refused.setdefault(host, status)
        elif status.startswith("2"):
            served.add(host)
    dead = {host: status for host, status in refused.items() if host not in served}
    if not dead:
        return None
    where = ", ".join(f"{host} (HTTP {status})" for host, status in sorted(dead.items()))
    return ScanFailure(
        ATTACKER_LLM_UNAVAILABLE,
        f"garak's attacker/judge model credentials were rejected by {where}",
        "Check the key garak receives (NIM_API_KEY, aliased from the run's inference key) and that it is "
        "entitled to the attacker and judge models.",
    )


def _zero_attempts(lines: list[str]) -> ScanFailure:
    if any(_ATTACK_STARTED in line for line in lines):
        return _ATTEMPTS_NOT_RECORDED
    chosen = [match for line in lines if (match := _TARGETS_CHOSEN.search(line))]
    if chosen:
        targets = f"{chosen[-1].group(1)} tool(s)"
        return replace(_NO_ATTACKS_GENERATED, message=_NO_ATTACKS_GENERATED.message.format(targets=targets))
    for rule in ZERO_ATTEMPT_RULES:
        stop = _last_index(lines, rule.terminal)
        if stop is None:
            continue
        if rule.preceding is None:
            return rule.failure
        cause = _last_index(lines[:stop], rule.preceding)
        if cause is not None:
            detail = lines[cause].split(rule.preceding, 1)[-1].strip(" :")
            return replace(rule.failure, message=f"{rule.failure.message} ({detail})") if detail else rule.failure
    noteworthy = [line for line in lines if _NOTEWORTHY.search(line)] or lines
    tail = "\n".join(noteworthy[-_TAIL_LINES:])
    return ScanFailure(
        "zero_attempts",
        f"garak finished without sending a single attack; its last warnings and steps:\n{tail}",
        "Read garak's log for the reason the probe stopped.",
    )


def _current_run(lines: list[str]) -> list[str]:
    """The lines the last garak process wrote, when several appended to one log."""
    starts = [index for index, line in enumerate(lines) if _RUN_START.search(line)]
    return lines[starts[-1] :] if starts else lines


def _last_index(lines: list[str], marker: str) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if marker in lines[index]:
            return index
    return None
