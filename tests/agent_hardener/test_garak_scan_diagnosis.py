# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telling why a garak scan produced nothing, from the report and log it left.

The log lines are verbatim from garak 0.17.0 runs that exited 0 with no attempt.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.attackers.agent_breaker import run_agent_breaker
from agent_hardener.agents.attackers.agent_breaker.diagnosis import (
    ScanEvidence,
    detail_log_path,
    diagnose_scan,
    read_scan_evidence,
)
from agent_hardener.errors import (
    ATTACKER_LLM_UNAVAILABLE,
    AttackerError,
    ModelUnavailableError,
    VictimUnavailableError,
)
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput
from agent_hardener.models.contracts import AttackRecord
from agent_hardener.runtime.stages.attack import raise_on_attacker_failure

if TYPE_CHECKING:
    from pathlib import Path

VICTIM = "http://127.0.0.1:8000/v1/chat/completions"
NO_ATTEMPTS = ScanEvidence()
START = "2026-11-04 18:25:22,675  INFO  invoked"
UNPARSED_ANALYSIS = """\
2026-11-04 18:25:37,497  INFO  AgentBreaker # Discovered 4 tools from agent
2026-11-04 18:25:43,784  WARNING  AgentBreaker # Failed to parse JSON response: Invalid \\escape: line 16 column 234 (char 1662)
2026-11-04 18:25:43,785  DEBUG  AgentBreaker # Analyzed 0 tools, Priority targets: []
2026-11-04 18:25:43,785  WARNING  AgentBreaker # No tools to attack
2026-11-04 18:25:43,801  WARNING  zero attempt results: probe garak.probes.agent_breaker.AgentBreaker"""


def _log(*blocks: str) -> str:
    return "\n".join((START, *blocks))


@pytest.mark.parametrize(
    ("log", "code", "retryable"),
    [
        (_log(UNPARSED_ANALYSIS), "tool_analysis_unparsed", True),
        (
            _log(
                "2026-11-04 18:25:30,000  WARNING  AgentBreaker # Parse model failed to parse discovery response",
                "2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack",
            ),
            "discovery_unparsed",
            True,
        ),
        (
            _log(
                "2026-11-04 18:25:30,000  WARNING  AgentBreaker # Failed to parse discovery JSON: Expecting value",
                "2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack",
            ),
            "discovery_unparsed",
            True,
        ),
        (
            _log(
                "2026-11-04 18:25:30,000  WARNING  AgentBreaker # Discovery call failed: Connection refused",
                "2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack",
            ),
            "victim_unreachable",
            False,
        ),
        (
            _log(
                "2026-11-04 18:25:30,000  WARNING  AgentBreaker # Agent returned empty response during discovery",
                "2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack",
            ),
            "victim_unreachable",
            False,
        ),
        (
            _log(
                "2026-11-04 18:25:30,000  INFO  AgentBreaker # Discovered agent purpose from agent",
                "2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack",
            ),
            "victim_disclosed_no_tools",
            False,
        ),
        (
            _log(
                "2026-11-04 18:25:43,785  DEBUG  AgentBreaker # Analyzed 3 tools, Priority targets: []",
                "2026-11-04 18:25:43,785  WARNING  AgentBreaker # No tools to attack",
            ),
            "no_priority_targets",
            False,
        ),
        (
            _log("2026-11-04 18:26:00,000  INFO  AgentBreaker # Starting NEW conversation with improved payload"),
            "attempts_not_recorded",
            True,
        ),
        (
            _log(
                "2026-11-18 22:25:49,573  INFO  AgentBreaker # Attacking 5 tools",
                "2026-11-18 22:25:49,588  WARNING  zero attempt results: probe garak.probes.agent_breaker.AgentBreaker",
            ),
            "no_attacks_generated",
            False,
        ),
        (_log("2026-11-04 18:26:00,000  ERROR  something garak 0.18 says instead"), "zero_attempts", False),
    ],
)
def test_a_scan_with_no_attempt_is_explained_by_where_the_probe_stopped(log: str, code: str, retryable: bool) -> None:
    failure = diagnose_scan(NO_ATTEMPTS, log, target_uri=VICTIM)

    assert failure is not None
    assert (failure.code, failure.retryable) == (code, retryable)


def test_the_parse_error_is_quoted_and_an_unknown_stop_keeps_the_log_tail() -> None:
    explained = diagnose_scan(NO_ATTEMPTS, _log(UNPARSED_ANALYSIS), target_uri=VICTIM)
    unexplained = diagnose_scan(
        NO_ATTEMPTS,
        _log("2026-11-04 18:26:00,000  WARNING  the last thing garak said", "2026-11-04  DEBUG  transport noise"),
        target_uri=VICTIM,
    )

    assert explained is not None
    assert "(Invalid \\escape: line 16 column 234 (char 1662))" in explained.message
    assert unexplained is not None
    assert "the last thing garak said" in unexplained.message
    assert "transport noise" not in unexplained.message


def test_only_the_last_run_in_a_shared_log_is_read() -> None:
    """A retried scan appends to the same log; the first scan's warnings must not explain the second's."""
    log = "\n".join((START, UNPARSED_ANALYSIS, START, "2026-11-04 18:30:00,000  INFO  done"))

    failure = diagnose_scan(NO_ATTEMPTS, log, target_uri=VICTIM)

    assert failure is not None
    assert failure.code == "zero_attempts"


def _response(url: str, status: str) -> str:
    return f'2026-11-04 18:25:37,484  DEBUG  HTTP Response: POST {url} "{status}" Headers({{}})'


def test_rejected_model_credentials_outrank_the_empty_scan_they_cause() -> None:
    log = _log(_response("https://integrate.api.nvidia.com/v1/chat/completions", "401 Unauthorized"), UNPARSED_ANALYSIS)

    failure = diagnose_scan(NO_ATTEMPTS, log, target_uri=VICTIM)

    assert failure is not None
    assert failure.code == ATTACKER_LLM_UNAVAILABLE
    assert "integrate.api.nvidia.com (HTTP 401)" in failure.message


def test_an_occasional_refusal_from_a_working_host_or_the_victim_is_not_an_auth_failure() -> None:
    log = _log(
        _response("https://inference-api.nvidia.com/v1/chat/completions", "403 Forbidden"),
        _response("https://inference-api.nvidia.com/v1/chat/completions", "200 OK"),
        _response(VICTIM, "401 Unauthorized"),
    )

    assert diagnose_scan(ScanEvidence(attempts=1, answered=1), log, target_uri=VICTIM) is None


def test_attacks_the_victim_never_answered_mean_an_unreachable_victim() -> None:
    failure = diagnose_scan(ScanEvidence(attempts=3, answered=0), _log(), target_uri=VICTIM)

    assert failure is not None
    assert failure.code == "victim_unreachable"
    assert "sent 3 attack(s)" in failure.message


def test_evidence_counts_each_attempt_once_and_collects_its_targets(tmp_path: Path) -> None:
    report = tmp_path / "agent-breaker.report.jsonl"
    rows: list[Any] = [
        {"entry_type": "init"},
        *(
            {
                "entry_type": "attempt",
                "uuid": "a",
                "status": s,
                "outputs": [{"text": "done"}],
                "notes": {"current_target": "run_sql"},
            }
            for s in (1, 2)
        ),
        {"entry_type": "attempt", "uuid": "b", "outputs": [None, {"text": "  "}], "notes": {"current_target": "x"}},
        {"entry_type": "attempt", "uuid": "c", "outputs": ["plain text"], "notes": None},
    ]
    report.write_text("\n".join([*(json.dumps(row) for row in rows), "not json", "[]"]), encoding="utf-8")

    assert read_scan_evidence(report) == ScanEvidence(attempts=3, answered=2, targets=("run_sql", "x"))
    assert read_scan_evidence(None) == ScanEvidence()


def test_the_detailed_log_is_found_from_garaks_stdout(tmp_path: Path) -> None:
    detail = tmp_path / "garak.detail.log"
    stdout = tmp_path / "garak-agent-breaker.log"
    stdout.write_text(f"garak LLM vulnerability scanner\n📜 logging to {detail}\n", encoding="utf-8")

    stdout.write_text("garak LLM vulnerability scanner\n", encoding="utf-8")
    assert detail_log_path(stdout) is None  # never announced
    stdout.write_text(f"garak LLM vulnerability scanner\n📜 logging to {detail}\n", encoding="utf-8")
    assert detail_log_path(stdout) is None  # announced but never written
    detail.write_text("x", encoding="utf-8")
    assert detail_log_path(stdout) == detail
    assert detail_log_path(tmp_path / "missing.log") is None


# --- the attacker ----------------------------------------------------------------------------------------


def _write_report(path: Path, attempts: int) -> str:
    rows = [{"entry_type": "attempt", "uuid": str(i), "outputs": [{"text": "ok"}]} for i in range(attempts)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scans: list[tuple[int, str, int]],
    hits: list[dict[str, Any]] | None = None,
    stdout: str = "boom",
) -> tuple[AttackRecord, int]:
    """Run the attacker over scripted scans of ``(attempts, detailed log, exit code)``."""
    monkeypatch.chdir(tmp_path)
    detail = tmp_path / "detail.log"
    calls = 0

    async def fake_scan(**kwargs: Any) -> run_agent_breaker.ScanResult:
        nonlocal calls
        attempts, log, code = scans[calls]
        calls += 1
        report_dir = tmp_path / kwargs["report_dir"]
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / run_agent_breaker.STDOUT_LOG_NAME).write_text(
            f"📜 logging to {detail}\n{stdout}\n", encoding="utf-8"
        )
        detail.write_text(log, encoding="utf-8")
        report = _write_report(tmp_path / f"report-{calls}.jsonl", attempts)
        return run_agent_breaker.ScanResult(
            return_code=code, hits=hits or [], config_path="/c.yaml", report_path=report
        )

    monkeypatch.setattr(run_agent_breaker, "run_agent_breaker_scan", fake_scan)
    request = AgentRunInput(round_id="r1", target=TargetInput(name="victim", base_url=VICTIM))
    record = asyncio.run(run_agent_breaker.run(request, AgentConfig(name="agent-breaker", role="attacker")))
    return record, calls


def test_an_intermittent_empty_scan_is_run_once_more(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record, scans = _run(monkeypatch, tmp_path, [(0, _log(UNPARSED_ANALYSIS), 0), (2, _log(), 0)])

    assert scans == 2
    assert record.ok is True
    assert record.metadata["retried_after"] == "tool_analysis_unparsed"
    assert record.metadata["attempts"] == 2


def test_an_empty_scan_that_recurs_fails_with_its_cause(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record, scans = _run(monkeypatch, tmp_path, [(0, _log(UNPARSED_ANALYSIS), 0)] * 2)

    assert scans == 2
    assert record.ok is False
    assert record.error is not None
    assert record.error.startswith("[tool_analysis_unparsed] garak's attacker model returned malformed JSON")
    assert record.error.endswith("(also after one retry)")
    assert record.metadata["failure_code"] == "tool_analysis_unparsed"
    assert record.metadata["remediation"]


def test_a_cause_that_will_not_change_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_tools = _log("2026-11-04 18:25:30,001  WARNING  AgentBreaker # No tools found -- cannot run attack")
    record, scans = _run(monkeypatch, tmp_path, [(0, no_tools, 0)])

    assert scans == 1
    assert record.metadata["failure_code"] == "victim_disclosed_no_tools"


def test_a_crash_with_no_hits_fails_with_garaks_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record, _ = _run(monkeypatch, tmp_path, [(1, _log(), 2)])

    assert record.ok is False
    assert record.metadata["failure_code"] == "garak_crashed"
    assert record.error is not None
    assert "exited 2 after 1 attack(s)" in record.error
    assert "boom" in record.error


def test_a_victim_turn_that_outruns_request_timeout_is_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The generic crash text hid the cause: garak 0.17 does not catch the victim's read timeout.

    The cap named is the one garak ran with, here set outside the attacker's config (env or scaffold).
    """
    traceback = """\
Traceback (most recent call last):
  File "/venv/lib/python3.12/site-packages/garak/generators/rest.py", line 349, in _call_model
    resp = self.http_function(self.uri, **req_kArgs)
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=1200)"""
    record, _ = _run(monkeypatch, tmp_path, [(2, _log(), 1)], stdout=traceback)

    assert record.ok is False
    assert record.metadata["failure_code"] == "victim_turn_timeout"
    assert record.error is not None
    assert "after 2 attack(s)" in record.error
    assert "1200s (garak.request_timeout)" in record.error


def test_hits_are_kept_when_the_report_counts_no_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hit = [{"probe": "agent_breaker.AgentBreaker"}]
    record, _ = _run(monkeypatch, tmp_path, [(0, _log(), 0)], hits=hit)

    assert record.ok is True
    assert record.error is None
    assert len(record.records) == 1


def test_hits_do_not_excuse_rejected_model_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hit = [{"probe": "agent_breaker.AgentBreaker"}]
    record, _ = _run(
        monkeypatch,
        tmp_path,
        [
            (
                0,
                _log(
                    _response("https://integrate.api.nvidia.com/v1/chat/completions", "401 Unauthorized"),
                    UNPARSED_ANALYSIS,
                ),
                0,
            )
        ],
        hits=hit,
    )

    assert record.ok is False
    assert record.metadata["failure_code"] == ATTACKER_LLM_UNAVAILABLE


def test_garaks_detailed_log_goes_beside_the_report_and_old_reports_are_ignored(tmp_path: Path) -> None:
    report_dir = tmp_path / "r"
    report_dir.mkdir()
    stale = report_dir / "agent-breaker.hitlog.jsonl"
    stale.write_text('{"stale": true}\n', encoding="utf-8")
    os.utime(stale, (0, 0))
    envs: list[dict[str, str] | None] = []

    async def fake_runner(_command: list[str], _log: Path, _timeout: float | None, env: dict[str, str] | None) -> int:
        envs.append(env)
        return 0

    result = asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri=VICTIM, report_dir=str(report_dir), garak_command=["garak"], runner=fake_runner
        )
    )

    assert envs[0] is not None
    assert envs[0]["GARAK_LOG_FILE"] == str(report_dir / run_agent_breaker.DETAIL_LOG_NAME)
    assert (result.hits, result.hitlog_path) == ([], None)


def test_a_garak_log_the_caller_pinned_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GARAK_LOG_FILE", str(tmp_path / "caller.log"))
    envs: list[dict[str, str] | None] = []

    async def fake_runner(_command: list[str], _log: Path, _timeout: float | None, env: dict[str, str] | None) -> int:
        envs.append(env)
        return 0

    asyncio.run(
        run_agent_breaker.run_agent_breaker_scan(
            target_uri=VICTIM, report_dir=str(tmp_path / "r"), garak_command=["garak"], runner=fake_runner
        )
    )

    assert envs[0] is not None
    assert envs[0]["GARAK_LOG_FILE"] == str(tmp_path / "caller.log")


# --- the attack stage ------------------------------------------------------------------------------------


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


def _failed(code: str | None) -> AttackRecord:
    metadata = {"failure_code": code, "remediation": f"fix {code}"} if code else {}
    return AttackRecord(agent_id="a", agent_name="garak", ok=False, error=f"[{code}] why", metadata=metadata)


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("victim_unreachable", VictimUnavailableError),
        (ATTACKER_LLM_UNAVAILABLE, ModelUnavailableError),
        ("tool_analysis_unparsed", AttackerError),
    ],
)
def test_the_attack_stage_raises_the_error_the_failure_code_names(code: str, error: type[Exception]) -> None:
    with pytest.raises(error, match=rf"\[{code}\] why") as raised:
        raise_on_attacker_failure([_Agent("garak")], [_failed(code)])

    assert type(raised.value) is error
    assert getattr(raised.value, "remediation", None) == f"fix {code}"


def test_an_undiagnosed_failure_keeps_the_default_remediation() -> None:
    with pytest.raises(AttackerError) as raised:
        raise_on_attacker_failure([_Agent("garak")], [_failed(None)])

    assert raised.value.remediation == AttackerError.default_remediation
