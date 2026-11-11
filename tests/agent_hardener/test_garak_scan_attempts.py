# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from agent_hardener.agents.attackers.agent_breaker.run_agent_breaker import (
    ScanResult,
    _attempt_count,
    _empty_scan_error,
)


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ('{"entry_type":"completion"}\n', 0),
        ('{"entry_type":"attempt","uuid":"a","status":2}\n', 1),
        (
            '{"entry_type":"attempt","uuid":"a","status":1}\n'
            '{"entry_type":"attempt","uuid":"a","status":2}\n'
            '{"entry_type":"attempt","uuid":"b","status":2}\n',
            2,
        ),
        ('{"entry_type":"attempt"}\n{"entry_type":"attempt"}\n', 2),
        ('\nnot-json\n[]\n{"entry_type":"completion"}\n', 0),
    ],
)
def test_counts_attempts_including_successful_zero_hit_scans(tmp_path: Path, report: str, expected: int) -> None:
    """Status updates must not inflate attempts or turn an empty scan into zero hits."""
    path = tmp_path / "scan.report.jsonl"
    path.write_text(report, encoding="utf-8")
    assert _attempt_count(path) == expected


@pytest.mark.parametrize(
    ("attempt_count", "has_hits", "is_error"),
    [(0, False, True), (1, False, False), (None, False, False), (0, True, False)],
)
def test_empty_scan_does_not_hide_genuine_zero_hits_or_partial_findings(
    attempt_count: int | None, has_hits: bool, is_error: bool
) -> None:
    """Only a proven empty scan fails; actual zero-hit scans and findings remain valid."""
    result = ScanResult(return_code=0, hits=[], config_path="scan.yaml", attempt_count=attempt_count)
    error = _empty_scan_error(result, has_hits=has_hits)
    assert (error is not None) is is_error
    if error:
        assert "no attack attempts" in error
