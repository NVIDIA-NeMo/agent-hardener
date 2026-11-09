# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One table of what the traces guardrails defender did across a run.

    uv run python -m agent_hardener.agents.defenders.traces_guardrails_defender.summary <run_dir> [--csv out.csv]

Reads every ``outcome.json`` under the directory (a run dir, its ``traces_guardrails_defender/`` folder, or a parent
of several runs) and prints one row per attack, plus totals by status and route.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .observability import OUTCOME_FILENAME

COLUMNS = (
    "round",
    "attack",
    "status",
    "route",
    "tool_name",
    "guardrail_name",
    "grader_score",
    "grader_iterations",
    "lint_warnings",
    "fallback_reason",
    "reason",
)


def collect(root: Path) -> list[dict[str, Any]]:
    """One row per ``outcome.json`` under ``root``."""
    rows = []
    for path in sorted(root.rglob(OUTCOME_FILENAME)):
        outcome = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                **{column: outcome.get(column) for column in COLUMNS},
                "round": path.parent.parent.name,
                "lint_warnings": len(outcome.get("lint_warnings") or []),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = collect(args.root)
    if not rows:
        print(f"no {OUTCOME_FILENAME} under {args.root}", file=sys.stderr)
        return 1
    with args.csv.open("w", newline="") if args.csv else contextlib.nullcontext(sys.stdout) as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = Counter(row["status"] for row in rows)
    routes = Counter(row["route"] for row in rows if row["route"])
    print(
        f"\n{len(rows)} attacks: "
        + ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
        + " | routes: "
        + ", ".join(f"{route}={count}" for route, count in sorted(routes.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
