# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for rendering the Agent Hardener final run log from report.json artifacts.

Discovering and loading reports is a ``final_log`` (data) concern; formatting is delegated to
:mod:`agent_hardener.display.final`. This entry point is the composition root that wires the two together
— it is the only place in the ``final_log`` package that imports ``display``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_hardener.display.final import render_final_run_log
from agent_hardener.final_log import load_json, path_mtime


def find_report_paths(
    *,
    root: Path = Path(".agent-hardener"),
    marker_path: Path | None = None,
    mission_ids: list[str] | None = None,
) -> list[Path]:
    """Find report.json artifacts for a wrapper run."""
    marker_mtime = path_mtime(marker_path)
    if not root.exists():
        return []
    paths = []
    for path in root.rglob("report.json"):
        if marker_mtime is not None and path_mtime(path) is not None and path_mtime(path) < marker_mtime:
            continue
        path_text = str(path)
        if mission_ids and not any(mission_id in path_text for mission_id in mission_ids):
            continue
        paths.append(path)
    return sorted(paths)


def render_final_run_log_from_paths(
    report_paths: list[Path],
    *,
    marker_path: Path | None = None,
    mission_ids: list[str] | None = None,
) -> str:
    """Load report JSON files and render the final run log."""
    reports = [load_json(path) for path in report_paths]
    return render_final_run_log(reports, marker_path=marker_path, mission_ids=mission_ids)


def main(argv: list[str] | None = None) -> int:
    """Render a final log from report paths or the local .agent-hardener tree."""
    parser = argparse.ArgumentParser(description="Render an Agent Hardener final run log.")
    parser.add_argument("--report", action="append", type=Path, default=[], help="Path to a report.json artifact.")
    parser.add_argument("--marker", type=Path, help="Only include artifacts newer than this marker file.")
    parser.add_argument("--mission-id", action="append", default=[], help="Mission id to include.")
    parser.add_argument(
        "--root", type=Path, default=Path(".agent-hardener"), help="Root to search for report.json files."
    )
    args = parser.parse_args(argv)

    report_paths = args.report or find_report_paths(
        root=args.root,
        marker_path=args.marker,
        mission_ids=args.mission_id or None,
    )
    if not report_paths:
        print("Agent Hardener final log")
        print("=" * 80)
        print("No report.json artifacts found for this run.")
        return 0

    print(
        render_final_run_log_from_paths(
            report_paths,
            marker_path=args.marker,
            mission_ids=args.mission_id or None,
        ),
        end="",
    )
    return 0
