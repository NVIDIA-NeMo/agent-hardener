# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rich (terminal) rendering of the Agent Hardener final report."""

from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent_hardener.display.console import get_console, is_rich
from agent_hardener.display.final._summary_data import (
    ResultDetail,
    collect_defenders,
    collect_validators,
    result_detail,
    session_overview,
    validator_summary,
)
from agent_hardener.display.final.render_plain import (
    render_attack_results,
    render_final_run_log,
    render_garak_artifacts,
    render_policy_results,
    render_session_comparison,
    render_session_overview,
    render_validator_results,
)
from agent_hardener.display.render_diff import diff_renderable, diff_stat, load_yaml_diff_from_patch
from agent_hardener.display.renderers.garak_attacker import (
    _attacker_table,
    _transcript_conversation,
)
from agent_hardener.display.renderers.garak_replay import _validator_conversation
from agent_hardener.final_log import (
    collect_attempt_transcripts,
    collect_final_log_data,
    iterations,
    policy_patches,
    short_text,
)
from agent_hardener.yaml_diff import YamlDiff


def build_summary_renderable(
    reports: list[Any],
    *,
    verbose: bool = False,
    mission_ids: list[str] | None = None,
) -> Group:
    """Build a rich renderable for the final report.

    Pure presentation over :class:`FinalLogData`. The compact (non-verbose) view is the
    overview + attacker tables (plus defender/validator tables when those agents ran).
    ``verbose`` renders the per-phase transcripts and detail first — attack, defense, then the
    validator replay conversations — and collects every summary table at the very end, so the
    headline numbers sit below the full transcripts rather than scrolling away above them.
    """
    data = collect_final_log_data(reports)
    defenders = collect_defenders(data.report_dicts)
    validators = collect_validators(data.report_dicts)
    header = Rule("Agent Hardener — Final Report")

    if not verbose:
        parts: list[Any] = [
            header,
            _overview_table(data.report_dicts, mission_ids),
            Text(""),
            _attacker_table(data.stats),
        ]
        if defenders:
            parts += [Text(""), _defender_table(defenders)]
        if validators:
            parts += [Text(""), _validator_table(validators)]
        parts.append(Text("\nRun with --verbose for prompts, responses, policy, and Garak detail.", style="dim"))
        return Group(*parts)

    # Verbose: compose the same per-section renderables used by the tabbed TUI, separated by rules so
    # the paged/inline fallback reads as one document.
    parts = [header]
    for title, renderable in build_report_sections(reports, mission_ids=mission_ids).items():
        parts += [Rule(title), renderable]
    return Group(*parts)


def _detail(lines: list[str]) -> Text:
    """Join plain-text detail lines (from the ``render_plain`` helpers) into a single ``Text``."""
    return Text("\n".join(lines).rstrip())


def build_report_sections(
    reports: list[Any],
    *,
    mission_ids: list[str] | None = None,
) -> dict[str, Group]:
    """Build the verbose report as one renderable per section (ordered: tab order for the TUI).

    Each value is a self-contained Rich renderable reusing the same builders as the paged report, so
    the tabbed TUI and the inline/pager fallback render identical content.
    """
    data = collect_final_log_data(reports)
    defenders = collect_defenders(data.report_dicts)
    validators = collect_validators(data.report_dicts)

    # Overview: session table + before/after round comparison.
    overview: list[Any] = [_overview_table(data.report_dicts, mission_ids)]
    comparison = render_session_comparison(data.report_dicts)
    if comparison:
        overview += [Text(""), _detail(comparison)]

    # Attack: attacker↔victim transcript, then Garak artifact detail.
    attack: list[Any] = []
    transcripts = collect_attempt_transcripts(data.garak_files)
    if transcripts:
        attack += [
            Text("Attacker ↔ victim conversation (all attempts):", style="bold"),
            _transcript_conversation(transcripts),
        ]
    attack.append(_detail(render_garak_artifacts(data.stats, data.garak_files)))

    # Defense: policy/guardrail detail, then GitHub-style previous → current YAML diffs.
    defense: list[Any] = [_detail(render_policy_results(data.report_dicts))]
    change_parts = _workflow_change_renderables(data.report_dicts)
    if change_parts:
        defense.append(Text("\nWorkflow / policy changes (previous → current):", style="bold"))
        defense += change_parts

    # Validation: detail summary, then replay conversations.
    validation: list[Any] = [_detail(render_validator_results(data.report_dicts))]
    validator_chat = _validator_conversation(validators)
    if validator_chat is not None:
        validation += [Text("\nValidator replay conversations:", style="bold"), validator_chat]

    # Tables: the headline summary tables.
    tables: list[Any] = [_attacker_table(data.stats)]
    if defenders:
        tables += [Text(""), _defender_table(defenders)]
    if validators:
        tables += [Text(""), _validator_table(validators)]

    return {
        "Overview": Group(*overview),
        "Attack": Group(*attack),
        "Defense": Group(*defense),
        "Validation": Group(*validation),
        "Tables": Group(*tables),
    }


def _overview_table(reports: list[dict[str, Any]], mission_ids: list[str] | None) -> Table:
    """Build the per-session overview table."""
    overview = session_overview(reports, mission_ids)
    caption = (
        f"missions: {', '.join(overview.missions) if overview.missions else '<none>'}  |  "
        f"sessions: {overview.total} ({overview.succeeded} succeeded, {overview.failed} failed)"
    )
    table = Table(title="Session overview", title_justify="left", caption=caption, caption_justify="left")
    table.add_column("Session")
    table.add_column("Success", justify="center")
    table.add_column("Iterations", justify="right")
    table.add_column("Storage", overflow="fold")
    if not overview.rows:
        table.add_row("<no reports loaded>", "", "", "")
    for row in overview.rows:
        table.add_row(row.round_id, _success_glyph(row.success), str(row.iterations), row.storage)
    return table


def _detail_text(detail: ResultDetail, limit: int) -> Text:
    """Render a :class:`ResultDetail` as a (red-if-error) ``Text`` cell."""
    return Text(short_text(detail.text, limit), style="red" if detail.is_error else "")


def _defender_table(defenders: list[dict[str, Any]]) -> Table:
    """Per-defender results table (status, patches yielded, YAML changes, summary/error)."""
    table = Table(title="Defender results", title_justify="left")
    table.add_column("Defender")
    table.add_column("Status", justify="center")
    table.add_column("Patches", justify="right")
    table.add_column("Changes", justify="right")
    table.add_column("Detail", overflow="fold")
    for defender in defenders:
        patches = len(defender.get("policy_patches") or [])
        table.add_row(
            str(defender.get("agent_name") or defender.get("agent_id") or "<unknown>"),
            _success_glyph(defender.get("ok", True)),
            Text(str(patches), style="green" if patches else "dim"),
            diff_stat(_defender_change_diff(defender)),
            _detail_text(result_detail(defender), 100),
        )
    return table


def _defender_change_diff(defender: dict[str, Any]) -> YamlDiff:
    """Combine a defender's per-patch YAML diffs into one stat-bearing diff (adds/removes summed)."""
    combined = YamlDiff()
    for patch in defender.get("policy_patches") or []:
        if not isinstance(patch, dict):
            continue
        resolved = load_yaml_diff_from_patch(patch)
        if resolved is not None:
            _, diff = resolved
            combined.hunks.extend(diff.hunks)
            combined.added += diff.added
            combined.removed += diff.removed
    return combined


def _workflow_change_renderables(reports: list[dict[str, Any]]) -> list[Any]:
    """Build the verbose 'Workflow / policy changes' section: a colored diff per changed patch."""
    parts: list[Any] = []
    for report in reports:
        for iteration in iterations(report):
            for patch in policy_patches(iteration):
                resolved = load_yaml_diff_from_patch(patch)
                if resolved is None:
                    continue
                label, diff = resolved
                parts += [Text(f"\n{label}", style="bold"), diff_renderable(diff, more_hint=label)]
    return parts


def _validator_table(validators: list[dict[str, Any]]) -> Table:
    """Per-validator results table (kind, status, summary/error) with per-kind tallies as caption."""
    summary = validator_summary(validators)
    table = Table(
        title="Validator results",
        title_justify="left",
        caption=summary.caption or None,
        caption_justify="left",
    )
    table.add_column("Validator")
    table.add_column("Kind")
    table.add_column("Status", justify="center")
    table.add_column("Result", overflow="fold")
    for row in summary.rows:
        table.add_row(row.name, row.kind, _success_glyph(row.ok), _detail_text(row.detail, 120))
    return table


def _success_glyph(value: Any) -> Text:
    """Return a colored ✓/✗ for a success flag (plain ``?`` when unknown)."""
    if value is True:
        return Text("✓", style="green")
    if value is False:
        return Text("✗", style="red")
    return Text("?")


def _try_report_tui(reports: list[Any], *, mission_ids: list[str] | None) -> bool:
    """Launch the tabbed report TUI; return True on success, False to fall back to the pager.

    Textual is imported lazily so a missing/broken TUI dependency never blocks the (already-on-disk)
    report — any failure is swallowed and the caller pages the report instead.
    """
    try:
        from agent_hardener.display.final.report_tui import run_report_tui  # noqa: PLC0415

        sections = build_report_sections(reports, mission_ids=mission_ids)
        run_report_tui(sections)
    except Exception:  # display must never crash a completed run
        logging.getLogger("agent_hardener.session").debug("report TUI failed; falling back to pager", exc_info=True)
        return False
    return True


def print_final_summary(
    reports: list[Any],
    *,
    verbose: bool = False,
    mission_ids: list[str] | None = None,
) -> None:
    """Print the final report; rich tables on a TTY, plain text otherwise.

    ``verbose`` adds the full detail sections (rich) / the complete plain log; the default shows only
    the headline overview + attacker tables.
    """
    if is_rich():
        console = get_console()
        if verbose:
            # The verbose report is long (transcripts, Garak tables, YAML diffs). On an interactive
            # TTY, show it in a tabbed TUI (Overview/Attack/Defense/Validation/Tables) so sections are
            # navigable instead of one wall of text. If the TUI is unavailable or fails to launch,
            # fall back to paging the single renderable. A display failure must never crash a
            # completed run — the full report is already on disk in agent-hardener.log.
            if sys.stdout.isatty() and _try_report_tui(reports, mission_ids=mission_ids):
                return
            renderable = build_summary_renderable(reports, verbose=True, mission_ids=mission_ids)
            try:
                with console.pager(styles=True):
                    console.print(renderable)
            except (BrokenPipeError, OSError):
                console.print(renderable)
        else:
            console.print(build_summary_renderable(reports, verbose=False, mission_ids=mission_ids))
    elif verbose:
        print(render_final_run_log(reports, mission_ids=mission_ids), end="")
    else:
        # Compact plain fallback: header + the headline sections, reusing the text renderers. Mirror the
        # rich compact view — include the defender/validator sections when those agents ran, so plain/CI
        # output isn't missing them.
        data = collect_final_log_data(reports)
        lines = [
            "Agent Hardener final log",
            "=" * 80,
            *render_session_overview(data.report_dicts, mission_ids),
            "",
            *render_attack_results(data.stats),
        ]
        if collect_defenders(data.report_dicts):
            lines += ["", *render_policy_results(data.report_dicts)]
        if collect_validators(data.report_dicts):
            lines += ["", *render_validator_results(data.report_dicts)]
        print("\n".join(lines).rstrip() + "\n", end="")
