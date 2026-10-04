# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Garak attack replay validator display renderer."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.rule import Rule
from rich.text import Text

from agent_hardener.display.helpers import verbose_result_lines
from agent_hardener.display.renderers.garak_attacker import _exchange

if TYPE_CHECKING:
    from agent_hardener.display.context import DisplayContext

_ATTACK_VERDICT_STATUS = {"not_blocked": "NOT_BLOCKED", "blocked": "BLOCKED", "error": "ERROR"}
_BENIGN_VERDICT_STATUS = {"refused": "REFUSED", "complied": "COMPLIED", "error": "ERROR"}


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a dict — or its repr-string, as some validators persist nested fields — into a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _validator_exchanges(validator: dict[str, Any]) -> list[Any]:
    """Build per-prompt exchange renderables for one validator, with kind-appropriate labels."""
    metadata = validator.get("metadata") or {}
    items = metadata.get("results") or metadata.get("attack_results") or []
    kind = str(validator.get("kind") or "")
    is_benign = kind == "benign"
    sender = "🧪 benign probe    " if is_benign else "⚔ attacker (red-team)"
    status_map = _BENIGN_VERDICT_STATUS if is_benign else _ATTACK_VERDICT_STATUS
    exchanges = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        verdict_status = _as_mapping(item.get("verdict")).get("status") or ""
        status = status_map.get(verdict_status, verdict_status.upper())
        exchanges.append(
            _exchange(
                index,
                str(item.get("tool") or item.get("probe") or item.get("label") or ""),
                item.get("payload_excerpt") or item.get("prompt_excerpt") or "",
                _as_mapping(item.get("replay")).get("response_excerpt") or "",
                status=status,
                sender_label=sender,
                note=_as_mapping(item.get("verdict")).get("reasoning") or str(item.get("error") or ""),
            )
        )
    return exchanges


def _validator_conversation(validators: list[dict[str, Any]]) -> Group | None:
    """Chat-style replay conversations grouped by kind: attack first, benign second, Rule between them."""
    attack_validators = [v for v in validators if v.get("kind") != "benign"]
    benign_validators = [v for v in validators if v.get("kind") == "benign"]

    blocks: list[Any] = []
    for group_label, group in (("Attack replay", attack_validators), ("Benign replay", benign_validators)):
        group_exchanges: list[Any] = []
        for validator in group:
            exchanges = _validator_exchanges(validator)
            if exchanges:
                name = validator.get("agent_name") or validator.get("agent_id") or "<validator>"
                group_exchanges.append(Text(f"  {name}:", style="dim"))
                group_exchanges.extend(exchanges)
        if group_exchanges:
            if blocks:
                blocks.append(Rule(style="dim"))
            blocks.append(Text(f"\n{group_label}", style="bold"))
            blocks.extend(group_exchanges)

    return Group(*blocks) if blocks else None


def _attack_line(result: dict[str, Any]) -> str:
    probe = result.get("probe", result.get("attack_type", "?"))
    return f"{probe}: {result.get('status', '?')}"


class GarakReplayValidatorRenderer:
    """Render garak attack replay validator summaries."""

    def verbose(self, output: dict[str, Any], ctx: DisplayContext) -> Text | None:
        return Text("\n".join(verbose_result_lines(output, label="attack replay", line_for=_attack_line)))
