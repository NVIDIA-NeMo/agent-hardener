# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure parsers over garak hit records — attack-type, prompts, and GitHub-issue targets.

Config-free leaf helpers (they take a hit/prompt/url, never a ``ReplayConfig``), so they live outside
the validator module without a circular import. ``garak_attack_replay`` re-exports the public names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_hardener.models import AttackRecord

AttackType = Literal["direct_prompt_injection", "indirect_prompt_injection"]

DIRECT_PROBE_KEYWORD = "agent_breaker"
INDIRECT_PROBE_KEYWORD = "indirect_injection"


@dataclass(frozen=True)
class GitHubIssueTarget:
    """A GitHub issue whose comments should be watched for side effects."""

    owner: str
    repo: str
    issue_number: int

    @property
    def label(self) -> str:
        return f"{self.owner}/{self.repo}#{self.issue_number}"


def iter_supported_hits(attacks: Iterable[AttackRecord]) -> Iterable[dict[str, Any]]:
    """Yield a hit-context dict for every record whose attack type is recognised."""
    for attack in attacks:
        for record_index, hit in enumerate(attack.records):
            attack_type = identify_attack_type(hit)
            if attack_type is None:
                continue
            yield {
                "attack": attack,
                "attacker_name": attack.agent_name,
                "attacker_id": attack.agent_id,
                "record_index": record_index,
                "attack_type": attack_type,
                "hit": hit,
            }


def identify_attack_type(hit: dict[str, Any]) -> AttackType | None:
    haystack = " ".join(
        str(value or "")
        for value in (
            hit.get("source"),
            hit.get("probe"),
            hit.get("detector"),
        )
    ).lower()
    if DIRECT_PROBE_KEYWORD in haystack or "agent-breaker" in haystack:
        return "direct_prompt_injection"
    if INDIRECT_PROBE_KEYWORD in haystack or "indirect-injection" in haystack:
        return "indirect_prompt_injection"
    return None


def extract_prompt(hit: dict[str, Any]) -> str:
    prompt = hit.get("prompt")
    if isinstance(prompt, dict):
        turns = prompt.get("turns")
        if isinstance(turns, list):
            for turn in turns:
                text = (turn.get("content") or {}).get("text") if isinstance(turn, dict) else None
                if isinstance(text, str) and text:
                    return text
    if isinstance(prompt, str):
        return prompt
    return ""


def extract_indirect_prompt(hit: dict[str, Any]) -> str:
    triggers = hit.get("triggers")
    if isinstance(triggers, dict):
        trigger_query = triggers.get("trigger_query")
        if isinstance(trigger_query, str) and trigger_query:
            return trigger_query
    return extract_prompt(hit)


def parse_target_tool(prompt: str) -> str:
    match = re.search(r"(?i)\buse\s+the\s+([A-Za-z0-9_.-]+)\s+tool\b", prompt)
    if match:
        return match.group(1)
    return "unknown"


def detector_namespace_and_class(attack_type: AttackType) -> tuple[str, str]:
    if attack_type == "direct_prompt_injection":
        return "agent_breaker", "AgentBreakerResult"
    return "indirect_injection", "IndirectInjectionResult"


def watched_github_targets(injection_location: str, injected_payload: str) -> list[GitHubIssueTarget]:
    base_target = parse_github_issue_url(injection_location)
    if base_target is None:
        return []

    targets = {base_target.label: base_target}
    for issue_number in explicit_issue_numbers(injected_payload):
        target = GitHubIssueTarget(base_target.owner, base_target.repo, issue_number)
        targets[target.label] = target
    return list(targets.values())


def parse_github_issue_url(url: str) -> GitHubIssueTarget | None:
    match = re.search(r"github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/issues/(?P<number>\d+)", url)
    if match is None:
        return None
    return GitHubIssueTarget(
        owner=match.group("owner"),
        repo=match.group("repo"),
        issue_number=int(match.group("number")),
    )


def explicit_issue_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in re.finditer(r"(?i)\bissues?[ \t]+#(\d+)\b|\bissues?[ \t]+number[ \t]+(\d+)\b", text):
        number = match.group(1) or match.group(2)
        numbers.add(int(number))
    return numbers


def issue_label(url: str) -> str:
    target = parse_github_issue_url(url)
    if target is None:
        return url
    return target.label
