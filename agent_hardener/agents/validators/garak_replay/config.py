# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for the garak replay validator: the ``ReplayConfig`` dataclass and its builder.

The leaf modules (``detectors``, ``verdict``, ``redaction``) import ``ReplayConfig`` from here, so this
module is the single home for replay configuration — it never imports the validator orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from agent_hardener.agents.validators.garak_replay import detectors
from agent_hardener.concurrency import agent_concurrency

if TYPE_CHECKING:
    from agent_hardener.models import AgentConfig, AgentRunInput, ValidatorKind

DEFAULT_CONFIDENCE_CUTOFF = 0.7
DEFAULT_EXCERPT_CHARS = 500
DEFAULT_TIMEOUT_SECONDS = 240.0

# The garak detector DEFAULT_PARAMS point the evaluation model at the public
# integrate.api.nvidia.com endpoint, which rejects our key with HTTP 401. For the direct
# (agent_breaker) detector we seed the model settings from the same config builder the attacker
# uses, so the internal endpoint/model live in exactly one place. Indirect injection (unused) can
# still seed from a bundled scan config found via GARAK_REPO_PATH for dev editable checkouts.
INDIRECT_DETECTOR_SCAN_YAML = "scan_indirect.yaml"


@dataclass(frozen=True)
class ReplayConfig:
    """Runtime configuration for attack replay validation."""

    replay_url: str
    replay_mode: Literal["openai_chat", "json"]
    model: str
    input_field: str
    response_json_path: str | None
    timeout_seconds: float
    confidence_cutoff: float
    excerpt_chars: int
    redact_outputs: bool
    store_full_replay: bool
    garak_repo_path: Path | None
    direct_detector_config: Path | None
    indirect_detector_config: Path | None
    detector_model_type: str | None
    detector_model_name: str | None
    detector_model_config: dict[str, Any]
    github_comments_enabled: bool
    github_token: str | None
    concurrency: int


def build_config(request: AgentRunInput, agent: AgentConfig) -> ReplayConfig:
    configured_url = str(
        agent.config.get("replay_url") or request.context.get("replay_url") or request.target.base_url or ""
    )
    replay_mode = str(agent.config.get("replay_mode") or _default_replay_mode(configured_url))
    if replay_mode not in {"openai_chat", "json"}:
        msg = "validator replay_mode must be 'openai_chat' or 'json'"
        raise ValueError(msg)

    garak_repo_path = _optional_path(agent.config.get("garak_repo_path"))
    # Direct (agent_breaker) detector settings are seeded from the shared config builder, so only an
    # explicit override file is honoured here. Indirect (unused) still resolves a bundled scan config.
    direct_detector_config = _optional_path(agent.config.get("direct_detector_config"))
    indirect_detector_config = _optional_path(
        agent.config.get("indirect_detector_config")
    ) or detectors.default_detector_config(garak_repo_path, INDIRECT_DETECTOR_SCAN_YAML)

    return ReplayConfig(
        replay_url=configured_url,
        replay_mode=cast("Literal['openai_chat', 'json']", replay_mode),
        model=str(agent.config.get("model") or "garak"),
        input_field=str(agent.config.get("input_field") or "input_message"),
        response_json_path=_optional_str(agent.config.get("response_json_path")),
        timeout_seconds=float(agent.config.get("timeout_seconds") or agent.timeout_seconds or DEFAULT_TIMEOUT_SECONDS),
        confidence_cutoff=float(agent.config.get("confidence_cutoff") or DEFAULT_CONFIDENCE_CUTOFF),
        excerpt_chars=int(agent.config.get("excerpt_chars") or DEFAULT_EXCERPT_CHARS),
        redact_outputs=bool(agent.config.get("redact_outputs", True)),
        store_full_replay=bool(agent.config.get("store_full_replay", False)),
        garak_repo_path=garak_repo_path,
        direct_detector_config=direct_detector_config,
        indirect_detector_config=indirect_detector_config,
        detector_model_type=_optional_str(agent.config.get("detector_model_type")),
        detector_model_name=_optional_str(agent.config.get("detector_model_name")),
        detector_model_config=_optional_mapping(agent.config.get("detector_model_config")),
        github_comments_enabled=bool(agent.config.get("github_comments_enabled", True)),
        github_token=_github_token(agent),
        concurrency=agent_concurrency(
            agent.config,
            request.context,
            "attack_validator_hits",
            "AGENT_HARDENER_ATTACK_VALIDATOR_CONCURRENCY",
        ),
    )


def validator_kind(request: AgentRunInput, agent: AgentConfig) -> ValidatorKind:
    configured = agent.config.get("kind")
    if configured in {"attack", "benign"}:
        return cast("ValidatorKind", configured)
    return request.validator_kind or "attack"


def _default_replay_mode(replay_url: str) -> str:
    if "/v1/chat/completions" in replay_url:
        return "openai_chat"
    return "json"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _optional_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _github_token(agent: AgentConfig) -> str | None:
    env_vars = agent.config.get("github_token_env_vars") or ("GITHUB_TOKEN", "GITHUB_PAT", "GH_TOKEN")
    for env_var in env_vars:
        token = os.getenv(str(env_var))
        if token:
            return token
    return None
