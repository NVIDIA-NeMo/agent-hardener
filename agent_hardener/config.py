# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON and YAML configuration loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from agent_hardener.models import AgentRole, SessionConfig, ValidatorKind

logger = logging.getLogger(__name__)


def load_config(path: str | Path, *, relay_plugins: Path | None = None, policy: Path | None = None) -> SessionConfig:
    """Load an orchestrator config from JSON or YAML.

    Accepts either a thin agent manifest (top-level ``agent:`` key) — expanded into a full
    :class:`SessionConfig` — or a full session config (back-compat).

    ``relay_plugins`` / ``policy`` override the run's initial Relay guardrail set and OpenShell policy; they are
    only meaningful for a manifest, whose expansion wires them into every derived key.
    """
    config_path = Path(path)
    raw = _read_config_data(config_path)
    if "agent" in raw:
        from agent_hardener.manifest import expand_manifest_data  # noqa: PLC0415 - avoid import cycle.

        return expand_manifest_data(raw, base_dir=config_path.parent, relay_plugins=relay_plugins, policy=policy)
    if relay_plugins is not None or policy is not None:
        msg = "--relay-plugins/--policy require a manifest (agent:) config, not a full session config"
        raise ValueError(msg)
    return parse_config_data(raw, base_dir=config_path.parent)


def parse_config_data(data: dict[str, Any], base_dir: str | Path | None = None) -> SessionConfig:
    """Parse raw config data into a validated session config."""
    if not isinstance(data, dict):
        msg = "config root must be a mapping"
        raise ValueError(msg)
    prepared = _prepare_config(data)
    context = {"base_dir": Path(base_dir)} if base_dir is not None else None
    return SessionConfig.model_validate(prepared, context=context)


def _read_config_data(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as config_file:
        if suffix == ".json":
            data = json.load(config_file)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(config_file)
        else:
            msg = f"unsupported config file type: {path.suffix}"
            raise ValueError(msg)
    if not isinstance(data, dict):
        msg = "config root must be a mapping"
        raise ValueError(msg)
    return data


def _prepare_config(data: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(data)
    if isinstance(prepared.get("storage"), str):
        prepared["storage"] = {"root_dir": prepared["storage"]}
    if "storage_dir" in prepared:
        if "storage" in prepared:
            logger.warning("config has both 'storage' and 'storage_dir'; 'storage_dir' will be ignored")
        else:
            prepared["storage"] = {"root_dir": prepared.pop("storage_dir")}

    prepared["attackers"] = _prepare_agent_list(prepared.get("attackers", []), "attacker")
    prepared["defenders"] = _prepare_agent_list(prepared.get("defenders", []), "defender")
    if "victim" not in prepared:
        msg = "config must include a 'victim' section"
        raise ValueError(msg)
    prepared["victim"] = _prepare_agent(prepared["victim"], "victim")
    prepared["attack_validators"] = _prepare_agent_list(
        prepared.get("attack_validators", []),
        "validator",
        "attack",
    )
    prepared["benign_validators"] = _prepare_agent_list(
        prepared.get("benign_validators", []),
        "validator",
        "benign",
    )
    return prepared


def _prepare_agent_list(
    agents: list[dict[str, Any]],
    role: AgentRole,
    validator_kind: ValidatorKind | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(agents, list):
        msg = f"{role} agents must be a list"
        raise ValueError(msg)
    return [_prepare_agent(agent, role, validator_kind) for agent in agents]


def _prepare_agent(
    agent: dict[str, Any],
    role: AgentRole,
    validator_kind: ValidatorKind | None = None,
) -> dict[str, Any]:
    if not isinstance(agent, dict):
        msg = "agent config entries must be mappings"
        raise ValueError(msg)

    prepared = dict(agent)
    prepared.setdefault("role", role)
    if validator_kind is not None:
        role_config = dict(prepared.get("config", {}))
        role_config.setdefault("kind", validator_kind)
        prepared["config"] = role_config
    return prepared
