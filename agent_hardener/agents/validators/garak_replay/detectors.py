# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector-config assembly for the garak replay validator.

Builds the ``{"detectors": ...}`` config root handed to the garak worker, and resolves a bundled scan
config to seed detector model settings. ``ReplayConfig`` is used only in annotations, so it is imported
under ``TYPE_CHECKING`` — no runtime import cycle with the validator module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from agent_hardener.agents.attackers.agent_breaker.config import build_agent_breaker_config
from agent_hardener.agents.validators.garak_replay import parsing
from agent_hardener.env import GARAK_REPO_PATH

if TYPE_CHECKING:
    from agent_hardener.agents.validators.garak_replay.config import ReplayConfig
    from agent_hardener.agents.validators.garak_replay.parsing import AttackType


def detector_config_root(attack_type: AttackType, config: ReplayConfig) -> dict[str, Any]:
    namespace, class_name = parsing.detector_namespace_and_class(attack_type)
    root: dict[str, Any] = {"detectors": {namespace: {class_name: {}}}}
    path = (
        config.direct_detector_config if attack_type == "direct_prompt_injection" else config.indirect_detector_config
    )
    if path is not None:
        # Explicit override file wins (same shape as a garak scan config).
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError:
            loaded = {}
        if isinstance(loaded, dict):
            file_detector_config = (loaded.get("plugins") or {}).get("detectors") or {}
            if isinstance(file_detector_config, dict):
                root = {"detectors": file_detector_config}
    elif attack_type == "direct_prompt_injection":
        # Seed the agent_breaker detector model settings from the shared config builder so the
        # internal endpoint/model defaults match the attacker (single source of truth).
        built = build_agent_breaker_config(target_uri="", report_dir="", report_prefix="")
        root = {"detectors": built["plugins"]["detectors"]}

    class_config = root.setdefault("detectors", {}).setdefault(namespace, {}).setdefault(class_name, {})
    if config.detector_model_type:
        class_config["detector_model_type"] = config.detector_model_type
    if config.detector_model_name:
        class_config["detector_model_name"] = config.detector_model_name
    if config.detector_model_config:
        class_config["detector_model_config"] = config.detector_model_config
    return root


def default_detector_config(garak_repo_path: Path | None, scan_yaml_name: str) -> Path | None:
    """Resolve a bundled garak scan config to seed detector model settings from.

    Searches the configured garak repo and the ``GARAK_REPO_PATH`` env override (a dev editable
    checkout). Returns ``None`` when no bundled config is found (callers then fall back to the
    config-builder defaults / garak's own defaults).
    """
    candidates: list[Path] = []
    if garak_repo_path is not None:
        candidates.append(garak_repo_path)
    env_path = os.getenv(GARAK_REPO_PATH)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for base in candidates:
        candidate = base / scan_yaml_name
        if candidate.is_file():
            return candidate
    return None
