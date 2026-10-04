# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-component file writers for Swarm Tracker."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from agent_hardener.storage import write_json

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def write_component_output(attempt_dir: Path, role: str, name: str, output: Any) -> None:
    """Write <attempt_dir>/<role>_<name>.json with the component's output."""
    path = attempt_dir / f"{_safe(role)}_{_safe(name)}.json"
    write_json(path, output)


def write_run_config(run_dir: Path, config: Any) -> None:
    """Write <run_dir>/run_config.json. Idempotent — skips if the file already exists."""
    path = run_dir / "run_config.json"
    if path.exists():
        return
    write_json(path, config)


def _safe(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("_") or "unknown"
    if not _SAFE_NAME.fullmatch(normalized):
        normalized = "unknown"
    return normalized
