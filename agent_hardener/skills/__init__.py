# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The customer skill shipped with the package, and the copy ``agent-hardener init`` installs.

The skill lives inside the package so the copy a user gets always matches the version they installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SKILL_NAME = "agent-hardener"

# Kept out of the installed copy: ``tests.json`` is the routing test suite, for this repo only.
_NOT_INSTALLED = shutil.ignore_patterns("tests.json", "__pycache__")


def bundled_skill_dir() -> Path:
    """Return the directory of the skill bundled with this package."""
    return Path(__file__).parent / SKILL_NAME


def install_skill(root: Path, *, force: bool = False) -> tuple[Path, bool]:
    """Copy the bundled skill into ``<root>/.claude/skills/``.

    An existing copy is kept unless ``force`` is set, so a user's edits are never silently clobbered.

    Args:
        root: Directory whose ``.claude/skills/`` receives the skill (the user's project).
        force: Replace an existing copy.

    Returns:
        The installed skill directory, and whether it was written on this call.
    """
    dest = root / ".claude" / "skills" / SKILL_NAME
    if dest.exists():
        if not force:
            return dest, False
        shutil.rmtree(dest)
    shutil.copytree(bundled_skill_dir(), dest, ignore=_NOT_INSTALLED)
    return dest, True
