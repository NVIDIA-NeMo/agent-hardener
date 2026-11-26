# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recon: the victim profile (purpose and tools) the garak attackers are handed before they attack.

Run once per run, before the first attack stage, and shared by every attacker through the run context
(:data:`~agent_hardener.models.RECON_PROFILE_KEY`). Without it each probe finds the victim's tools itself:
AgentBreaker asks the victim, and a probe that cannot has nothing to attack.

A source is a :class:`Recon` in :data:`RECON`. ``static`` reads a profile written ahead of time; a source that
asks the victim (as AgentBreaker's own discovery does) registers here under its own name.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml

from agent_hardener.garak_venv import garak_user_data_dir
from agent_hardener.models import AgentProfile, ReconSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The run-log file the profile a run attacked with is kept in.
RECON_FILENAME = "recon.yaml"
#: garak's user data subdirectory agent-hardener stages profiles in, one fresh directory per scan.
GARAK_STAGING_DIR = "agent_hardener"
#: The file name garak's agent probes give their profile (``agent_breaker/agent.yaml``).
GARAK_PROFILE_NAME = "agent.yaml"


class Recon(Protocol):
    """A source of the victim profile."""

    async def profile(self, settings: ReconSettings, *, target_url: str) -> AgentProfile:
        """Return the profile of the victim served at ``target_url``."""
        ...


class StaticRecon:
    """The profile written ahead of time: the settings' ``file`` or inline ``profile``."""

    async def profile(self, settings: ReconSettings, *, target_url: str) -> AgentProfile:
        """Load the profile; a profile with no tools is refused, since probes would then rediscover them."""
        if settings.file is not None:
            raw = yaml.safe_load(settings.file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"recon file {settings.file} must hold a mapping with agent_purpose and tools")
            profile = AgentProfile.model_validate(raw)
        else:
            assert settings.profile is not None, "static recon needs a file or a profile"
            profile = settings.profile
        if not profile.tools:
            raise ValueError(
                f"static recon profile {settings.file or '(inline)'} lists no tools; list the victim's tools or "
                "set recon source to none so the probe discovers them"
            )
        return profile


RECON: dict[str, Recon] = {"static": StaticRecon()}


async def run_recon(settings: ReconSettings, *, target_url: str) -> AgentProfile:
    """The victim profile from the configured source, which must not be ``none``."""
    assert settings.source != "none", "recon source 'none' has no profile to produce"
    return await RECON[settings.source].profile(settings, target_url=target_url)


def write_profile(profile: AgentProfile, path: Path) -> Path:
    """Write ``profile`` as garak's ``agent.yaml`` shape to ``path``; return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@contextlib.contextmanager
def staged_for_garak(profile: AgentProfile, env: dict[str, str]) -> Iterator[str]:
    """Write ``profile`` where a garak process run with ``env`` can open it; yield the name garak takes.

    garak opens a probe's ``agent_config_file`` only relative to its data directories, so the profile is
    written to a fresh directory under the user one and removed when the block exits.
    """
    data_dir = garak_user_data_dir(env)
    staging = data_dir / GARAK_STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    scan_dir = Path(tempfile.mkdtemp(dir=staging))
    try:
        path = write_profile(profile, scan_dir / GARAK_PROFILE_NAME)
        yield path.relative_to(data_dir).as_posix()
    finally:
        shutil.rmtree(scan_dir)
