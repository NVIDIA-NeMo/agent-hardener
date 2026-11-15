# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A salted record of the credential values an OpenShell provider was last given.

OpenShell never returns a provider's secret values, so whether the keys it holds still match the
configured ones is only knowable from a record made when they were written. The record keeps one salted
digest per credential and never a value, so it is safe on disk and names exactly which keys changed.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _digest(salt: str, name: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{name}\0{value}".encode()).hexdigest()


@dataclass(frozen=True)
class CredentialRecord:
    """The credentials a provider holds, as a salted digest per name."""

    salt: str
    digests: dict[str, str]

    @classmethod
    def of(cls, credentials: Mapping[str, str]) -> CredentialRecord:
        """Record ``credentials`` under a fresh salt."""
        salt = secrets.token_hex(16)
        return cls(salt=salt, digests={name: _digest(salt, name, value) for name, value in credentials.items()})

    def changed(self, credentials: Mapping[str, str]) -> list[str]:
        """Names added, removed, or holding a different value than when recorded, sorted."""
        names = set(self.digests) | set(credentials)
        return sorted(
            name
            for name in names
            if name not in credentials or self.digests.get(name) != _digest(self.salt, name, credentials[name])
        )

    @property
    def revision(self) -> str:
        """Identifies this one write of the provider; a sandbox records it to know which keys it was given."""
        return hashlib.sha256(json.dumps([self.salt, sorted(self.digests.items())]).encode()).hexdigest()


def read_credential_record(path: Path) -> CredentialRecord | None:
    """The record at ``path``, or None when there is none or it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CredentialRecord(salt=str(data["salt"]), digests={str(k): str(v) for k, v in data["digests"].items()})
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_credential_record(path: Path, record: CredentialRecord) -> None:
    """Write ``record`` to ``path``, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"salt": record.salt, "digests": record.digests}, indent=2) + "\n", encoding="utf-8")


def read_revision(path: Path) -> str | None:
    """The provider revision a sandbox was created with, or None when unknown."""
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_revision(path: Path, revision: str | None) -> None:
    """Record (or, with None, forget) the provider revision a sandbox was created with."""
    if revision is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(revision + "\n", encoding="utf-8")
