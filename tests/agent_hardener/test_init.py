# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import pytest

import agent_hardener


def test_init_warns_when_package_version_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_version)

    # The lookup uses the distribution name, which differs from the import package name.
    with pytest.warns(UserWarning, match="nvidia-agent-hardener is not installed"):
        reloaded = importlib.reload(agent_hardener)

    assert reloaded.__version__ == "0.0.0-dev"

    monkeypatch.undo()
    importlib.reload(sys.modules["agent_hardener"])
