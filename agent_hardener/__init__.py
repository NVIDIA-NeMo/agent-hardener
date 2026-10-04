# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# __init__.py
"""Project: agent_hardener."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# The distribution is `nvidia-agent-hardener`; only the import package is `agent_hardener`. Looking
# up the old name here fails silently — __version__ degrades to the dev sentinel and every user sees
# a spurious "not installed" warning.
_DISTRIBUTION = "nvidia-agent-hardener"

try:
    __version__ = version(_DISTRIBUTION)
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

    warnings.warn(
        f"{_DISTRIBUTION} is not installed. Run 'just install' to set up the project.",
        stacklevel=2,
    )
