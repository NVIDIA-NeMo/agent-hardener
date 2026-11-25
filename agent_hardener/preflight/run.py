# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single pre-run gate ``agent-hardener run`` calls before it builds the sandbox."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.preflight.host import preflight_host
from agent_hardener.preflight.models import preflight_run_models

if TYPE_CHECKING:
    from agent_hardener.models import SessionConfig


def preflight_run(session: SessionConfig, *, reuse: bool = False) -> None:
    """Fail on the first unusable host tool, then on the first model that won't answer.

    Host checks go first: they are local and fast, while the model checks send real requests.
    """
    preflight_host(session, reuse=reuse)
    preflight_run_models(session)
