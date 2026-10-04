# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a sandboxed victim writes its telemetry.

The obligations this module used to impose on the victim now install themselves — see
:mod:`agent_hardener.relay_plugin.autopatch` for the registration, request scope and refusal recovery,
and the uploaded ``plugins.toml`` for the ATOF sink. What remains is the path the two sides agree
on: the victim writes here, and Agent Hardener copies the file out — the sandbox shares no filesystem
with the host.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set inside the victim by Agent Hardener. The default matches the manifest default so a victim started
#: by hand still lands where a run would look.
ARTIFACTS_ENV = "AGENT_HARDENER_RELAY_ARTIFACTS"
DEFAULT_ARTIFACTS_DIR = "artifacts/relay"

#: Where a *sandboxed* victim writes instead. Under /home/sandbox because every OpenShell policy
#: template already grants that read-write, so telemetry needs no filesystem exception of its own —
#: unlike the guardrail file, which must live in /etc for Relay to treat it as system policy.
SANDBOX_ARTIFACTS_DIR = "/home/sandbox/.agent-hardener/relay"
ATOF_FILENAME = "events.atof.jsonl"


def artifacts_dir() -> Path:
    """Where this victim writes its ATOF stream."""
    return Path(os.environ.get(ARTIFACTS_ENV) or DEFAULT_ARTIFACTS_DIR)
