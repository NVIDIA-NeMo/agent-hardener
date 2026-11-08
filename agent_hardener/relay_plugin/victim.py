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

#: Where a *sandboxed* victim writes instead. Under /tmp because every OpenShell policy template grants it
#: read-write and every compute driver lets the sandbox write there; /home/sandbox is not writable on the
#: Kubernetes driver, whose sandbox home is /sandbox. Unlike the guardrail file, which must live in /etc for
#: Relay to treat it as system policy, telemetry needs no filesystem exception of its own.
SANDBOX_ARTIFACTS_DIR = "/tmp/agent-hardener/relay"
ATOF_FILENAME = "events.atof.jsonl"


def artifacts_dir() -> Path:
    """Where this victim writes its ATOF stream."""
    return Path(os.environ.get(ARTIFACTS_ENV) or DEFAULT_ARTIFACTS_DIR)
