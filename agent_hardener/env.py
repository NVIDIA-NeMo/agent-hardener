# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central catalog of the environment variables Agent Hardener reads.

One module names every well-known environment variable and exposes a typed accessor for it,
so the keys are not duplicated as bare string literals across the codebase and there is a
single place to see what the process reads from the environment.

Accessors read ``os.environ`` *live* (not a captured snapshot) on purpose: the CLI populates
the process environment from a dotenv file at startup, before any agent runs, so a snapshot
taken at import time would miss those values.
"""

from __future__ import annotations

import os

# --- variable names (single source of truth) ---------------------------------------------

# Credential shared by every LLM call against NVIDIA's inference endpoint.
INFERENCE_API_KEY = "INFERENCE_API_KEY"  # pragma: allowlist secret
# The credential name garak's subprocess expects; Agent Hardener mirrors INFERENCE_API_KEY into it.
NIM_API_KEY = "NIM_API_KEY"  # pragma: allowlist secret
# API-key vars garak's generators require to be *present* to even start (it refuses to boot when any
# is unset, even when the generator never uses them). We default the unused ones so garak starts on
# any platform instead of depending on the ambient environment carrying them.
GARAK_REQUIRED_API_KEYS = (
    NIM_API_KEY,
    "OPENAI_API_KEY",  # pragma: allowlist secret
    "REST_API_KEY",  # pragma: allowlist secret
    "OPENAICOMPATIBLE_API_KEY",  # pragma: allowlist secret
)
# Extra environment variable names to forward into the garak subprocess, comma-separated. The
# subprocess env is an allow-list (see :func:`agent_hardener.garak_venv.garak_subprocess_env`); this
# is the escape hatch for hosts that need something unusual (e.g. ``HF_TOKEN`` for a gated model).
GARAK_ENV_PASSTHROUGH = "AGENT_HARDENER_GARAK_ENV_PASSTHROUGH"
# Overrides the ``garak`` invocation (e.g. ``uv run garak``) for the agent_breaker attacker.
GARAK_COMMAND = "GARAK_COMMAND"
# Path to a local garak checkout used by the attack-replay validator.
GARAK_REPO_PATH = "GARAK_REPO_PATH"
# Optional GitHub token for the smart-benign github_analyzer clone step.
GH_TOKEN = "GH_TOKEN"  # noqa: S105 - env var name, not a secret  # pragma: allowlist secret
# Opt-in flag for Langfuse tracing in the smart-benign validator.
LANGFUSE_ENABLED = "LANGFUSE_ENABLED"


# --- typed accessors ----------------------------------------------------------------------


def inference_api_key() -> str | None:
    """Return the inference API credential, or ``None`` when unset."""
    return os.environ.get(INFERENCE_API_KEY)


def garak_command() -> str | None:
    """Return the configured garak command override, or ``None`` when unset."""
    return os.environ.get(GARAK_COMMAND)


def garak_repo_path() -> str | None:
    """Return the local garak checkout path, or ``None`` when unset."""
    return os.environ.get(GARAK_REPO_PATH)


def github_token() -> str | None:
    """Return the GitHub token for repo cloning, or ``None`` when unset."""
    return os.environ.get(GH_TOKEN) or None


def langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is switched on."""
    return bool(os.environ.get(LANGFUSE_ENABLED, "").strip())
