# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the OpenShell sandbox identity from a user-facing agent name.

OpenShell caps a sandbox name at 19 characters while an agent name is an unconstrained public
string, so the name cannot be prefixed verbatim. The mapping keeps a readable head of the agent
name and appends a digest of the whole name: deterministic, so a rerun reuses and cleans up the
same sandbox, and collision-resistant, so two long names sharing a head stay isolated.
"""

from __future__ import annotations

import hashlib
import re

MAX_SANDBOX_NAME_LEN = 19

_PREFIX = "ah-"
_DIGEST_LEN = 7
_SLUG_LEN = MAX_SANDBOX_NAME_LEN - len(_PREFIX) - _DIGEST_LEN - 1
_UNSAFE = re.compile(r"[^a-z0-9]+")


def sandbox_name(agent_name: str) -> str:
    """The OpenShell sandbox name for *agent_name*, always within :data:`MAX_SANDBOX_NAME_LEN`."""
    slug = _UNSAFE.sub("-", agent_name.lower()).strip("-")[:_SLUG_LEN].rstrip("-")
    digest = hashlib.sha256(agent_name.encode("utf-8")).hexdigest()[:_DIGEST_LEN]
    return f"{_PREFIX}{slug}-{digest}" if slug else f"{_PREFIX}{digest}"
