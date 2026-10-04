# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NeMo Relay plugin that enforces Agent Hardener's guardrails inside the victim.

The guardrails defender writes configuration (:mod:`agent_hardener.relay_plugin.config`); this package
reads it at runtime, in the victim's own process, and refuses a tool call whose request a safety
judge scores over the guardrail's threshold.

**The victim writes no Agent Hardener code.** Its only obligation is to attach Relay's own
``NemoRelayMiddleware``; :mod:`agent_hardener.relay_plugin.autopatch` installs the rest from a
``sitecustomize`` Agent Hardener stages into the image, and the ATOF sink travels in the uploaded
``plugins.toml``.

Layering is deliberate: :mod:`agent_hardener.relay_plugin.policy` holds the decision logic and imports
nothing from Relay, so it can be unit-tested without a Relay runtime.
"""

from .config import PLUGIN_KIND
from .plugin import register

__all__ = ["PLUGIN_KIND", "register"]
