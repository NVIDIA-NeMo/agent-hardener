# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the guardrail kind known to Relay, without touching the victim's code.

The contract asked of a user is one line — attach ``NemoRelayMiddleware``. The one thing that
cannot be config is registration: ``nemo_relay.plugin.initialize()`` matches each discovered
``[[components]]`` entry against the kinds registered *so far*, and raises on one it cannot resolve
(``not found: plugin component '...' is not registered``). So a kind registered too late does not
leave a quietly unguarded victim — it stops the victim from starting at all, once the war-game has
written its first guardrail into the uploaded config.

So this runs from a ``sitecustomize`` on the victim's ``PYTHONPATH``, which Python imports before
any user module.

**Nothing here patches Relay.** Everything else the war-game needs comes from Relay's own surface:
the ATOF sink travels in the ``plugins.toml`` Agent Hardener already uploads, and blocking is Relay's
``register_tool_execution_intercept``, which *returns* its refusal rather than raising — so no
recovery code is needed anywhere either.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent_hardener.guardrail")


def install() -> bool:
    """Register the guardrail kind; return whether it is now registered.

    Never raises. A victim that fails to start is worse than one that starts uninstrumented,
    because the relay preflight reports the second clearly and the first looks like a broken image.
    """
    try:
        import nemo_relay  # noqa: PLC0415

        from .config import PLUGIN_KIND  # noqa: PLC0415
        from .plugin import register  # noqa: PLC0415

        if PLUGIN_KIND not in nemo_relay.plugin.list_kinds():
            register()
    except Exception:
        logger.warning("agent-hardener could not register its guardrail kind", exc_info=True)
        return False
    return True
