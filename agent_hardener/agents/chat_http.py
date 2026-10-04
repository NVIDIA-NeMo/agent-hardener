# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response-body decoding shared by every victim caller.

Request shaping and text extraction moved to :class:`~agent_hardener.endpoint.EndpointContract`, which the
attacker, the victim probe and validator replay all resolve from one config — this module is only the
transport-level "did the victim even return JSON?" step that sits in front of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


def response_body_or_text(response: httpx.Response) -> Any:
    """Return the parsed JSON body, or the raw text when the response isn't JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text
