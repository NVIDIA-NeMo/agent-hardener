# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering context passed through every display call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DisplayContext:
    """Controls what the display layer renders.

    ``verbose=True`` enables per-agent detail sections; ``False`` shows only
    headline counts. Future fields (color, width, etc.) belong here.
    """

    verbose: bool = False
