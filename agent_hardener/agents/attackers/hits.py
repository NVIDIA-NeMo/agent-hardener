# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared normalization for attacker hit records (live garak scan and preloaded replay)."""

from __future__ import annotations

from typing import Any


def normalize_hit_record(raw: Any, index: int, source: str | None) -> dict[str, Any]:
    """Coerce a raw hit into a dict tagged with its source and index.

    A non-dict hit is wrapped as ``{"value": raw}``. ``source`` (when given) and ``hit_index`` are set
    without overwriting values the record already carries.
    """
    record = dict(raw) if isinstance(raw, dict) else {"value": raw}
    if source:
        record.setdefault("source", source)
    record.setdefault("hit_index", index)
    return record
