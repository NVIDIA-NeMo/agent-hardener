# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-serializable execution context injected into in-process agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _noop_progress(msg: str, cur: int | None = None, tot: int | None = None) -> None:
    pass


@dataclass(slots=True)
class RunContext:
    """Carries live-feedback hooks into an in-process agent run.

    HTTP agents receive no RunContext; in-process agents may declare an optional
    ``ctx: RunContext`` parameter and the framework injects this automatically.
    """

    emit_progress: Callable[[str, int | None, int | None], None] = field(default=_noop_progress)
