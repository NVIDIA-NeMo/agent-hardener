# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A provider with no backend, for tests of anything that consumes traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.traces.provider import TraceProvider

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_hardener.models.traces import AgentTrace


class FakeTraceProvider(TraceProvider):
    """Returns a fixed list, and counts calls so tests can assert a path never read traces.

    ``batches`` lets a test model traffic arriving between calls: call ``n`` returns the
    concatenation of the first ``n`` batches (the last batch repeats once exhausted).
    """

    def __init__(self, traces: Iterable[AgentTrace] = (), *, batches: list[list[AgentTrace]] | None = None) -> None:
        self._batches = batches if batches is not None else [list(traces)]
        self.calls = 0

    def get_traces(self) -> list[AgentTrace]:
        self.calls += 1
        upto = min(self.calls, len(self._batches))
        return [trace for batch in self._batches[:upto] for trace in batch]
