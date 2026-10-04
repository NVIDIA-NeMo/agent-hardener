# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``extraction.benign.predict_benign_tuples``: structural extraction over benign request text,

and its ``ExtractionCache`` integration (dedupes repeated calls with identical ``benign_requests``
across attacks within one ``DefendersManager.run()`` call).
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.extraction_cache import ExtractionCache
from agent_hardener.agents.defenders.openshell_defender_v2.config import DefenderConfig
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.benign import predict_benign_tuples
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.prompts import BenignTuplesOutput
from agent_hardener.agents.defenders.openshell_defender_v2.models import RequestTuple

pytestmark = pytest.mark.unit

_MODULE = "agent_hardener.agents.defenders.openshell_defender_v2.extraction.benign"


def test_extracts_tuples_per_request(monkeypatch) -> None:
    def fake_complete_batch(prompts, _schema, _config):
        return [
            BenignTuplesOutput(tuples=[RequestTuple(host="api.github.com", port=443, protocol="https")])
            for _ in prompts
        ]

    monkeypatch.setattr(f"{_MODULE}.complete_batch", fake_complete_batch)
    predictions = predict_benign_tuples(["list issues"], DefenderConfig())

    assert len(predictions) == 1
    assert predictions[0].source_request == "list issues"
    assert predictions[0].tuples[0].host == "api.github.com"


def test_cache_dedupes_llm_calls_across_predict_calls(monkeypatch) -> None:
    calls = []

    def fake_complete_batch(prompts, _schema, _config):
        calls.append(len(prompts))
        return [BenignTuplesOutput(tuples=[]) for _ in prompts]

    monkeypatch.setattr(f"{_MODULE}.complete_batch", fake_complete_batch)
    cache = ExtractionCache()
    benign_requests = ["list issues", "get pull request"]

    first = predict_benign_tuples(benign_requests, DefenderConfig(), cache=cache)
    second = predict_benign_tuples(list(benign_requests), DefenderConfig(), cache=cache)

    assert len(calls) == 1
    assert first == second


def test_no_cache_recomputes_every_call(monkeypatch) -> None:
    calls = []

    def fake_complete_batch(prompts, _schema, _config):
        calls.append(len(prompts))
        return [BenignTuplesOutput(tuples=[]) for _ in prompts]

    monkeypatch.setattr(f"{_MODULE}.complete_batch", fake_complete_batch)
    benign_requests = ["list issues"]

    predict_benign_tuples(benign_requests, DefenderConfig())
    predict_benign_tuples(benign_requests, DefenderConfig())

    assert len(calls) == 2
