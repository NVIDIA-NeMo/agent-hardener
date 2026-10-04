# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural (never benign-judging) tuple extraction over ``DefenderInput.benign_requests``.

Every string here is already ground-truth benign — these functions only predict *what network
request it would cause*, never *whether it's benign*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import load_config
from ..llm.client import complete_batch
from ..models import BenignPrediction
from .prompts import BENIGN_TUPLE_EXTRACTION_PROMPT, BenignTuplesOutput
from .request_tuple import canonicalize

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...extraction_cache import ExtractionCache
    from ..config import DefenderConfig

_BATCH_SIZE = 8


def _batched(requests: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(requests), size):
        yield requests[i : i + size]


def _predict_benign_tuples_uncached(benign_requests: list[str], cfg: DefenderConfig) -> list[BenignPrediction]:
    predictions: list[BenignPrediction] = []
    for chunk in _batched(benign_requests, _BATCH_SIZE):
        prompts = [BENIGN_TUPLE_EXTRACTION_PROMPT.format(payload=payload) for payload in chunk]
        outputs = complete_batch(prompts, BenignTuplesOutput, cfg)
        for payload, output in zip(chunk, outputs, strict=True):
            predictions.append(
                BenignPrediction(
                    source_request=payload,
                    tuples=[canonicalize(t) for t in output.tuples],
                    confidence=1.0 if output.tuples else 0.0,
                )
            )
    return predictions


def predict_benign_tuples(
    benign_requests: list[str],
    config: DefenderConfig | None = None,
    cache: ExtractionCache | None = None,
) -> list[BenignPrediction]:
    """Extract the network endpoints each benign request text explicitly references.

    ``benign_requests`` is the same list for every attack within one ``DefendersManager.run()``
    call — when ``cache`` is given (see ``DefenderInput.context["defender_extraction_cache"]``),
    this result is computed once per run and reused rather than re-derived per attack.
    """
    cfg = config or load_config()
    if cache is None:
        return _predict_benign_tuples_uncached(benign_requests, cfg)
    key = ("benign", tuple(benign_requests))
    return cache.get_or_compute(key, lambda: _predict_benign_tuples_uncached(benign_requests, cfg))


def coverage_for_endpoint(predictions: list[BenignPrediction], host: str | None, port: int | None) -> float:
    """Count of predicted benign requests that touch ``(host, port)`` — feeds ``Candidate.confidence``.

    A candidate narrowing an endpoint that only two predicted-benign requests ever touched is
    weak evidence; ``nodes.base.confidence_for`` compares this against ``config.min_corpus_coverage``.
    """
    count = 0.0
    for prediction in predictions:
        if any(t.host == host and t.port == port for t in prediction.tuples):
            count += 1.0
    return count
