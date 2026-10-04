# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the victim endpoint contract.

The contract exists so the attacker, the victim probe and validator replay send the *same* request
and read the answer the *same* way. If they drift, an attack and its replay stop being comparable and
a "blocked" verdict stops meaning anything — so the shared-shape tests matter more than they look.
"""

from __future__ import annotations

import pytest

from agent_hardener.endpoint import (
    DEFAULT_VICTIM_CONCURRENCY,
    SESSION_ID_HEADER,
    EndpointContract,
    extract_json_path,
)


def _openai(**kwargs) -> EndpointContract:
    return EndpointContract(url="http://victim/v1/chat/completions", **kwargs)


def test_openai_is_the_zero_config_default() -> None:
    assert _openai().request_payload("hi") == {
        "model": "agent-hardener",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_a_victim_with_another_shape_declares_a_field_instead() -> None:
    """Requiring OpenAI-compat would make the user adapt their agent to us; this is the escape hatch."""
    contract = _openai(mode="field", input_field="query")
    assert contract.request_payload("hi") == {"query": "hi"}


def test_session_id_is_sent_so_attribution_is_ours_not_incidental() -> None:
    """A Fabric-served victim opens a fresh runtime per request when no session header is sent."""
    assert _openai().headers("attack-7")[SESSION_ID_HEADER] == "attack-7"


def test_headers_omit_the_session_when_there_is_none() -> None:
    assert SESSION_ID_HEADER not in _openai().headers()


def test_extra_headers_are_preserved_and_not_mutated() -> None:
    extra = {"authorization": "Bearer x"}
    contract = _openai(extra_headers=extra)
    contract.headers("s1")
    assert extra == {"authorization": "Bearer x"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"choices": [{"message": {"content": "answer"}}]}, "answer"),
        ({"choices": [{"text": "legacy"}]}, "legacy"),
        ({"output": "out"}, "out"),
        ({"response": "resp"}, "resp"),
        ("plain text", "plain text"),
    ],
)
def test_extract_text_handles_the_shapes_victims_actually_return(body: object, expected: str) -> None:
    assert _openai().extract_text(body) == expected


def test_unknown_shape_falls_back_to_raw_json_not_empty_string() -> None:
    """An empty string would read as a refusal and score the attack as blocked when it was not."""
    assert _openai().extract_text({"unexpected": "shape"}) == '{"unexpected": "shape"}'


def test_explicit_path_wins_over_the_default_shape() -> None:
    contract = _openai(response_json_path="$.data.answer")
    assert contract.extract_text({"data": {"answer": "via path"}, "output": "ignored"}) == "via path"


def test_path_that_does_not_match_falls_back_rather_than_failing() -> None:
    contract = _openai(response_json_path="$.missing.key")
    assert contract.extract_text({"output": "still found"}) == "still found"


def test_garak_gets_the_same_mapping_as_replay() -> None:
    """The attacker is a subprocess, so it receives the shape as config; it must be the same shape."""
    contract = _openai(mode="field", input_field="query")
    fields = contract.garak_rest_fields()
    assert fields["req_template_json_object"] == {"query": "$INPUT"}
    assert fields["response_json"] is True


def test_garak_default_response_field_matches_openai_extraction() -> None:
    assert _openai().garak_rest_fields()["response_json_field"] == "$.choices[0].message.content"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("$.a.b", "found"),
        ("a.b", "found"),
        ("$.a", {"b": "found"}),
        ("$.a.missing", None),
        ("$.a.b.c", None),
        ("$.a[0]", None),
        ("$.bad-syntax!", None),
    ],
)
def test_extract_json_path_fails_closed(path: str, expected: object) -> None:
    """A malformed or inapplicable path returns None rather than raising into the run."""
    assert extract_json_path({"a": {"b": "found"}}, path) == expected


def test_extract_json_path_indexes_lists_and_bounds_check() -> None:
    body = {"choices": [{"message": {"content": "c"}}]}
    assert extract_json_path(body, "$.choices[0].message.content") == "c"
    assert extract_json_path(body, "$.choices[5].message.content") is None


def test_default_concurrency_matches_the_victims_own_semaphore() -> None:
    """Agent Hardener should not queue behind a cap it cannot see, where the wait reads as victim latency."""
    assert DEFAULT_VICTIM_CONCURRENCY == 8
