# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The garak detect worker: a newline-delimited JSON request/response loop run by the garak venv's interpreter.

garak itself is not installed here, so detector modules are faked in ``sys.modules`` — the contract under test
is the worker's own protocol, caching, and failure isolation.
"""

from __future__ import annotations

import io
import json
import runpy
import sys
import types
from typing import Any, ClassVar

import pytest

from agent_hardener.agents.validators import _garak_detect_worker as worker

_DIRECT = "direct_prompt_injection"
_INDIRECT = "indirect_prompt_injection"


class _Detector:
    """Records its construction and answers ``verify`` from the kwargs it is given."""

    instances: ClassVar[list[_Detector]] = []

    def __init__(self, *, config_root: dict[str, Any]) -> None:
        self.config_root = config_root
        self.verify_calls: list[dict[str, Any]] = []
        _Detector.instances.append(self)

    def verify(self, **kwargs: Any) -> tuple[Any, Any, Any] | None:
        self.verify_calls.append(kwargs)
        if kwargs.get("unjudged"):
            return None
        if kwargs.get("explode"):
            raise ValueError("detector blew up")
        if kwargs.get("noise"):
            print("garak import chatter")
        return kwargs.get("success", True), kwargs.get("confidence", 1), kwargs.get("reasoning", "because")


_DETECTOR_CLASSES = {
    "garak.detectors.agent_breaker": "AgentBreakerResult",
    "garak.detectors.indirect_injection": "IndirectInjectionResult",
}


@pytest.fixture(autouse=True)
def fake_garak(monkeypatch: pytest.MonkeyPatch) -> None:
    _Detector.instances = []
    for module_name, class_name in _DETECTOR_CLASSES.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (_Detector,), {}))
        monkeypatch.setitem(sys.modules, module_name, module)


def _request(
    attack_type: str = _DIRECT,
    config_root: dict[str, Any] | None = None,
    repo_path: str | None = None,
    **verify: Any,
) -> dict[str, Any]:
    return {
        "attack_type": attack_type,
        "config_root": config_root if config_root is not None else {"model": "m"},
        "verify_kwargs": verify,
        "repo_path": repo_path,
    }


def _serve(monkeypatch: pytest.MonkeyPatch, *lines: str) -> tuple[int, list[dict[str, Any]]]:
    """Run the worker file as a script, as the garak venv's interpreter does, with a fresh detector cache."""
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    monkeypatch.setattr(sys, "stdout", stdout)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(worker.__file__, run_name="__main__")
    return int(exit_info.value.code or 0), [json.loads(line) for line in stdout.getvalue().splitlines()]


def _serve_requests(monkeypatch: pytest.MonkeyPatch, *requests: dict[str, Any]) -> list[dict[str, Any]]:
    status, replies = _serve(monkeypatch, *(json.dumps(r) for r in requests))
    assert status == 0
    return replies


# --- detector construction and caching ---------------------------------------------------------------------


def test_each_attack_type_loads_its_own_detector_class_with_the_config_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_requests(monkeypatch, _request(_DIRECT, {"k": "v"}), _request(_INDIRECT, {"k": "v"}))

    assert [(type(d).__name__, d.config_root) for d in _Detector.instances] == [
        ("AgentBreakerResult", {"k": "v"}),
        ("IndirectInjectionResult", {"k": "v"}),
    ]


def test_an_identical_config_reuses_the_detector_regardless_of_key_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_requests(monkeypatch, _request(config_root={"a": 1, "b": 2}), _request(config_root={"b": 2, "a": 1}))

    assert len(_Detector.instances) == 1
    assert len(_Detector.instances[0].verify_calls) == 2


@pytest.mark.parametrize(
    ("other_type", "other_config"),
    [(_DIRECT, {"a": 2}), (_INDIRECT, {"a": 1})],
    ids=["different-config", "different-attack-type"],
)
def test_a_different_config_or_attack_type_gets_a_new_detector(
    monkeypatch: pytest.MonkeyPatch, other_type: str, other_config: dict[str, Any]
) -> None:
    _serve_requests(monkeypatch, _request(config_root={"a": 1}), _request(other_type, other_config))

    assert len(_Detector.instances) == 2


def test_a_dev_garak_checkout_is_put_on_the_path_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", ["/site-packages"])

    _serve_requests(
        monkeypatch,
        _request(config_root={"a": 1}, repo_path="/dev/garak"),
        _request(config_root={"a": 2}, repo_path="/dev/garak"),
    )

    assert sys.path == ["/dev/garak", "/site-packages"]


def test_a_repo_path_already_on_the_path_is_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", ["/dev/garak"])

    _serve_requests(monkeypatch, _request(repo_path="/dev/garak"))

    assert sys.path == ["/dev/garak"]


# --- one request -------------------------------------------------------------------------------------------


def test_the_verdict_is_coerced_to_json_types(monkeypatch: pytest.MonkeyPatch) -> None:
    (response,) = _serve_requests(monkeypatch, _request(success=1, confidence="0.25", reasoning=42))

    assert response == {"ok": True, "is_success": True, "confidence": 0.25, "reasoning": "42"}


def test_a_detector_that_cannot_judge_is_reported_as_unjudged(monkeypatch: pytest.MonkeyPatch) -> None:
    (response,) = _serve_requests(monkeypatch, _request(unjudged=True))

    assert response == {"ok": True, "judged": False}


def test_the_verify_kwargs_are_forwarded_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_requests(monkeypatch, _request(prompt="p", output="o"))

    assert _Detector.instances[0].verify_calls == [{"prompt": "p", "output": "o"}]


# --- the stdin/stdout loop ---------------------------------------------------------------------------------


def test_one_json_reply_per_request_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = [_request(success=True, reasoning="first"), _request(success=False, reasoning="second")]

    status, replies = _serve(monkeypatch, *(json.dumps(r) for r in requests))

    assert status == 0
    assert [(r["is_success"], r["reasoning"]) for r in replies] == [(True, "first"), (False, "second")]


def test_blank_lines_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _, replies = _serve(monkeypatch, "", "   ", json.dumps(_request()), "")

    assert len(replies) == 1


@pytest.mark.parametrize(
    ("line", "error_prefix"),
    [
        ("{not json", "JSONDecodeError"),
        (json.dumps({"config_root": {}, "verify_kwargs": {}}), "KeyError"),
        (json.dumps(_request("sql_injection")), "KeyError"),
        (json.dumps(_request(explode=True)), "ValueError: detector blew up"),
    ],
    ids=["bad-json", "missing-field", "unknown-attack", "detector-error"],
)
def test_a_bad_request_gets_an_error_reply_and_the_worker_keeps_serving(
    monkeypatch: pytest.MonkeyPatch, line: str, error_prefix: str
) -> None:
    _, replies = _serve(monkeypatch, line, json.dumps(_request(reasoning="still alive")))

    assert replies[0]["ok"] is False
    assert replies[0]["error"].startswith(error_prefix)
    assert replies[1] == {"ok": True, "is_success": True, "confidence": 1.0, "reasoning": "still alive"}


def test_library_chatter_goes_to_stderr_and_never_into_the_json_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, replies = _serve(monkeypatch, json.dumps(_request(noise=True)))

    assert len(replies) == 1  # every stdout line parsed as JSON
    assert "garak import chatter" in capsys.readouterr().err
