# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pre-run model connectivity preflight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from agent_hardener import model_check as mc
from agent_hardener.errors import ModelUnavailableError

if TYPE_CHECKING:
    from conftest import ModelProbe


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)


def test_validate_lists_available_and_flags_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]}))
    v = mc.validate_choice("typo", "https://x/v1", "k")
    assert not v.ok
    assert v.reason == "unknown_model"
    assert v.available == ["a", "b"]
    assert mc.validate_choice("a", "https://x/v1", "k").ok


def test_validate_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.Response(401))
    v = mc.validate_choice("m", "https://x/v1", "bad")
    assert not v.ok
    assert v.reason == "auth"


def test_validate_no_model_list_soft_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.Response(404))
    assert mc.validate_choice("m", "https://x/v1", "k").ok


_REAL_PROBE_GENERATION = mc.probe_generation


@pytest.fixture(autouse=True)
def _real_probe_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared conftest stubs the probe out for every other test; these tests are about it."""
    monkeypatch.setattr(mc, "probe_generation", _REAL_PROBE_GENERATION)


def test_a_default_run_proves_its_key_with_a_one_token_request(model_probes: list[ModelProbe]) -> None:
    """Listing /models passed a bad key, which then failed 10s into the attack."""
    mc.preflight_configured_models()

    assert model_probes == [(mc.default_model(), mc.default_base_url(), "run-key")]


def test_uses_sharing_a_model_endpoint_and_key_are_asked_once_under_both_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(mc, "probe_generation", lambda *_a, **_k: mc.Validation(ok=False, reason="auth"))
    monkeypatch.setattr(mc, "raise_if_unavailable", lambda label, *_a, **_k: labels.append(label))

    mc.require_answers(
        [mc.ModelUse("analysis", "m", "https://x/v1/", "k", "A"), mc.ModelUse("judge", "m", "https://x/v1", "k", "B")]
    )

    assert labels == ["analysis and judge"]


def test_a_rejected_key_names_the_variable_that_holds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc, "probe_generation", lambda *_a, **_k: mc.Validation(ok=False, reason="auth", detail="401"))

    with pytest.raises(ModelUnavailableError, match="judge") as exc:
        mc.require_answers([mc.ModelUse("judge", "m", "https://x/v1", "bad", "AGENT_HARDENER_SAFETY_API_KEY")])
    assert "AGENT_HARDENER_SAFETY_API_KEY" in exc.value.remediation


def test_preflight_raises_when_inference_key_missing_on_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_MODEL", raising=False)
    monkeypatch.delenv("GARAK_RED_TEAM_MODEL_NAME", raising=False)
    with pytest.raises(ModelUnavailableError, match="INFERENCE_API_KEY"):
        mc.preflight_configured_models()


def test_a_keyless_custom_endpoint_is_probed_without_a_key(
    monkeypatch: pytest.MonkeyPatch, model_probes: list[ModelProbe]
) -> None:
    """A custom endpoint may be a local model that needs no credential; the probe says whether it does."""
    monkeypatch.delenv("INFERENCE_API_KEY")
    monkeypatch.setenv("AGENT_HARDENER_BASE_URL", "http://localhost:8000/v1")

    mc.preflight_configured_models()

    assert model_probes == [(mc.default_model(), "http://localhost:8000/v1", None)]


def test_probe_generation_sends_one_token_with_the_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        sent.update(url=url, **kwargs)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "post", fake_post)
    verdict = mc.probe_generation("m", "https://x/v1/", "k", timeout=5, extra_body={"chat_template_kwargs": {}})
    assert verdict.ok
    assert sent["url"] == "https://x/v1/chat/completions"
    assert sent["json"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "chat_template_kwargs": {},
    }
    assert sent["timeout"] == 5


def test_probe_generation_reports_a_hung_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*_a: object, **_k: object) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "post", hang)
    verdict = mc.probe_generation("m", "https://x/v1", "k", timeout=600)
    assert verdict.reason == "timeout"
    with pytest.raises(ModelUnavailableError, match="600s"):
        mc.raise_if_unavailable("red-team", "m", "https://x/v1", verdict)


def test_probe_generation_reports_a_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.Response(404))
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: httpx.Response(404, text="model not found"))
    verdict = mc.probe_generation("m", "https://x/v1", "k", timeout=5)
    assert verdict.reason == "error"
    assert "model not found" in verdict.detail


def _patch_posts(monkeypatch: pytest.MonkeyPatch, *responses: httpx.Response) -> list[float]:
    """Answer successive probe requests with *responses*; return the sleeps between them."""
    queue = list(responses)
    sleeps: list[float] = []
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: queue.pop(0))
    monkeypatch.setattr(mc.time, "sleep", sleeps.append)
    return sleeps


def test_probe_generation_rides_out_a_brief_overload(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _patch_posts(monkeypatch, httpx.Response(503), httpx.Response(429), httpx.Response(200, json={}))
    assert mc.probe_generation("m", "https://x/v1", "k", timeout=5).ok
    assert sleeps == [5.0, 15.0]


def test_probe_generation_fails_a_backend_that_stays_down(monkeypatch: pytest.MonkeyPatch) -> None:
    down = httpx.Response(503, headers={"retry-after": "600"}, json={"error": {"message": "backend unavailable"}})
    sleeps = _patch_posts(monkeypatch, down, down, down)
    verdict = mc.probe_generation("m", "https://x/v1", "k", timeout=5)
    assert verdict.reason == "error"
    assert verdict.detail == "HTTP 503 after 3 attempts: backend unavailable"
    assert sleeps == [60.0, 60.0]  # Retry-After honoured, but capped


_DENIED = {"error": {"message": "The requested model 'm' is not available for this API key", "code": "403"}}


def test_probe_generation_reports_an_unlisted_model_as_unknown_not_as_a_bad_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get(monkeypatch, httpx.Response(200, json={"data": [{"id": "real"}]}))
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: httpx.Response(403, json=_DENIED))
    verdict = mc.probe_generation("m", "https://x/v1", "k", timeout=5)
    assert verdict.reason == "unknown_model"
    with pytest.raises(ModelUnavailableError, match="Reachable models: real"):
        mc.raise_if_unavailable("red-team", "m", "https://x/v1", verdict)


def test_probe_generation_keeps_the_server_message_for_a_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.Response(200, json={"data": [{"id": "m"}]}))
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: httpx.Response(403, json=_DENIED))
    verdict = mc.probe_generation("m", "https://x/v1", "k", timeout=5)
    assert verdict.reason == "auth"
    assert verdict.detail == f"HTTP 403: {_DENIED['error']['message']}"
