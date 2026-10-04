# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pre-run model connectivity preflight."""

from __future__ import annotations

import httpx
import pytest

from agent_hardener import model_check as mc
from agent_hardener.errors import ModelUnavailableError


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


def test_preflight_noop_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "k")  # credential present, so the presence gate passes
    monkeypatch.delenv("AGENT_HARDENER_MODEL", raising=False)
    monkeypatch.delenv("GARAK_RED_TEAM_MODEL_NAME", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(mc, "validate_choice", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1))
    mc.preflight_configured_models()
    assert called["n"] == 0  # defaults are known-good; nothing probed


def test_preflight_raises_for_configured_attack_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "k")
    monkeypatch.delenv("AGENT_HARDENER_MODEL", raising=False)
    monkeypatch.setenv("GARAK_RED_TEAM_MODEL_NAME", "typo/model")
    monkeypatch.setattr(
        mc, "validate_choice", lambda *_a, **_k: mc.Validation(ok=False, reason="unknown_model", available=["real"])
    )
    with pytest.raises(ModelUnavailableError, match="real"):
        mc.preflight_configured_models()


def test_preflight_raises_when_inference_key_missing_on_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_MODEL", raising=False)
    monkeypatch.delenv("GARAK_RED_TEAM_MODEL_NAME", raising=False)
    with pytest.raises(ModelUnavailableError, match="INFERENCE_API_KEY"):
        mc.preflight_configured_models()


def test_preflight_skips_key_check_for_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # A custom endpoint may be a keyless local model — don't require the credential, don't probe defaults.
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_HARDENER_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("AGENT_HARDENER_MODEL", raising=False)
    monkeypatch.delenv("GARAK_RED_TEAM_MODEL_NAME", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(mc, "validate_choice", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1))
    mc.preflight_configured_models()  # no raise
    assert called["n"] == 0
