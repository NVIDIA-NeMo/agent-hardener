# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for garak venv resolution/provisioning and the `agent-hardener setup` command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_hardener import garak_venv
from agent_hardener.cli import app

runner = CliRunner()


# --- subprocess env: explicit dict, aliasing, required-key defaults -----------------------


def _clear_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in garak_venv.GARAK_REQUIRED_API_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)


def test_garak_subprocess_env_keeps_existing_nim_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("NIM_API_KEY", "nim-key")
    monkeypatch.setenv("INFERENCE_API_KEY", "inf-key")
    env = garak_venv.garak_subprocess_env()
    assert env["NIM_API_KEY"] == "nim-key"  # not overwritten by the INFERENCE alias


def test_garak_subprocess_env_defaults_required_keys_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    env = garak_venv.garak_subprocess_env()
    assert env is not None  # explicit dict, never None
    for key in garak_venv.GARAK_REQUIRED_API_KEYS:
        assert env[key] == "NOT_SET"  # garak starts even without ambient keys


def test_garak_subprocess_env_aliases_inference_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("INFERENCE_API_KEY", "inf-key")
    env = garak_venv.garak_subprocess_env()
    assert env["NIM_API_KEY"] == "inf-key"  # mirrored from INFERENCE_API_KEY
    assert env["OPENAI_API_KEY"] == "NOT_SET"  # other required keys still defaulted


def test_garak_subprocess_env_drops_unrelated_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("INFERENCE_API_KEY", "inf-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")  # pragma: allowlist secret
    monkeypatch.setenv("GH_TOKEN", "gh-secret")  # pragma: allowlist secret
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret")  # pragma: allowlist secret
    env = garak_venv.garak_subprocess_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GH_TOKEN" not in env
    assert "LANGFUSE_SECRET_KEY" not in env
    # The inference credential reaches garak only through the NIM alias, never under its own name.
    assert "INFERENCE_API_KEY" not in env
    assert env["NIM_API_KEY"] == "inf-key"


def test_garak_subprocess_env_forwards_proxy_and_ca_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/corp.pem")
    monkeypatch.setenv("NO_PROXY", "localhost")
    env = garak_venv.garak_subprocess_env()
    assert env["HTTPS_PROXY"] == "http://proxy.example.com:3128"
    assert env["REQUESTS_CA_BUNDLE"] == "/etc/ssl/corp.pem"
    assert env["NO_PROXY"] == "localhost"


def test_garak_subprocess_env_forwards_garak_prefixed_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("GARAK_LOG_LEVEL", "debug")
    env = garak_venv.garak_subprocess_env()
    assert env["GARAK_LOG_LEVEL"] == "debug"


def test_garak_subprocess_env_passthrough_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_required_keys(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf-secret")  # pragma: allowlist secret
    monkeypatch.setenv("UNRELATED", "nope")
    assert "HF_TOKEN" not in garak_venv.garak_subprocess_env()  # dropped by default
    monkeypatch.setenv(garak_venv.GARAK_ENV_PASSTHROUGH, "HF_TOKEN, MISSING_VAR")
    env = garak_venv.garak_subprocess_env()
    assert env["HF_TOKEN"] == "hf-secret"  # opted in explicitly
    assert "MISSING_VAR" not in env  # named but unset — not invented
    assert "UNRELATED" not in env


# --- resolution ---------------------------------------------------------------------------


def test_resolve_garak_python_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(garak_venv.GARAK_PYTHON_ENVVAR, raising=False)
    resolved = garak_venv.resolve_garak_python()
    assert resolved == str((Path("~/.agent-hardener/garak-venv") / "bin" / "python").expanduser())
    assert "~" not in resolved  # expanded


def test_resolve_garak_python_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(garak_venv.GARAK_PYTHON_ENVVAR, "/custom/venv/bin/python")
    assert garak_venv.resolve_garak_python() == "/custom/venv/bin/python"


def test_resolve_garak_venv_dir_is_python_grandparent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(garak_venv.GARAK_PYTHON_ENVVAR, "/custom/venv/bin/python")
    assert garak_venv.resolve_garak_venv_dir() == Path("/custom/venv")


# --- provisioning -------------------------------------------------------------------------


def test_provision_requires_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(garak_venv.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="uv not found"):
        garak_venv.provision_garak_venv()


def test_provision_idempotent_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "garak-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv(garak_venv.GARAK_PYTHON_ENVVAR, str(python))
    monkeypatch.setattr(garak_venv.shutil, "which", lambda _: "uv")
    calls: list[list[str]] = []
    monkeypatch.setattr(garak_venv, "_run", calls.append)

    result = garak_venv.provision_garak_venv()
    assert result == python
    assert calls == []  # nothing run when already provisioned


def test_provision_creates_and_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "garak-venv" / "bin" / "python"
    monkeypatch.setenv(garak_venv.GARAK_PYTHON_ENVVAR, str(python))
    monkeypatch.setattr(garak_venv.shutil, "which", lambda _: "uv")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        calls.append(cmd)
        python.parent.mkdir(parents=True, exist_ok=True)  # simulate `uv venv` creating the tree
        python.touch()

    monkeypatch.setattr(garak_venv, "_run", fake_run)

    result = garak_venv.provision_garak_venv(force=True)
    assert result == python
    assert calls[0][:2] == ["uv", "venv"]
    assert calls[0][-1] == str(tmp_path / "garak-venv")
    assert calls[1][:3] == ["uv", "pip", "install"]
    assert garak_venv.GARAK_PINNED_SPEC in calls[1]


def test_pinned_spec_carries_the_wheel_hash() -> None:
    """A version bump that forgets the hash would silently drop artifact verification."""
    assert garak_venv.GARAK_VERSION in garak_venv.GARAK_WHEEL_URL
    assert garak_venv.GARAK_PINNED_SPEC.endswith(f"#sha256={garak_venv.GARAK_WHEEL_SHA256}")
    assert len(garak_venv.GARAK_WHEEL_SHA256) == 64


# --- `agent-hardener setup` command -----------------------------------------------------------


def test_setup_command_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_provision(*, force: bool) -> Path:
        return Path("/x/bin/python")

    monkeypatch.setattr(garak_venv, "provision_garak_venv", fake_provision)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "garak ready" in result.stdout


def test_setup_command_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*, force: bool) -> Path:
        raise RuntimeError("uv not found — install it")

    monkeypatch.setattr(garak_venv, "provision_garak_venv", boom)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 1
