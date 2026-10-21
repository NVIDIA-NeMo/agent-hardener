# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port-forward management and readiness polling in the OpenShell lifecycle mixins."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import error

import pytest

from agent_hardener.openshell import _forward, _polling, support
from agent_hardener.openshell.config import OpenShellCommandResult, OpenShellConfig, SubprocessCommandRunner
from agent_hardener.openshell.lifecycle import OpenShellLifecycle

if TYPE_CHECKING:
    from collections.abc import Callable

_FORWARD = "0.0.0.0:8000"
_ARGS = ["forward", "start", "--gateway", "gw", _FORWARD, "sb"]


class ScriptedRunner:
    """Answers each command with ``respond(args)`` -> (returncode, stdout) and records them."""

    def __init__(self, respond: Callable[[list[str]], tuple[int, str]] = lambda _args: (0, "")) -> None:
        self.respond = respond
        self.commands: list[list[str]] = []

    def run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        returncode, stdout = self.respond(args)
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)


class FakeProcess:
    """Popen stand-in: ``exit_code`` None means it is still running after the grace period."""

    pid = 4242

    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code

    def wait(self, timeout: float) -> int:
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(cmd="openshell", timeout=timeout)
        return self.exit_code


def _lifecycle(tmp_path: Path, runner: Any = None, **overrides: Any) -> OpenShellLifecycle:
    config = OpenShellConfig(
        gateway="gw", sandbox="sb", policy_path=tmp_path / "policy.yaml", forward=_FORWARD, **overrides
    )
    return OpenShellLifecycle(config, runner=runner or ScriptedRunner(), log_dir=tmp_path / "logs")


def _pid_file(lifecycle: OpenShellLifecycle) -> Path:
    return lifecycle._managed_forward_pid_path()


class _FakeClock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Polling runs on fake time: timeouts elapse instantly and independently of the real clock."""
    fake = _FakeClock()
    monkeypatch.setattr(support, "time", fake)
    return fake


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace Popen with a still-running fake; returns the recorded launches."""
    launches: list[dict[str, Any]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        launches.append({"command": command, **kwargs})
        return FakeProcess(exit_code=None)

    monkeypatch.setattr(_forward.subprocess, "Popen", fake_popen)
    return launches


# --- start_forward -----------------------------------------------------------------------------------------


def test_start_forward_without_a_configured_forward_is_a_noop(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    lifecycle = _lifecycle(tmp_path, runner)
    lifecycle.config = OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "p.yaml")

    assert lifecycle.start_forward() == OpenShellCommandResult(ok=True, command=[], output="")
    assert runner.commands == []


def test_background_forward_failure_is_returned_without_polling(tmp_path: Path) -> None:
    runner = ScriptedRunner(lambda _args: (1, "gateway unreachable"))
    lifecycle = _lifecycle(tmp_path, runner)

    result = lifecycle.start_forward()

    assert (result.ok, result.output) == (False, "gateway unreachable")
    assert len(runner.commands) == 1
    assert "--background" in runner.commands[0]


def test_background_forward_waits_until_the_forward_is_listed(tmp_path: Path) -> None:
    def respond(args: list[str]) -> tuple[int, str]:
        return (0, f"{_FORWARD} sb running") if args[1:3] == ["forward", "list"] else (0, "started")

    lifecycle = _lifecycle(tmp_path, ScriptedRunner(respond))

    result = lifecycle.start_forward()

    assert result.ok
    assert f"forward ready: {_FORWARD} -> sb" in result.output


def test_managed_forward_launches_a_detached_process_and_records_its_pid(
    tmp_path: Path, popen: list[dict[str, Any]]
) -> None:
    lifecycle = _lifecycle(tmp_path, SubprocessCommandRunner())

    result = lifecycle.start_forward()

    assert result.ok
    assert result.command == ["openshell", *_ARGS]
    assert popen[0]["start_new_session"] is True
    assert popen[0]["stderr"] == subprocess.STDOUT
    assert _pid_file(lifecycle).read_text() == "4242\n"
    assert "managed forward started pid=4242" in result.output
    log_path = Path(result.output.split("log=")[1])
    assert log_path.read_text(encoding="utf-8").startswith("command: openshell forward start")


@pytest.mark.parametrize("health_ok", [True, False])
def test_managed_forward_is_only_as_healthy_as_the_victim_health_check(
    tmp_path: Path, popen: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, health_ok: bool
) -> None:
    lifecycle = _lifecycle(tmp_path, SubprocessCommandRunner(), health_url="http://127.0.0.1:8000/health")
    health = OpenShellCommandResult(ok=health_ok, command=["wait-for-health"], output="health says so")
    monkeypatch.setattr(lifecycle, "wait_for_health", lambda: health)

    result = lifecycle.start_forward()

    assert result.ok is health_ok
    assert "managed forward started" in result.output
    assert "health says so" in result.output


def test_managed_forward_launch_oserror_is_reported_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("openshell: not found")

    monkeypatch.setattr(_forward.subprocess, "Popen", failing_popen)
    lifecycle = _lifecycle(tmp_path, SubprocessCommandRunner())

    result = lifecycle.start_forward()

    assert (result.ok, result.output) == (False, "openshell: not found")


def test_managed_forward_surfaces_a_failure_to_stop_the_previous_one(
    tmp_path: Path, popen: list[dict[str, Any]]
) -> None:
    lifecycle = _lifecycle(tmp_path, SubprocessCommandRunner())
    _pid_file(lifecycle).mkdir()  # reading a directory as the pid file raises OSError

    result = lifecycle.start_forward()

    assert result.ok is False
    assert "failed to read" in result.output
    assert popen == []


def test_restarting_a_managed_forward_reports_the_one_it_replaced(
    tmp_path: Path, popen: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_forward.os, "killpg", lambda _pid, _sig: None)
    lifecycle = _lifecycle(tmp_path, SubprocessCommandRunner())
    _pid_file(lifecycle).write_text("999\n")

    result = lifecycle.start_forward()

    assert result.output.splitlines()[0] == "stopped managed forward process 999"
    assert "managed forward started pid=4242" in result.output.splitlines()[1]


# --- stop_managed_forward ----------------------------------------------------------------------------------


def test_stop_without_a_pid_file_is_a_clean_noop(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path).stop_managed_forward()

    assert (result.ok, result.output) == (True, "")


def test_stop_discards_an_invalid_pid_file(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    _pid_file(lifecycle).write_text("not-a-pid")

    result = lifecycle.stop_managed_forward()

    assert result.ok
    assert "removed invalid forward pid file" in result.output
    assert not _pid_file(lifecycle).exists()


def test_stop_signals_the_whole_process_group_and_clears_the_pid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(_forward.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    lifecycle = _lifecycle(tmp_path)
    _pid_file(lifecycle).write_text("321\n")

    result = lifecycle.stop_managed_forward()

    assert sent == [(321, signal.SIGTERM)]
    assert (result.ok, result.output) == (True, "stopped managed forward process 321")
    assert not _pid_file(lifecycle).exists()


def test_stop_treats_an_already_dead_process_as_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(_forward.os, "killpg", gone)
    lifecycle = _lifecycle(tmp_path)
    _pid_file(lifecycle).write_text("321")

    result = lifecycle.stop_managed_forward()

    assert (result.ok, result.output) == (True, "managed forward process 321 was not running")
    assert not _pid_file(lifecycle).exists()


def test_stop_keeps_the_pid_file_when_the_process_cannot_be_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(_pid: int, _sig: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(_forward.os, "killpg", denied)
    lifecycle = _lifecycle(tmp_path)
    _pid_file(lifecycle).write_text("321")

    result = lifecycle.stop_managed_forward()

    assert result.ok is False
    assert "failed to stop managed forward 321" in result.output
    assert _pid_file(lifecycle).exists()


# --- wait_for_forward_active / _forward_is_active ----------------------------------------------------------


def test_wait_for_forward_active_without_a_forward_is_a_noop(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    lifecycle.config = OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "p.yaml")

    assert lifecycle.wait_for_forward_active().ok


@pytest.mark.parametrize(
    ("forward", "listing", "expected"),
    [
        (_FORWARD, f"{_FORWARD}  sb  running", True),
        (_FORWARD, "0.0.0.0 | sb | 8000", True),
        (_FORWARD, "0.0.0.0 | other-sandbox | 8000", False),
        (_FORWARD, "127.0.0.1 | sb | 9000", False),
        (_FORWARD, "No active forwards.", False),
        ("8000", "8000 sb", True),
        ("8000", "sb only", False),
        (":8000", "sb 8000", False),
        ("0.0.0.0:", "sb 0.0.0.0", False),
    ],
    ids=[
        "exact-match",
        "bind-and-port-in-columns",
        "wrong-sandbox",
        "wrong-port",
        "empty-listing",
        "bare-port-exact",
        "bare-port-missing",
        "missing-bind",
        "missing-port",
    ],
)
def test_forward_is_active_matches_on_forward_and_sandbox(
    tmp_path: Path, forward: str, listing: str, expected: bool
) -> None:
    lifecycle = _lifecycle(tmp_path)
    lifecycle.config = OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "p.yaml", forward=forward)

    assert lifecycle._forward_is_active(listing) is expected


def test_forward_is_active_is_false_without_a_configured_forward(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    lifecycle.config = OpenShellConfig(gateway="gw", sandbox="sb", policy_path=tmp_path / "p.yaml")

    assert lifecycle._forward_is_active("sb 0.0.0.0:8000") is False


def test_pid_file_name_is_filesystem_safe_and_lives_under_the_log_dir(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)

    path = _pid_file(lifecycle)

    assert path.parent == tmp_path / "logs"
    assert path.name == "sb-0.0.0.0_8000.pid"


def test_pid_file_defaults_to_the_cwd_when_no_log_dir_is_injected(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, cwd=tmp_path)
    lifecycle.log_dir = None

    assert _pid_file(lifecycle).parent == tmp_path / ".agent-hardener" / "openshell-forwards"


# --- sandbox readiness -------------------------------------------------------------------------------------


def test_sandbox_ready_when_status_reports_ready(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, ScriptedRunner(lambda _args: (0, "Phase: Ready")))

    result = lifecycle.wait_for_sandbox_ready()

    assert (result.ok, result.output) == (True, "sandbox ready: sb")


def test_sandbox_ready_polls_through_provisioning(tmp_path: Path) -> None:
    phases = iter(["Phase: Provisioning", "Phase: Provisioning", "Phase: Ready"])
    runner = ScriptedRunner(lambda _args: (0, next(phases)))

    result = _lifecycle(tmp_path, runner).wait_for_sandbox_ready()

    assert result.ok
    assert len(runner.commands) == 3


def test_sandbox_ready_times_out_with_the_last_status_output(tmp_path: Path, clock: _FakeClock) -> None:
    lifecycle = _lifecycle(tmp_path, ScriptedRunner(lambda _args: (0, "Phase: Provisioning")), sandbox_timeout=30)

    result = lifecycle.wait_for_sandbox_ready()

    assert result.ok is False
    assert result.output == "timed out waiting for sandbox sb to become ready\nPhase: Provisioning"
    assert result.command == ["wait-for-sandbox", "sb"]
    assert clock.now >= 30


def test_a_failing_status_command_never_counts_as_ready(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, ScriptedRunner(lambda _args: (1, "Ready")), sandbox_timeout=30)

    assert lifecycle.wait_for_sandbox_ready().ok is False


# --- victim health -----------------------------------------------------------------------------------------


def test_wait_for_health_without_a_url_is_a_noop(tmp_path: Path) -> None:
    assert _lifecycle(tmp_path).wait_for_health() == OpenShellCommandResult(ok=True, command=[], output="")


def test_wait_for_health_reports_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, float]] = []
    monkeypatch.setattr(_polling, "wait_for_http_health", lambda url, timeout: seen.append((url, timeout)))
    lifecycle = _lifecycle(tmp_path, health_url="http://127.0.0.1:8000/health", health_timeout=7)

    result = lifecycle.wait_for_health()

    assert (result.ok, result.output) == (True, "health ready: http://127.0.0.1:8000/health")
    assert seen == [("http://127.0.0.1:8000/health", 7)]


def test_wait_for_health_failure_without_diagnostics_keeps_only_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_url: str, _timeout: float) -> None:
        raise TimeoutError("never healthy")

    monkeypatch.setattr(_polling, "wait_for_http_health", fail)
    lifecycle = _lifecycle(tmp_path, health_url="http://127.0.0.1:8000/health")
    monkeypatch.setattr(
        lifecycle, "collect_victim_diagnostics", lambda: OpenShellCommandResult(ok=True, command=[], output="")
    )

    result = lifecycle.wait_for_health()

    assert (result.ok, result.output) == (False, "never healthy")
    assert result.command == ["wait-for-health", "http://127.0.0.1:8000/health"]


# --- wait_for_http_health ----------------------------------------------------------------------------------


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _urlopen(monkeypatch: pytest.MonkeyPatch, outcomes: list[int | Exception]) -> list[str]:
    urls: list[str] = []
    queue = list(outcomes)

    def fake(url: str, timeout: float) -> _Response:
        urls.append(url)
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(_polling.request, "urlopen", fake)
    return urls


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/health", "127.0.0.1:8000/health"])
def test_http_health_rejects_non_http_schemes_without_connecting(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    urls = _urlopen(monkeypatch, [200])

    with pytest.raises(ValueError, match="unsupported health URL scheme"):
        _polling.wait_for_http_health(url, 1)

    assert urls == []


@pytest.mark.parametrize("status", [200, 204, 299])
def test_http_health_accepts_any_2xx(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    urls = _urlopen(monkeypatch, [status])

    _polling.wait_for_http_health("https://victim/health", 1)

    assert urls == ["https://victim/health"]


def test_http_health_retries_through_connection_errors_and_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _urlopen(monkeypatch, [ConnectionRefusedError(), error.URLError("reset"), 503, 200])

    _polling.wait_for_http_health("http://victim/health", 30)

    assert len(urls) == 4


@pytest.mark.parametrize("outcome", [503, 301, ConnectionRefusedError()], ids=["5xx", "3xx", "refused"])
def test_http_health_times_out_when_never_healthy(monkeypatch: pytest.MonkeyPatch, outcome: int | Exception) -> None:
    _urlopen(monkeypatch, [outcome])

    with pytest.raises(TimeoutError, match="timed out waiting for victim health at http://victim/health"):
        _polling.wait_for_http_health("http://victim/health", 30)
