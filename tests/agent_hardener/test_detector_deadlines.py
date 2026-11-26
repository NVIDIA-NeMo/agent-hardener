# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real pipes, rather than readiness-only mocks."""

import concurrent.futures
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_hardener.agents.validators.garak_attack_replay import PersistentGarakWorker


@pytest.mark.parametrize("mode", ["partial", "silent", "no_read"])
def test_worker_deadline_disposes_and_restarts(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    worker = PersistentGarakWorker(repo_path=None, timeout_seconds=0.15)
    processes: list[subprocess.Popen[str]] = []

    def ensure() -> subprocess.Popen[str]:
        if worker._proc is not None:
            return worker._proc
        script = "import sys,time; "
        if mode == "partial" and not processes:
            script += "sys.stdin.readline();sys.stdout.write('{');sys.stdout.flush();time.sleep(30)"
        elif mode in {"silent", "no_read"} and not processes:
            script += "time.sleep(30)"
        else:
            script += 'sys.stdin.readline();print(\'{"ok":true,"is_success":false,"confidence":1,"reasoning":"fresh"}\',flush=True);time.sleep(30)'
        worker._proc = subprocess.Popen(  # noqa: S603 - fixed local test script
            [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        processes.append(worker._proc)
        return worker._proc

    monkeypatch.setattr(worker, "_ensure", ensure)
    try:
        with pytest.raises(TimeoutError):
            worker.verify("direct_prompt_injection", {}, {"payload": "x" * (1_000_000 if mode == "no_read" else 1)})
        assert processes[0].poll() is not None
        assert worker.verify("direct_prompt_injection", {}, {}) == (False, 1, "fresh")
    finally:
        worker.close()
    assert all(proc.poll() is not None for proc in processes)


def test_queued_detector_cannot_start_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = PersistentGarakWorker(repo_path=None, timeout_seconds=1)
    calls: list[Any] = []
    monkeypatch.setattr(worker, "_ensure", lambda: calls.append("unexpected"))
    worker._lock.acquire()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(worker.verify, "direct_prompt_injection", {}, {})
        time.sleep(0.02)
        worker.close()
        with pytest.raises(TimeoutError):
            pending.result(timeout=0.3)
    worker._lock.release()
    assert calls == []


@pytest.mark.parametrize(
    "response", ["not-json\n", "{}\n{}\n", "x" * (1024 * 1024 + 1)], ids=["malformed", "multiple", "oversized"]
)
def test_invalid_response_disposes_worker(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    worker = PersistentGarakWorker(repo_path=None, timeout_seconds=1)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time;sys.stdin.readline();sys.stdout.write(sys.stdin.read());sys.stdout.flush();time.sleep(30)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Inject bytes at the read boundary, keeping actual process teardown under test.
    worker._proc = proc
    monkeypatch.setattr(worker, "_ensure", lambda: proc)
    if response.startswith("x"):
        monkeypatch.setattr(
            "agent_hardener.agents.validators.garak_attack_replay.select.select", lambda r, w, *_a: (r, w, [])
        )
        monkeypatch.setattr("agent_hardener.agents.validators.garak_attack_replay.os.read", lambda *_a: b"x" * 65536)
    else:
        monkeypatch.setattr(worker, "_read_line", lambda *_a: response)
    try:
        with pytest.raises(ValueError):
            worker.verify("direct_prompt_injection", {}, {})
        assert proc.poll() is not None
        assert worker._proc is None
    finally:
        worker.close()


@pytest.mark.parametrize("error", [ProcessLookupError(), subprocess.TimeoutExpired("worker", 1)])
def test_teardown_does_not_override_report(error: Exception) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    worker = PersistentGarakWorker(repo_path=None, timeout_seconds=1)
    worker._proc = SimpleNamespace(poll=lambda: None, kill=fail, wait=fail, stdin=None, stdout=None)
    worker.close()
    assert worker._proc is None
