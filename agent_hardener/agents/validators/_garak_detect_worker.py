# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived worker that runs garak detectors' ``verify()`` in the dedicated garak venv.

garak is intentionally isolated in its own venv (it pins torch/litellm that conflict with
agent-hardener), so the attack-replay validator cannot import garak in-process. This script is executed
by the *garak-venv* interpreter (see :mod:`agent_hardener.garak_venv`) as a plain file, so it must import
ONLY the standard library and garak — never ``agent_hardener`` (which is not installed in that venv).

Protocol: a newline-delimited JSON request/response stream. The process imports garak once (on the
first request) and then services many requests over its lifetime — one cold torch/garak import per
*run*, not per hit. Read one JSON request object per input line, write one JSON response line:

    request  = {"attack_type": str, "config_root": dict, "verify_kwargs": dict, "repo_path": str|null}
    response = {"ok": true, "is_success": bool, "confidence": float, "reasoning": str}
             | {"ok": false, "error": str}

garak/torch chatter is redirected to stderr for the whole process so stdout carries only JSON lines.
Detector instances are cached by (attack_type, serialized config_root) so repeated configs reuse the
already-constructed detector.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

# attack_type -> (garak detector module, detector class). Mirrors the direct/indirect split the
# validator drives; the class contract is ``verify(**kwargs) -> (is_success, confidence, reasoning)``.
_DETECTORS = {
    "direct_prompt_injection": ("garak.detectors.agent_breaker", "AgentBreakerResult"),
    "indirect_prompt_injection": ("garak.detectors.indirect_injection", "IndirectInjectionResult"),
}

_detector_cache: dict[tuple[str, str], Any] = {}
_seen_repo_paths: set[str] = set()


def _detector_for(attack_type: str, config_root: dict[str, Any], repo_path: str | None) -> Any:
    if repo_path and repo_path not in _seen_repo_paths:
        # Honour a dev editable garak checkout (GARAK_REPO_PATH) even inside the worker.
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        _seen_repo_paths.add(repo_path)
    cache_key = (attack_type, json.dumps(config_root, sort_keys=True))
    detector = _detector_cache.get(cache_key)
    if detector is None:
        module_name, class_name = _DETECTORS[attack_type]
        module = importlib.import_module(module_name)
        detector = getattr(module, class_name)(config_root=config_root)
        _detector_cache[cache_key] = detector
    return detector


def _run(request: dict[str, Any]) -> dict[str, Any]:
    detector = _detector_for(request["attack_type"], request["config_root"], request.get("repo_path"))
    is_success, confidence, reasoning = detector.verify(**request["verify_kwargs"])
    return {
        "ok": True,
        "is_success": bool(is_success),
        "confidence": float(confidence),
        "reasoning": str(reasoning),
    }


def main() -> int:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # keep garak/torch import + logging noise off the JSON result channel

    def _reply(response: dict[str, Any]) -> None:
        real_stdout.write(json.dumps(response) + "\n")
        real_stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            response = _run(json.loads(line))
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _reply(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
