# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that each skill's ``tests.json`` prompts make a live Claude Code load the right skill.

Developer skills are routed inside the repo, where ``.claude/skills`` holds them. The customer skill is
routed the way customers get it: in an empty project after ``install_skill``, with no developer skills.
Each prompt runs headless with file-changing tools disabled; the first ``Skill`` call is what counts.
This calls a model and costs tokens, so it runs on demand (``just skill-routing``), not in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_hardener.skills import bundled_skill_dir, install_skill

REPO = Path(__file__).resolve().parents[1]


def _first_skill(claude: str, model: str, prompt: str, cwd: Path) -> str | None:
    """Return the name of the first skill Claude Code loads for ``prompt``, or None."""
    cmd = [claude, "-p", prompt, "--model", model, "--max-turns", "3", "--output-format", "stream-json"]
    cmd += ["--verbose", "--disallowedTools", "Edit,Write,Bash,NotebookEdit"]
    try:
        # An explicit arg list; the prompts come from this repo's own tests.json files.
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=240, check=False)  # noqa: S603
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    for line in stdout.splitlines():
        try:
            content = (json.loads(line).get("message") or {}).get("content") or []
        except (json.JSONDecodeError, AttributeError):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Skill":
                return block["input"].get("skill")
    return None


def main() -> int:
    """Route every case, print the misses, and exit non-zero if any case failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", default="claude", help="Claude Code binary.")
    parser.add_argument("--model", default="claude-haiku-5-5", help="Model to route with.")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per prompt; routing is not deterministic.")
    parser.add_argument("--out", type=Path, help="Write every result as JSON here.")
    args = parser.parse_args()

    customer_project = Path(tempfile.mkdtemp(prefix="agent-hardener-customer-"))
    install_skill(customer_project)
    jobs = []
    for tests in [*sorted((REPO / ".agents" / "skills").glob("*/tests.json")), bundled_skill_dir() / "tests.json"]:
        cwd = customer_project if tests.parent == bundled_skill_dir() else REPO
        jobs += [(tests.parent.name, case, cwd) for case in json.loads(tests.read_text())["tests"]] * args.repeats

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(lambda job: _first_skill(args.claude, args.model, job[1]["prompt"], job[2]), jobs))

    results = []
    for (owner, case, _), skill in zip(jobs, loaded, strict=True):
        ok = skill == case["expected_skill"] if "expected_skill" in case else skill != case["expected_skill_not"]
        results.append({"skill": owner, **case, "loaded": skill, "ok": ok})
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    passed = sum(r["ok"] for r in results)
    print(f"{passed}/{len(results)} routed correctly ({args.model}, {args.repeats} run(s) per prompt)")
    for r in results:
        if not r["ok"]:
            want = r.get("expected_skill") or f"not {r['expected_skill_not']}"
            print(f"  MISS {r['skill']} [{r['type']}] loaded={r['loaded']} want={want}: {r['prompt']}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
