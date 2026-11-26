#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#   versions.py build --room DIR --agent-hardener DIR --ah-repo URL --ah-ref REF \
#       --helix DIR --helix-repo URL --helix-ref REF --platform P --image IMG
#       (build.sh, on the host) print the build record as JSON. An empty --*-repo/--*-ref means a local
#       checkout: its origin and branch are recorded, and uncommitted changes mark it dirty.
#   versions.py labels RECORD
#       (build.sh) the image labels for RECORD, one NAME=VALUE per line
#   versions.py image RECORD
#       (the image build) print versions.json: RECORD plus the versions installed in the image
"""What a room image was built from, when, and by whom."""

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

LABEL = "com.nvidia.ah-room"
AH_PYTHON = "/opt/agent-hardener/src/.venv/bin/python"
GARAK_PYTHON = "/root/.agent-hardener/garak-venv/bin/python"
HELIX_PYTHON = "/opt/nemo-helix/.venv/bin/python"
SOURCES = ("agent_hardener", "nemo_helix", "room")


def run(*cmd: str, cwd: Path | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    assert out, f"{' '.join(cmd)} printed nothing"
    return out


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


def public_url(url: str) -> str:
    """The repository URL without any user or token in it."""
    return re.sub(r"^(\w+://)[^/@]+@", r"\1", url)


def source(path: Path, repo: str, ref: str) -> dict[str, str | bool]:
    if not (path / ".git").exists():
        raise SystemExit(f"versions: {path} is not a git checkout; the image must record what it was built from")
    commit = git(path, "rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"{path}: unexpected commit {commit!r}"
    if not repo:
        repo = git(path, "remote", "get-url", "origin")
        ref = git(path, "rev-parse", "--abbrev-ref", "HEAD")
    assert repo and ref, f"{path}: no repository or ref"
    return {
        "repository": public_url(repo),
        "ref": ref,
        "commit": commit,
        "committed": git(path, "log", "-1", "--format=%cI"),
        "dirty": bool(git(path, "status", "--porcelain", "--untracked-files=no")),
    }


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def actor(repo: Path) -> str:
    """Who is building or publishing: AH_ROOM_BUILT_BY, the GitHub Actions run, else REPO's git user.email."""
    if os.environ.get("AH_ROOM_BUILT_BY"):
        return os.environ["AH_ROOM_BUILT_BY"]
    if os.environ.get("GITHUB_ACTIONS") == "true":
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        return f"{os.environ['GITHUB_ACTOR']} (GitHub Actions {run_url})"
    email = subprocess.run(
        ["git", "-C", str(repo), "config", "user.email"], check=False, capture_output=True, text=True
    ).stdout.strip()
    return email or f"{getpass.getuser()}@{socket.gethostname()}"


def cmd_build(args: argparse.Namespace) -> None:
    ah = Path(args.agent_hardener)
    version = tomllib.loads((ah / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert version, f"no project.version in {ah}/pyproject.toml"
    record = {
        "image": args.image,
        "platform": args.platform,
        "built": now(),
        "built_by": actor(Path(args.room)),
        "agent_hardener": {"version": version, **source(ah, args.ah_repo, args.ah_ref)},
        "nemo_helix": source(Path(args.helix), args.helix_repo, args.helix_ref),
        "room": source(Path(args.room), "", ""),
    }
    print(json.dumps(record, separators=(",", ":")))


def load(text: str) -> dict:
    record = json.loads(text)
    missing = [k for k in ("image", "platform", "built", "built_by", *SOURCES) if k not in record]
    assert not missing, f"build record lacks {missing}"
    return record


def cmd_labels(args: argparse.Namespace) -> None:
    r = load(args.record)
    ah, helix, room = r["agent_hardener"], r["nemo_helix"], r["room"]
    labels = {
        "org.opencontainers.image.title": "nemo-agent-hardener-room",
        "org.opencontainers.image.created": r["built"],
        "org.opencontainers.image.version": ah["version"],
        "org.opencontainers.image.source": ah["repository"],
        "org.opencontainers.image.revision": ah["commit"],
        f"{LABEL}.built-by": r["built_by"],
        f"{LABEL}.agent-hardener.ref": ah["ref"],
        f"{LABEL}.agent-hardener.dirty": str(ah["dirty"]).lower(),
        f"{LABEL}.helix.source": helix["repository"],
        f"{LABEL}.helix.ref": helix["ref"],
        f"{LABEL}.helix.revision": helix["commit"],
        f"{LABEL}.helix.dirty": str(helix["dirty"]).lower(),
        f"{LABEL}.room.source": room["repository"],
        f"{LABEL}.room.revision": room["commit"],
        f"{LABEL}.room.dirty": str(room["dirty"]).lower(),
    }
    for name, value in labels.items():
        assert "\n" not in value, f"label {name} has a newline"
        print(f"{name}={value}")


def package_version(python: str, *packages: str) -> list[str]:
    code = "import importlib.metadata as m, sys; print(*(m.version(p) for p in sys.argv[1:]))"
    return run(python, "-c", code, *packages).split()


def cmd_image(args: argparse.Namespace) -> None:
    r = load(args.record)
    (installed,) = package_version(AH_PYTHON, "nvidia-agent-hardener")
    assert installed == r["agent_hardener"]["version"], (
        f"agent-hardener {installed} is installed, the record says {r['agent_hardener']['version']}"
    )
    garak, torch = package_version(GARAK_PYTHON, "garak", "torch")
    (plugin,) = package_version(HELIX_PYTHON, "nemo-agent-hardener-plugin")
    r["components"] = {
        "garak": garak,
        "torch": torch,
        "nemo_agent_hardener_plugin": plugin,
        "openshell": run("openshell", "--version").split()[-1],
        "openshell_gateway": run("openshell-gateway", "--version").split()[-1],
        "docker_engine": run("dockerd", "--version").split()[2].rstrip(","),
        "python_agent_hardener": run(AH_PYTHON, "-c", "import platform; print(platform.python_version())"),
        "python_helix": run(HELIX_PYTHON, "-c", "import platform; print(platform.python_version())"),
    }
    print(json.dumps(r, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    for flag in ("--room", "--agent-hardener", "--helix", "--platform", "--image"):
        build.add_argument(flag, required=True)
    for flag in ("--ah-repo", "--ah-ref", "--helix-repo", "--helix-ref"):
        build.add_argument(flag, default="")
    build.set_defaults(func=cmd_build)
    for name, func in (("labels", cmd_labels), ("image", cmd_image)):
        p = sub.add_parser(name)
        p.add_argument("record")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
