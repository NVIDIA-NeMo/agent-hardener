# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize a mounted keys file into agent-hardener's operator dotenv (mode 600), printing no value.

    python3 keys.py SOURCE DESTINATION

SOURCE has OPENROUTER_API_KEY=... or INFERENCE_API_KEY=... and any secrets the user's agent declares;
DESTINATION gets plain NAME=value lines, with the inference key as INFERENCE_API_KEY.
"""

import sys
from pathlib import Path


def read_keys(source: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.removeprefix("export ").strip()] = value.strip().strip('"').strip("'")
    key = values.get("INFERENCE_API_KEY") or values.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(f"ah-room: {source} has neither INFERENCE_API_KEY nor OPENROUTER_API_KEY")
    # Every other line is a secret the user's agent declares (OPENAI_API_KEY, a backend token, ...): the
    # war-game, the CLI and Studio jobs all read this one file, so one mounted file carries them all.
    values["INFERENCE_API_KEY"] = key
    bad = [name for name in values if not name.replace("_", "").isalnum() or name[0].isdigit()]
    if bad:
        raise SystemExit(f"ah-room: {source} has invalid variable names: {', '.join(bad)}")
    return values


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: keys.py SOURCE DESTINATION")
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    values = read_keys(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.touch(mode=0o600)
    destination.chmod(0o600)
    destination.write_text("".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
