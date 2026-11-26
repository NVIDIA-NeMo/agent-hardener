<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Hardener room

NeMo Helix (with Studio), Agent Hardener, garak and the OpenShell gateway in **one container**, for
demos, a first look and testing. The host needs only Docker.

The room is not a deployment. It runs privileged (or with the host's Docker socket), and Studio has no
login. For real use, install the released projects: Helix's Helm chart, or the native install in the
plugin's README.

## Build

```bash
docker/build.sh                                     # main of both → nemo-agent-hardener-room:dev
docker/build.sh --platform linux/arm64              # → ...:dev-arm64 (QEMU)
docker/build.sh --agent-hardener . --helix-ref <branch, tag or commit> [--helix-repo <fork URL>]
docker/build.sh --agent-hardener . --helix <nemo-helix checkout>   # local checkouts, as they are
```

By default both are fetched at main: `NVIDIA-NeMo/agent-hardener` and `NVIDIA-NeMo/nemo-helix`. CI
builds this checkout with Helix main (or the `HELIX_REF` repository variable). The build needs
`docker buildx`, git and about 60 GB of disk; the image is about 26 GB.

Each image records what it was built from: both repositories with ref, commit and commit date, whether a
local checkout had uncommitted changes, when and by whom (`AH_ROOM_BUILT_BY`, else git's `user.email`),
and the installed versions of garak, torch, the plugin, OpenShell, Docker and Python. Read it with
`./ah-room version` (`/usr/local/share/ah-room/versions.json`) or `docker inspect` (labels).

| Path | What it is |
|---|---|
| `room/` | The base: agent-hardener, garak, OpenShell and Docker Engine (`entrypoint.sh` starts them) |
| `helix-room/` | NeMo Helix and Studio on top, plus `ah-room` (the launcher) and `ah-cycle` (the one-command cycle) |
| `build.sh` | Builds both, and Studio's static files, from agent-hardener and Helix (main, a ref, or a local checkout) |
| `helix-room/versions.py` | Writes the image's build record |
| `smoke.sh` | The CI check: start, key handling, data kept across `down`/`up`, `down --purge` |

## Run

```bash
docker run --rm --entrypoint cat nemo-agent-hardener-room:dev /usr/local/share/ah-room/ah-room > ah-room
chmod +x ah-room
printf 'OPENROUTER_API_KEY=%s\n' "<your-key>" > keys.env && chmod 600 keys.env   # or INFERENCE_API_KEY=nvapi-...
./ah-room up --keys keys.env --agent ./my-agent        # waits until ready, opens Studio on 127.0.0.1
./ah-room cycle --project-dir /agent --name my-agent --harness langgraph   # or: --example langgraph
./ah-room down                                         # keeps the data; down --purge deletes it
./ah-room version                                      # what the image was built from, when, by whom
```

- **Modes.** The default runs its own Docker Engine (privileged). `--host-docker` uses the host's
  Docker instead (Linux). `--host-openshell` also uses the host's existing OpenShell gateway.
- **Keys.** The key file is copied to `~/.agent-hardener/.env` (mode 600) and stored as the NeMo
  Secret `agent-hardener-inference-key`. No process in the room gets it in its environment.
- **Studio** listens on `127.0.0.1` only. From another machine: `ssh -L 8080:localhost:8080 <host>`.
- **Data.** Helix's data, `/work` and the inner Docker's images are in the volumes `<name>-data`,
  `<name>-work` and `<name>-docker`; the next `up` reuses them.
- **Host.** Docker Engine 20.10 or later, 4 CPUs and 8 GB of memory, about 30 GB of disk.
