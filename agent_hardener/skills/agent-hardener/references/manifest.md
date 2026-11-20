<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `agent-hardener.yaml` fields

`agent-hardener init` writes this file. Unknown keys are rejected.

## `agent:` (required)

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Agent name; names the sandbox and the benign-suite folder. |
| `project_dir` | yes | The agent's project, relative to the manifest. |
| `dockerfile` | yes | Dockerfile within the project. Agent Hardener hardens the image you ship. |
| `start_command` | yes | Launch command inside the sandbox, with full binary paths. |
| `binaries` | yes | Globs of processes allowed to egress, e.g. `/usr/local/bin/python*`. |
| `harness` | no | `langgraph`, `langchain`, `deepagents`, `hermes` or `other`. |
| `relay_integration_confirmed` | no | `true` once the agent is Relay-connected. Recorded; the run still verifies it. |
| `port` | no | The agent's HTTP port (default 8000). |
| `secrets` | no | Env-var names taken from `secrets_file` and passed to the agent. |
| `secrets_file` | no | Dotenv file holding them (default `.env`). |
| `env` | no | Extra plain env vars for the agent. |
| `egress` | no | External `host` or `host:port` the agent may reach (port 443 if omitted). |
| `discover_egress` | no | Scan the agent code for endpoints (default `true`). |
| `relay_artifacts` | no | Host directory the agent's trace file is copied to (default `artifacts/relay`; keep it relative). |
| `guardrails_tool` | no | A hint naming the tool to guard. Usually leave unset: the defender guards whichever tool each attack reached. |

## `backends:` (optional list)

Services the agent's tools call.

| Field | Meaning |
| --- | --- |
| `name` | Backend name. |
| `compose_file` | A compose file Agent Hardener starts on this host. |
| `health_url` | Polled until healthy before the run. |
| `ports` | Ports to open from the sandbox. |
| `host` | A service elsewhere: only the route is opened (bare hostname or IP, no port). |
| `allowed_ips` | Private ranges OpenShell would otherwise block, e.g. a cluster CIDR. |

`host` and `compose_file` are mutually exclusive; `host` or `allowed_ips` need `ports`.

## Optional blocks

- `garak:` — tune the attacker (models, attempts, victim port) instead of editing `garak-scan.yaml`.
- `defenders:`, `traces:` — defender and trace settings.
- `overrides:` — anything else in the full session config.
