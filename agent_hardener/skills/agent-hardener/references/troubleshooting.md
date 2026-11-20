<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Start from the command's complete log: every command ends with `→ log: <path>`
(`.agent-hardener/logs/<command>-<UTC time>.log`, or `AGENT_HARDENER_LOG_FILE` when set), and that file holds
every record, tracebacks included. Or rerun with `--debug` to see the same records on the console as they
happen. A run's build and infra output is also in `.agent-hardener/run-logs/<run id>/agent-hardener.log`, and
OpenShell's own output is under `.agent-hardener/openshell-logs/`. If the caller set `AGENT_HARDENER_ERROR_FILE`,
a failed command also writes its classified error there.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `openshell status` not `Connected` | Gateway service not installed or not running | Install OpenShell with the native installer (below), then `agent-hardener setup`. |
| Every sandbox fails with `ControlSupervisorStartFailed` (macOS) | Gateway config lacks `grpc_endpoint` under Docker Desktop | `agent-hardener setup` writes `~/.config/openshell/gateway.toml`; restart the gateway service. |
| `INFERENCE_API_KEY is not set` | No key in the env file or shell | Add it to `.env`, pass `--env-file .env`. |
| Model check fails before the build | Wrong `AGENT_HARDENER_MODEL`, URL, or key | Pick a model from the list the error prints. |
| `<KEY> is set in the environment and differs from .env` | An exported value shadows the env file | `unset <KEY>` to use the file, or change the export. |
| Attack model unchanged after setting `GARAK_*` | `garak-scan.yaml` exists and wins | Edit the model fields in `garak-scan.yaml`. |
| "victim not instrumented" | Relay never started in the agent | Start it with `nemo_relay.plugin.activate(PluginConfig())` at startup, held open — see relay-connection.md. |
| Run fails saying Relay saw no tool calls | Tool calls bypass Relay | Route tools through Relay (middleware, or `create_tool_node` for a hand-built LangGraph graph). |
| No guardrails written, or the agent keeps going after a refusal (hand-built LangGraph) | Model calls bypass Relay | Send the model node through `nemo_relay.llm.execute` — see relay-connection.md. |
| Agent never starts in the sandbox | `start_command` relies on `PATH`, or the image lacks the `sandbox` user | Use full paths; follow the image contract. |
| `mkdir: cannot create directory '/etc/nemo-relay'` | Directory not created in the image | `mkdir -p /etc/nemo-relay` and `chown` it to `sandbox` in the Dockerfile. |
| Tool calls fail with connection errors | Host not allowed by the sandbox | Add it to `egress`, or a `backends` entry for local services. |
| `--dockerfile requires at least one --binary` | `init` needs both | Pass `--binary '<glob>'`. |
| New API key ignored | The sandbox kept the old provider | Run without `--reuse` so the sandbox is rebuilt. |
| Warning about image digest drift | `--victim-image` built from another manifest or version | Re-run `agent-hardener stage`, rebuild, push. |
| No garak / attacker fails to start | `setup` never ran | `agent-hardener setup` (`--force` to rebuild garak's venv). |

## Installing OpenShell

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | OPENSHELL_VERSION=v0.1.4 sh
openshell status
```

Upgrading from 0.0.x: delete every 0.0.x sandbox, uninstall the old CLI and service, remove the old state
directory (`~/.local/state/openshell`), then install as above and run `agent-hardener setup`. A 0.0.x gateway
database crashes the 0.1 gateway at startup.
