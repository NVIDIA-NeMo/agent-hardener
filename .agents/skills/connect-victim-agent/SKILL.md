---
name: connect-victim-agent
description: Run a LangGraph, LangChain, Hermes or custom-image agent as the victim in the OpenShell sandbox — manifest, image requirements, secrets, egress, backends. Use when "my agent won't start in the sandbox" or "how do I war-game my own agent". Not for wiring Relay into agent code — use `nemo-relay`.
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connect a victim agent

Agent Hardener hardens the agent the user ships: it builds *their* Dockerfile into an OpenShell
sandbox, starts it with *their* command, and calls it over HTTP. This skill gets that image built,
started and answering. Making its tool calls visible to Relay is the `nemo-relay` skill.

## Scope

- **In scope:** the `agent-hardener.yaml` manifest, `agent-hardener init` / `agent-hardener inspect`,
  the image requirements, start command, port, secrets, egress, host backends, `harness`, and
  bringing the sandbox up with `agent-hardener up`.
- **Out of scope:** Relay middleware and starting Relay (`nemo_relay.plugin.activate()`) in the agent's code
  (`nemo-relay`); gateway, provider and policy operations (`openshell-ops`); choosing models
  (`models-and-providers`); a run that starts but fails later (`run-and-debug`).

Supported victims are anything with a Dockerfile that serves HTTP. The manifest's `harness`
records which kind it is: `langgraph`, `langchain`, `deepagents`, `hermes` or `other`.

## Key files

| What | Where |
|---|---|
| Manifest schema (`AgentSpec`, `BackendManifest`) and its expansion into a full session | `agent_hardener/manifest.py` |
| `init` command and the prompts' help text (the image requirements are in the `dockerfile` entry) | `agent_hardener/cli/init.py`, `agent_hardener/cli_prompts.py` |
| How the build context is staged and what gets appended to the user's Dockerfile | `agent_hardener/openshell/relay_victim.py` |
| Starting sandbox policies (permissive is the war-game default) | `agent_hardener/templates/relay-victim/` |
| How the victim is called, and non-OpenAI request/response shapes | `agent_hardener/agents/victims/openshell_victim.py`, `agent_hardener/endpoint.py` |
| Fail-fast checks on a Relay victim before a run | `agent_hardener/preflight/relay.py` |
| Worked reference: a LangGraph victim with its Dockerfile and manifest | `examples/langgraph-victim/` |

## Step 0: Preflight

1. `openshell status --gateway auto-defender` reports `Connected`. If not, see `openshell-ops`.
2. Docker builds locally (`docker info`), or you'll build elsewhere with `agent-hardener stage`
   (Step 5).
3. The agent already runs outside the sandbox, e.g. `docker run -p 8000:8000 ...` answers
   `curl localhost:8000/health`. Fix the agent first; the sandbox only adds constraints.

## Step 1: Serve the contract Agent Hardener calls

- `POST /v1/chat/completions`, OpenAI-shaped: it receives `{"model": ..., "messages": [...]}` and
  must answer with the text at `choices[0].message.content`. This is the zero-config default; see
  `examples/langgraph-victim/server.py`.
- `GET /health` returns 200 once ready. Bring-up polls it, with a 120 s timeout set by the manifest.
- Another shape works, but each component that calls the victim reads `input_field` /
  `response_json_path` from its own config, so they must all be set through `overrides:`. Read
  `agent_hardener/endpoint.py` before going that way. Prefer adding the OpenAI route.

## Step 2: Make the image sandbox-ready

The user owns the Dockerfile. It must provide:

1. **A `sandbox` user and group.** Without it the container crash-loops with "sandbox user not
   found in image".
2. **`iproute2`** (the `ip` binary). Without it: "Network namespace creation failed ... trusted ip
   helper not found".
3. **A start command that works without the image's `ENV PATH`.** `openshell sandbox exec` replaces
   PATH, so spell the interpreter out in full: `/usr/local/bin/python /app/server.py` or
   `/app/.venv/bin/python -m pkg.serve`. With a venv, use a wrapper script, not a symlink, because a
   symlinked python misses `pyvenv.cfg`.

Appended to the staged Dockerfile automatically (don't add them yourself): the aiohttp egress shim,
the guardrail plugin package, writable `/etc/nemo-relay` and artifact dirs (made `a+rwX` as root,
then the image's final `USER` is restored), and, for `harness: hermes`, the Hermes Relay wiring and
judge dependencies. See `stage_relay_victim_build` in `agent_hardener/openshell/relay_victim.py`.

`examples/langgraph-victim/Dockerfile` is the minimal image that meets all of this.

### Check the image before going further

Do this even if you will not start the sandbox yourself — an image that does not answer locally will not answer in the sandbox either:

```bash
docker build -t my-agent:check .
docker run --rm -p 8000:8000 my-agent:check     # then, in another shell:
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

Hand over only after the request returns 200 with the agent's answer.

## Step 3: Write the manifest

Prefer generating it: run `agent-hardener inspect --project-dir <dir>` to see what's detected
(read-only), then:

```bash
uv run agent-hardener init --project-dir ../my-agent --name my-agent \
  --dockerfile Dockerfile --binary '/app/.venv/bin/**' \
  --start-command '/app/.venv/bin/python -m my_agent.serve' --port 8000 \
  --secrets INFERENCE_API_KEY --secrets-file .env \
  --egress api.tavily.com --backend ledger:8087 --yes
```

`--dockerfile` without `--binary` is rejected up front. `--yes` takes egress and backends from the
flags; without it, `init` prompts (answer `?` at a prompt for its help). It also writes
`garak-scan.yaml` and refuses to overwrite an existing manifest unless you pass `--force`.

Then edit the fields `init` doesn't set:

| Field | Set it to |
|---|---|
| `harness` | `langgraph`, `langchain`, `deepagents`, `hermes` or `other`. Needed for Hermes wiring and for the guardability preflight. |
| `relay_integration_confirmed` | `true` once the agent's tool calls go through Relay (`nemo-relay`). The run still verifies it from real traffic. |
| `binaries` | Globs for the processes allowed to egress, e.g. `["/usr/local/bin/python*"]`. Agent Hardener can't infer an image layout. |
| `env` | Non-secret env for the agent. Rewrite `localhost` backend URLs to `host.docker.internal`. |
| `relay_artifacts` | A **relative host** path (default `artifacts/relay`). An absolute `/home/...` path fails on macOS. |

Compare with `examples/langgraph-victim/agent-hardener.yaml`, which explains each field inline.

## Step 4: Secrets, egress, backends

- **Secrets.** List env-var *names* under `secrets`; values stay in `secrets_file`. Dotenv files are
  never copied into the build context. Values reach the sandbox through the shared OpenShell provider
  `agent-hardener-secrets`, which has a reuse gotcha (see Gotchas).
- **Egress.** The policy denies by default. `discover_egress: true` scans the project's code,
  Dockerfile `ENV` and config for hosts. Add the ones it can't infer (SDK defaults, runtime URLs)
  under `egress`. Private IPs are blocked unless they're resolved and listed (see `openshell-ops`).
- **Backends.** Services the tools call stay on the host. Give `compose_file` and `ports` (plus an
  optional `health_url`) to have Agent Hardener start them. Give just `ports` for an
  already-running service, or `host` and `allowed_ips` for one elsewhere, such as a Kubernetes
  service.

## Step 5: Bring it up

```bash
uv run agent-hardener up --config agent-hardener.yaml --env-file .env
uv run agent-hardener status --config agent-hardener.yaml
curl -s localhost:8000/health
```

`up` stages the context, builds `agent-hardener/<sandbox>:latest`, creates the sandbox
(`ah-<slug>-<digest>`) from the permissive policy, uploads `plugins.toml`, starts the agent and
forwards the port. The sandbox name is always derived from the agent name; to pick one yourself (on a
shared gateway, say), set `overrides: {victim_control: {config: {sandbox: <name>}}}` in the manifest.

Without local Docker, run `agent-hardener stage --config agent-hardener.yaml`, build and push the
directory it prints, then pass `--victim-image <ref>` to `up` or `run`. If the image wasn't built
from the current staged context, the run fails unless you pass `--allow-context-drift`.

## Verify

- `uv run pytest tests/agent_hardener/test_manifest.py tests/agent_hardener/test_relay_victim.py tests/agent_hardener/test_openshell_victim.py -q -o addopts=''`
  for a change to the manifest or staging code.
- `agent-hardener status` shows the sandbox Ready, `/health` returns 200, and one chat completion
  returns text.
- A one-round `agent-hardener run --config agent-hardener.yaml --rounds 1` gets past the Relay
  preflight and the attack stage.
- `just test-openshell-smoke-e2e` for a real OpenShell round trip (needs a connected gateway).

## Gotchas

- **Victim never gets healthy.** Look at the victim log the lifecycle prints. Usually it's a bare
  `python` in `start_command`, a missing `sandbox` user, or the agent binding `127.0.0.1` instead of
  `0.0.0.0`.
- **The sandbox keeps the old image.** An existing sandbox is reused unless a different
  `--victim-image` is configured. After changing the Dockerfile or code, run
  `agent-hardener down --config agent-hardener.yaml` first.
- **The agent name is the identity.** The sandbox, run artifacts and benign-profile cache are keyed
  on `name`, so use a different name for each variant.
- **Two docs are out of date.** `examples/agent-hardener.yaml` points to a relay-victim example
  folder and an `endpoint:` block, neither of which exists. The README quickstart and the `init`
  `--dockerfile` help still describe the old `workflow:` and `nat` setup. Trust
  `agent_hardener/manifest.py`.
- **Claude and Codex gateway agents can't be guarded.** Their tool calls bypass Relay;
  `agent_hardener/preflight/relay.py` fails such runs up front instead of reporting them as clean.
