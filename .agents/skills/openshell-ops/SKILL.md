---
name: openshell-ops
description: Operate OpenShell for Agent Hardener — gateway repair, key rotation, egress policy, checking the live sandbox was hardened, cleanup. Use when "the gateway is disconnected", "my new API key isn't picked up" or "egress is blocked". Not for model choice (`models-and-providers`) or run triage (`run-and-debug`).
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Operate OpenShell

Every Agent Hardener victim runs in an OpenShell sandbox behind the local `auto-defender` gateway.
This skill covers the gateway, the secrets provider, the egress policy, and the sandbox's life
from create to delete.

## Scope

- **In scope:** gateway registration, the Docker driver, the per-agent secrets provider and
  key rotation, sandbox naming, `up`/`down`/`status`, policy and egress injection, checking
  hardening reached the live sandbox, cleanup.
- **Out of scope:** making an agent image start (`connect-victim-agent`), Relay plugin config
  (`nemo-relay`), which LLM each group uses (`models-and-providers`), a run failing for other
  reasons (`run-and-debug`).

## Key files

| What | Where |
|---|---|
| Gateway registration: Docker driver, `gateway.toml`, `openshell gateway add` | `agent_hardener/openshell/gateway.py`, `agent_hardener/cli/setup.py` |
| Sandbox lifecycle: up/down/restart, image build, provider, policy set/get, upload | `agent_hardener/openshell/lifecycle.py` |
| Config keys (`gateway`, `sandbox`, `provider`, `policy_path`, `forward`, ...) | `agent_hardener/openshell/config.py` |
| Sandbox name derived from the agent name | `agent_hardener/openshell/naming.py` |
| Backend and discovered-egress blocks injected into the policy | `agent_hardener/openshell/policy_egress.py` |
| Egress discovery (code, Dockerfile env, manual, OpenShell logs) | `agent_hardener/openshell/egress/` |
| Transient-error classification and retries | `agent_hardener/openshell/support.py` |
| Low-level helper CLI (`up`, `down`, `restart`, `status`, `ensure-provider`, `apply-policy`) | `agent_hardener/tools/openshell.py` |

Fixed values, set in `agent_hardener/manifest.py`:

| Item | Value |
|---|---|
| Gateway | `auto-defender` at `https://127.0.0.1:17670` |
| OpenShell version targeted | `v0.1.4` |
| Provider | `<sandbox name>-secrets`, one per agent, type `generic` |
| Sandbox name | `ah-<slug>-<7-char digest>`, at most 19 characters |

## Step 0: Preflight

```bash
openshell --version                          # v0.1.4 is what the code targets
openshell status --gateway auto-defender     # must say Connected
docker info >/dev/null && echo docker-ok
openshell sandbox list --gateway auto-defender
```

## Step 1: Register or repair the gateway

Run `uv run agent-hardener setup` (also part of `just setup` / `just install`). It's idempotent:

1. It writes the Docker driver and socket into `~/.config/openshell/gateway.toml`, keeping other
   keys. On macOS it also sets `grpc_endpoint` to `host.docker.internal`; otherwise every sandbox
   fails with `ControlSupervisorStartFailed`.
2. It restarts the gateway service: `brew services` on macOS, `systemctl --user restart
   openshell-gateway` on Linux. On Linux it also creates the `openshell-docker` network.
3. It runs `openshell gateway add https://127.0.0.1:17670 --local --name auto-defender` and waits
   up to 30 s for `Connected`.

The gateway step only warns on failure, so read its output. An `openshell` installed with
`uv tool install` has the CLI but no gateway service; `just setup` refuses it and prints the native
installer command.

## Step 2: Manage secrets and rotate keys

Read `ensure_provider` in `agent_hardener/openshell/lifecycle.py` before changing anything here.

- On a sandbox's first `up`, its provider `<sandbox name>-secrets` is created from the manifest's
  `secrets_file`. Only the names listed under `secrets` are used; if that list is empty, every key
  in the file is. A generated credentials-only profile named after the provider backs it.
- **After that, the provider is reused as-is.** Each agent has its own, so concurrent runs of different
  agents don't overwrite each other's keys. Manifests made before this change used one shared
  `agent-hardener-secrets` provider, which may still be on the gateway. Later runs only sync the
  safety judge's key (`AGENT_HARDENER_SAFETY_API_KEY`, through `provider update --credential`).
  **A changed or newly added victim secret in `.env` does not reach an existing provider.**
- Inside the sandbox, an env var holds a versioned placeholder that the gateway swaps for the real
  value at egress. A process started before an update keeps the old value, so the victim must be
  restarted or recreated.

To rotate a victim key on this branch:

```bash
uv run agent-hardener down --config agent-hardener.yaml           # every sandbox using the provider
openshell provider delete <sandbox name>-secrets --gateway auto-defender
uv run agent-hardener up --config agent-hardener.yaml --env-file .env   # recreated from current .env
```

The delete is refused while any sandbox still uses the provider, so check `openshell sandbox list`
first. Never put values in argv: the code passes keys by name with the value in the child process's
environment.

## Step 3: Egress and policy

- Every run starts from `agent_hardener/templates/relay-victim/openshell-policy-permissive.yaml`.
  The OpenShell defender narrows it each round.
- Two blocks are injected into the policy, idempotently: `relay_victim_backends` for the
  manifest's `backends` (as `host.docker.internal:<port>`) and `relay_victim_discovered_egress`
  for discovered hosts plus the manifest's `egress`.
- **To open a service on the host:** add a `backends:` entry (`name`, `ports`) to the manifest — or a host to
  `egress:` for an external one — then recreate the sandbox (`agent-hardener down`, then `agent-hardener up`).
  Editing the manifest alone changes nothing live. The job is done only when a request through the victim
  reaches the service.
- Host patterns: OpenShell rejects a bare `*` and wildcards on the top-level domain alone (like
  `*.com`), so three-label hosts become a wildcard over the parent domain. Hosts that resolve to
  private IPs get those IPs in `allowed_ips`; otherwise the anti-SSRF guard blocks them.
- `host.docker.internal` endpoints are rewritten to the running Docker daemon's bridge IP, which
  differs on Colima versus Docker Desktop.
- Apply a policy to a running sandbox through the lifecycle (`policy set --wait`). Read back the
  **base** policy with `openshell policy get <sandbox> --gateway auto-defender --base`;
  `sandbox get --policy-only` also includes rules the provider adds, so it never matches the file.

## Step 4: Confirm hardening reached the live sandbox

The run's hardened files are in `.agent-hardener/run-logs/<run>/victim-active-state/`; `init/`
holds the seeds. Compare them with the live sandbox:

```bash
SB=$(uv run python -c "from agent_hardener.openshell.naming import sandbox_name; print(sandbox_name('my-agent'))")
openshell policy get "$SB" --gateway auto-defender --base
openshell sandbox exec --gateway auto-defender --name "$SB" --no-tty -- sh -lc 'cat /etc/nemo-relay/plugins.toml'
```

A recreate re-uploads the active-state copies rather than the seeds, so the guardrails survive a
policy patch that forces a restart.

## Step 5: Lifecycle and cleanup

- `agent-hardener up`, `agent-hardener status` and `agent-hardener down` (each with `--config`)
  act on one manifest's sandbox. Remove a sandbox with `agent-hardener down`, not
  `openshell sandbox delete`, which leaves its port-forward process running. `down` stops the managed port forward and waits until the
  asynchronous delete finishes.
- `agent-hardener run --no-cleanup` leaves the sandbox and backends up; `--reuse` reuses a Ready
  one.
- `just stop` tears down the running Agent Hardener processes but keeps the gateway. Add
  `--stop-gateway` to stop it too.
- Images are tagged `agent-hardener/<sandbox>:latest`. Prune with `docker image prune`; avoid
  `docker system prune -a --volumes`.

## Verify

- `uv run pytest tests/agent_hardener/test_gateway.py tests/agent_hardener/test_egress.py tests/agent_hardener/test_host_gateway.py tests/agent_hardener/test_sandbox_naming.py tests/agent_hardener/test_openshell_policy_deploy.py tests/agent_hardener/test_openshell_tool.py -q -o addopts=''`
  after changing OpenShell code.
- `just test-openshell-smoke-e2e` runs a real sandbox through one round (needs a connected gateway).
- By hand: `openshell status --gateway auto-defender` is Connected, `agent-hardener status` is
  Ready, and the Step 4 comparison matches.

## Gotchas

- **`just openshell-bootstrap` and `just openshell-reset` are legacy.** Both run
  `scripts/bootstrap-openshell-victim.sh`, which targets the old agents-lab demo
  (its config, `vulnerable_demo_e2e.yaml`, is no longer in the repo). `openshell-reset` also runs
  `openshell sandbox delete --all`, which deletes **every** sandbox on the gateway. Use
  `agent-hardener setup` and `up`/`down` instead.
- **A new key isn't used.** The provider is reused (Step 2). This changes when PR #45
  (refresh the provider on key change) lands; update this skill then.
- **Transient gRPC resets** on `sandbox create` are retried, and the create is treated as
  successful if the sandbox appears. A persistent transport error means the gateway is down.
- **Kubernetes driver:** sandboxes run as uid 10001, not `sandbox`. The staged Dockerfile makes the
  run directories writable for any uid.
- The README's "Check That A Run Really Updated The Sandbox" and "OpenShell Helpers" sections
  describe the old workflow victim and missing example configs. Use Step 4 instead.
