---
name: agent-hardener
description: Use when the user wants to harden, red-team or war-game their own LangChain, LangGraph or custom-image agent with the agent-hardener CLI — setting it up, writing the manifest, generating a benign suite, running attack/defend/validate rounds, choosing models, or reading results and the hardened image bundle.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Hardener — war-game your agent

Agent Hardener attacks your agent inside an OpenShell sandbox, lets defenders write a tighter sandbox
policy and NeMo Relay guardrails, then replays the attacks and a benign suite to prove the hardened agent
blocks them without breaking normal requests. What leaves a run is a report and a buildable hardened image.

Running through NeMo Helix? Use Helix's `nemo-agent-hardener` skill instead.

**Only war-game agents the user owns or is authorised to test**, and point the agent's tools at test
backends: the run sends real attacks and the agent may call its tools in ways nobody intended.

## Interview first

Before running anything, ask the user — one short round, not one question at a time:

1. Which agent? Framework (LangGraph, LangChain, other) and where its project lives.
2. Does it already build into a Docker image? Which Dockerfile, and what command starts it?
3. Is it connected to NeMo Relay (see Step 2)? If not, that is the first job.
4. Which model key do they have (`INFERENCE_API_KEY`), and for which endpoint?
5. Which hosts and local services does the agent call (egress, backends)?

Confirm before anything destructive: `agent-hardener down`, deleting `.agent-hardener/`, `init --force`.

## Step 0: Preflight

| Need | Check |
| --- | --- |
| Python 3.11–3.13 | `python --version` |
| Docker running | `docker info` |
| OpenShell CLI + gateway service (0.1.4) | `openshell --version`, then `openshell status` → `Status: Connected` |
| A model key | `INFERENCE_API_KEY` in `.env` or the shell |

Install OpenShell with its native installer, not `uv tool install openshell` — that installs the CLI only,
with no gateway service. See [references/troubleshooting.md](references/troubleshooting.md) for the
installer line and the 0.0.x upgrade.

## Step 1: Install

```bash
pip install nvidia-agent-hardener
agent-hardener setup     # one-time: garak's own venv + registers the `auto-defender` OpenShell gateway
```

garak is not a dependency on purpose; until `setup` runs there is no attacker. `setup` is idempotent;
`--force` rebuilds the garak venv.

After upgrading the package, `agent-hardener skills install --force` refreshes this skill so it matches the
new version (`--target agents` installs it for Codex and Cursor too).

## Step 2: Make sure the agent is Relay-connected

Agent Hardener reads tool calls from Relay's trace stream and delivers guardrails as Relay config. The agent
must (a) start Relay with `nemo_relay.plugin.activate(PluginConfig())` at startup, held open, (b) route
every tool call through Relay, and (c) on a hand-built LangGraph graph, route every model call through
`nemo_relay.llm.execute` too. A run fails fast when (a) or (b) is missing rather than report fake results;
nothing catches a missing (c), which quietly weakens the hardening. The wiring per framework is in
[references/relay-connection.md](references/relay-connection.md); the image contract (a `sandbox` user,
`iproute2`, writable `/etc/nemo-relay`) is there too.

`agent-hardener inspect --project-dir <path>` (offline) shows what Agent Hardener detects: image, secrets,
egress. Add `--json` for machine-readable output.

## Step 3: `agent-hardener init`

Run it in the directory you will run from — everything resolves against the current directory.

```bash
agent-hardener init --project-dir <agent project> --dockerfile Dockerfile \
  --binary '/usr/local/bin/python*' --start-command '/usr/local/bin/python /app/server.py' \
  --port 8000 --secrets INFERENCE_API_KEY --secrets-file .env \
  --egress api.example.com --backend ledger:8087
```

Without flags it is interactive (answer `?` at a prompt for help); `--yes` takes defaults with no prompts.
It writes:

- `agent-hardener.yaml` — the manifest (`--output` to rename). Run `init` again to update it in place:
  the project is checked again, flags you pass are applied, and everything else you set (and your
  comments) is kept; it prints what changed. A failed update leaves the file untouched. `--force`
  rewrites it from scratch.
- `garak-scan.yaml` — the attacker's config, editable; an update keeps it (the run sets its target).
- the secrets file — prompts (hidden) for any listed secret it does not find.

Then edit the manifest: set `harness: langgraph` or `harness: langchain` and, once Step 2 is done,
`relay_integration_confirmed: true`. Field reference: [references/manifest.md](references/manifest.md).

## Step 4: Generate the benign suite

```bash
agent-hardener synth-benign --config agent-hardener.yaml --env-file .env
```

Brings the sandbox up, interviews the user about normal use, writes
`.agent-hardener/benign_profiles/<agent name>/requests.csv`, and tears down. `--yes` accepts the recommended
answers; `--no-interactive` skips the interview (rules-only suite, for CI). Ask the user to review the CSV;
editing rows is expected.

## Step 5: Run the war-game

```bash
agent-hardener run --config agent-hardener.yaml --env-file .env \
  --benign-suite .agent-hardener/benign_profiles/<agent name>/requests.csv
```

- `--rounds N` — repeat attack → defend → validate N times (default 1).
- `--benign-suite` is required: without it the run stops before starting, with `benign_suite_failed`. Run
  `agent-hardener synth-benign` first and pass the `requests.csv` it writes.
- `--replay` re-runs the last run's recorded hits instead of attacking live (cheaper; good after a change).
  Pass a hitlog path to replay a specific one.
- `--reuse` keeps a Ready sandbox; `--no-cleanup` leaves sandbox and backends running afterwards.
- `--policy` / `--relay-plugins` start from your own OpenShell policy or Relay `plugins.toml`.
- `-v` / `--verbose` opens the full report (transcripts, garak detail, policy diffs) in a pager.
- `--debug` prints every log record on the console: engine detail, OpenShell commands, tracebacks and
  third-party library logs. It works on every command, before or after its name
  (`AGENT_HARDENER_LOG_LEVEL=debug` does the same).

Exit code 0 means every round succeeded. A run takes tens of minutes; run it in the background and
watch the log.

Sandbox only, no war-game: `agent-hardener up`, `agent-hardener status`, `agent-hardener down`
(confirm first — it deletes the sandbox). No local Docker? `agent-hardener stage` writes the build
context; build and push it elsewhere and pass `--victim-image <ref>` to `run`, `up` or `synth-benign`.

## Choosing models

| Group | What | Override |
| --- | --- | --- |
| Attack | garak red-team and detector models | `garak-scan.yaml` (`red_team_model_name`, `red_team_model_config.uri`, `detector_model_*`); without that file, `GARAK_RED_TEAM_MODEL_NAME`, `GARAK_RED_TEAM_MODEL_URI`; key `NIM_API_KEY`, else `INFERENCE_API_KEY` |
| Analysis | defenders + benign suite generation and judging | `AGENT_HARDENER_MODEL`, `AGENT_HARDENER_BASE_URL`; key `INFERENCE_API_KEY` |
| Safety | the guardrail judge inside the hardened agent | `AGENT_HARDENER_SAFETY_BASE_URL`, `AGENT_HARDENER_SAFETY_API_KEY` (else the analysis ones) |
| Agent | the agent's own LLM | the agent's own config; pass its key via `secrets` |

Default model for attack and analysis: `nvidia/nemotron-3-ultra-550b-a55b`. `run` checks any model you
overrode against its endpoint before building the sandbox and lists the reachable models if it is wrong.

### Which value wins

Highest first; the first level that has a non-empty value is used:

1. Explicit config: a model field in the manifest or session config (e.g. the benign validator's `model` /
   `base_url`). For the attack model, `garak-scan.yaml` sits here and overrides everything below it.
2. A variable already in the environment: a shell `export`, or CI.
3. The `--env-file` file (default `.env`). It only fills variables that are unset or empty. If an export
   differs from it, the command warns `<KEY> is set in the environment and differs from .env; using the
   environment value.` — `unset` that export to use the file.
4. The built-in default, `nvidia/nemotron-3-ultra-550b-a55b` at `https://integrate.api.nvidia.com/v1`.

API keys follow the same order, with no default.

### Changing the model

1. List what the endpoint serves: `curl -s -H "Authorization: Bearer $KEY" "$BASE_URL/models"`, and copy the
   exact model id.
2. Analysis: put `AGENT_HARDENER_MODEL`, `AGENT_HARDENER_BASE_URL` and `INFERENCE_API_KEY` in `.env` (or export
   them). Check `env | grep AGENT_HARDENER_` for an old export that would win over the file.
3. Attack: edit the model names and `uri` fields in `garak-scan.yaml`, and also set `GARAK_RED_TEAM_MODEL_NAME` /
   `GARAK_RED_TEAM_MODEL_URI` so the pre-run check tests the same model. Re-running `init` rewrites that file.
4. Run. The pre-run check fails within seconds, listing the reachable models, if a name, URL or key is wrong.

## Reading the results

Everything lands under `.agent-hardener/` in the working directory:

| Path | What |
| --- | --- |
| `logs/<command>-<UTC time>.log` | the complete log of one command, every record and traceback included. Every command prints its path as `→ log: <path>` when it ends; set `AGENT_HARDENER_LOG_FILE=<path>` to write it elsewhere |
| `run-logs/<run id>/agent-hardener.log` | the run's log, including build and infra output |
| `run-logs/<run id>/round_<n>/report.json` | the round report: `success`, attacks, each iteration's defenders, patches and validators |
| `run-logs/<run id>/init/` | the agent's files before hardening |
| `run-logs/<run id>/victim-active-state/` | the files as last deployed (hardened policy, guardrails) |
| `run-logs/<run id>/hardened-image/` | **the deliverable**: a Dockerfile + guardrails that build the hardened agent |
| `benign_profiles/<agent name>/requests.csv` | the benign suite |

The console summary tallies validators per kind, e.g. `attack: failed=1, ok=5 | benign: ok=3`:

- `attack` ok = a recorded attack no longer lands; failed = it still does.
- `benign` ok = normal requests still work; failed = the hardening over-blocked.

Report counts, never a bare percentage: "5 of 6 attacks blocked, 3 of 3 benign requests passed".
To adopt the result, build `hardened-image/` and deploy that image.

`.agent-hardener/` holds the attacks that worked against the agent — treat it like a credential, and keep
it out of version control.

## Re-running and cleanup

- Changed the agent? Rebuild and `run` again; `--replay` to check the same attacks first.
- Stuck sandbox: `agent-hardener status`, then (after confirming) `agent-hardener down`.
- Fresh start: delete `.agent-hardener/` only after the user confirms and has kept anything they need.

## Gotchas

- `start_command` and `binaries` must use full paths: OpenShell replaces the image's `PATH`, so a bare
  `python` starts nothing.
- `relay_artifacts` is a host path; keep it relative (default `artifacts/relay`).
- The sandbox denies all egress by default. A host the agent calls but the manifest does not list shows up as
  failing tool calls, not as an error. Add it to `egress`, or `backends` for services on this machine.
- `localhost` inside the sandbox is not your machine: `--backend NAME:PORT` rewrites it to
  `host.docker.internal`.
- A changed API key needs the sandbox rebuilt: run without `--reuse`.
- More symptoms and fixes: [references/troubleshooting.md](references/troubleshooting.md).
