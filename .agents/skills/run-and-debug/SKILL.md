---
name: run-and-debug
description: Run Agent Hardener locally and work out why a run failed. Use when setting up a dev machine, starting or stopping a war-game, asking where logs and reports go, picking a test tier, or triaging an error category like benign_suite_failed. Not for adding components — use add-* skills.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run and debug Agent Hardener

How to get a war-game running from a clean checkout, where everything it writes lands, and how to get from
a failure message to its cause.

## Scope

- In scope: setup, the CLI lifecycle, artifacts and logs, the test tiers, triage.
- Out of scope: choosing models (`models-and-providers`), the OpenShell gateway itself (`openshell-ops`),
  wiring Relay into an agent (`nemo-relay`), packaging a victim (`connect-victim-agent`).

## Step 0: Preflight

- Docker running, the `openshell` CLI on `PATH`, `uv` and `just` installed.
- `INFERENCE_API_KEY` exported (or in `.env`). See `models-and-providers` before changing any model.

## Step 1: One-time setup

```bash
just setup              # scripts/setup.sh: prerequisites, .env, uv sync, then `just install`
just garak-venv         # = uv run agent-hardener setup: garak venv + local OpenShell gateway
```

`agent-hardener setup --force` recreates the garak venv (`~/.agent-hardener/garak-venv`). garak is kept out of
the main venv on purpose; never `uv add` it.

## Step 2: Describe the agent

```bash
uv run agent-hardener inspect --project-dir ../my-agent      # read-only: image, secrets, egress it detects
uv run agent-hardener init --project-dir ../my-agent         # writes agent-hardener.yaml + garak-scan.yaml
```

`init --yes` is non-interactive; pass `--egress`, `--backend NAME:PORT`, `--dockerfile` with `--binary`
as needed. The agent must be NeMo Relay-connected or the run stops at the Relay preflight (`nemo-relay`).

## Step 3: Build a benign suite

```bash
uv run agent-hardener synth-benign --config agent-hardener.yaml
```

It builds the sandbox, probes the live victim, and writes `profile.json` + `requests.csv` under
`.agent-hardener/benign_profiles/<target>/`. Pass that CSV to `run --benign-suite`; without it the run stops
before starting anything, with `benign_suite_failed`.

## Step 4: Run the war-game

```bash
uv run agent-hardener run --config agent-hardener.yaml --benign-suite <requests.csv>
```

Useful flags (all on `run`):

| Flag | Use |
| --- | --- |
| `--rounds N` | iterative hardening rounds (default 1) |
| `--reuse` | skip the rebuild when a Ready sandbox is already running |
| `--no-cleanup` | leave sandbox and backends up for inspection |
| `--replay [PATH]` | skip the attacker; replay the last run's garak hitlog (or a given one) — the fastest loop when debugging defenders/validators |
| `--with-traces [PATH]` | with `--replay`, the attack traces to use |
| `--policy PATH`, `--relay-plugins PATH` | start from a given OpenShell policy / Relay `plugins.toml` |
| `--victim-image REF` | run a pre-built image (build it from `agent-hardener stage`) |
| `--verbose` | full final report in a pager |

Sandbox by hand: `agent-hardener up`, `agent-hardener status`, `agent-hardener down`. Tear down everything:
`just stop`.

## Step 5: Find what the run wrote

Everything is under `.agent-hardener/` in the working directory (`storage.root_dir`):

| What | Where |
| --- | --- |
| Run log (all build/infra output) | `.agent-hardener/run-logs/<run_id>/agent-hardener.log` |
| Per-round and per-iteration artifacts | `.agent-hardener/run-logs/<run_id>/round_<N>/…` (`agent_hardener/swarm_tracker/layout.py`) |
| garak reports and hitlogs | the run's garak dir, unless `garak.report_dir` is set |
| OpenShell command logs | `.agent-hardener/openshell-logs/` |
| Staged victim build context, policies | `.agent-hardener/builds/<sandbox>/`, `.agent-hardener/policies/` |
| Victim tool-call trace | `events.atof.jsonl` in the agent's `relay_artifacts` dir |

Re-render the final report from disk: `uv run python -m agent_hardener.final_log --root .agent-hardener`
(`--mission-id` narrows it; code in `agent_hardener/final_log/`).

For scripts and CI, set `AGENT_HARDENER_ERROR_FILE=<path>`: on failure the CLI writes
`{category, message, remediation, stack}` there (`agent_hardener/cli/_errors.py`).

## Step 6: Observability (optional)

`just obs-start` (or `obs-start-bg`) starts Phoenix on `:6006` and an OTel collector on `:4318`;
`just obs-status`, `just obs-stop`. The war-game does not need it — its traces come from Relay's ATOF
stream — but it shows the LLM spans of components that export OTel.

## Verify: test tiers

| Command | What it covers | Needs |
| --- | --- | --- |
| `just test-fast` | unit tests, no slow/integration | nothing |
| `just test-pipeline-e2e` | the whole pipeline, logically, with fakes (`tests/agent_hardener/test_pipeline_e2e.py`) | nothing |
| `just test-openshell-smoke-e2e` | a real sandbox up/down (`tests/agent_hardener/test_openshell_smoke_e2e.py`) | Docker, OpenShell gateway; skips otherwise |
| `just test` / `just test-coverage` | everything | — |

## Triage

Every classified failure has a category (`agent_hardener/errors.py`); match the category or symptom.

| Symptom / category | Likely cause | Open next |
| --- | --- | --- |
| `model_unavailable` before the sandbox builds | wrong model id, URL or key; missing `INFERENCE_API_KEY` | `models-and-providers` |
| Model override ignored, run still uses Nemotron | an older export shadows `--env-file` (look for the `differs from` warning), or `garak-scan.yaml` holds the attack model | `models-and-providers` |
| `sandbox` | Docker down, gateway not connected | `openshell-ops` |
| `victim_unavailable` at build | wrong `project_dir`/Dockerfile, BYO image missing `--binary` | `connect-victim-agent` |
| `victim_unavailable` mid-run | agent crashed, often after a malformed guardrail or policy was deployed; read the agent log | `openshell-ops`, then the defender whose output was deployed last |
| `victim_not_instrumented` | agent lacks Relay middleware or never starts Relay (`nemo_relay.plugin.activate()`) | `nemo-relay` |
| Few or no guardrails written, attack traces have no prompt | a hand-built LangGraph model node bypasses Relay (no `nemo_relay.llm.execute`) | `nemo-relay` |
| `attacker_failed` | garak timed out on a heavy agent | raise `garak.timeout_s`, lower `garak.max_attempts_per_tool`; `add-attacker` |
| `benign_suite_failed` | no usable suite | run `agent-hardener synth-benign`, pass `--benign-suite` |
| Run "passes" but nothing was blocked | policy never reached the sandbox, or guardrails never loaded | `openshell-ops` (check the sandbox was updated), `nemo-relay` |
| Defenders time out, few guardrails written | slow analysis model | raise `defenders.timeout_s`; `models-and-providers` |
| Victim can't reach a host | missing egress / backend route | `connect-victim-agent` |

## Gotchas

- `run` never generates a benign suite. Without `--benign-suite` (or `benign_suite_path` in the manifest) the
  default smart-benign validator stops the run up front with `BenignSuiteError` (`benign_suite_failed`).
  Run `synth-benign` first. (The README and the `--benign-suite` help still say it is skipped; they are wrong.)
- **A failed run still tears down.** Even a run that fails at a preflight check runs `down`, which deletes the
  sandbox named after the manifest's agent if one exists. Pass `--no-cleanup` to keep a sandbox you are debugging.
- `--reuse` only skips the rebuild when the sandbox is Ready *and* matches; otherwise it builds fresh.
- `just run` loads the repo's `.env` into the environment before Python (`set dotenv-load`), so its values beat
  any other `--env-file` you pass. Order: config, environment, `--env-file`, default (`models-and-providers`).
- A `0 hits` run after an attacker timeout is not a pass — the attacker raises `attacker_failed` instead,
  and a run that shows zero hits without it deserves a look at the garak report.
