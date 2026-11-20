---
name: models-and-providers
description: Choose the models Agent Hardener uses and point them at another OpenAI-compatible provider (OpenRouter, a local server, a NIM). Use when changing the attacker, analysis or safety-judge model or endpoint, asking which env var sets one or which value wins (config, export, --env-file, default), or when the model check fails.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Models and providers

Agent Hardener calls LLMs in four places. Each is steered separately, and every client is OpenAI-compatible,
so "switching provider" means changing a model name, a base URL and a key — never a code change.

## Scope

- In scope: which model each group uses, the env vars and manifest fields that change it, the pre-run
  model check, moving a group to a non-NVIDIA endpoint.
- Out of scope: how a key reaches the sandbox as an OpenShell provider (`openshell-ops`); how the victim
  agent itself is wired (`connect-victim-agent`).
- Naming trap: `agent_hardener/providers/backends.py` is **not** about LLM providers. It starts the Docker
  Compose backends a victim's tools call. Nothing in this skill touches it.

## The four model groups

| Group | Who calls it | Default | Steered by | Code |
| --- | --- | --- | --- | --- |
| Attack | garak red-team model + detector | `nvidia/nemotron-3-ultra-550b-a55b` at `https://integrate.api.nvidia.com/v1/` | `GARAK_RED_TEAM_MODEL_{NAME,URI,TYPE}`, `GARAK_DETECTOR_MODEL_{NAME,URI,TYPE}`, or the same keys lower-cased in the manifest `garak:` block — **but `garak-scan.yaml` wins when it exists** (`init` always writes it); env and manifest keys apply only when it is absent | `agent_hardener/agents/attackers/agent_breaker/config.py` |
| Analysis | defenders + benign validator (synthesis and judging) | same model, NVIDIA endpoint | `AGENT_HARDENER_MODEL`, `AGENT_HARDENER_BASE_URL`, key `INFERENCE_API_KEY` | `agent_hardener/llm.py` |
| Safety | the guardrail judge that runs inside the victim | the analysis model and endpoint | model: `safety_llm` in the `defender-guardrails` entry's `config`; endpoint/key: `AGENT_HARDENER_SAFETY_BASE_URL`, `AGENT_HARDENER_SAFETY_API_KEY` | `agent_hardener/relay_plugin/config.py`, `agent_hardener/agents/defenders/traces_guardrails_defender/nodes/component_writer.py` |
| Victim | the agent under test | whatever the agent's own code uses | the agent's code and its secrets file | not Agent Hardener's — see `connect-victim-agent` |

Keys: analysis uses `INFERENCE_API_KEY`. garak authenticates with `NIM_API_KEY`, which
`agent_hardener/garak_venv.py` fills from `INFERENCE_API_KEY` when unset. The judge uses
`AGENT_HARDENER_SAFETY_API_KEY`, else `INFERENCE_API_KEY`. The full catalogue of names is in
`agent_hardener/env.py`.

## Step 0: Preflight

1. Read the current values: `env | grep -E '^(AGENT_HARDENER_|GARAK_|INFERENCE_API_KEY|NIM_API_KEY)'` and the
   `garak:` block of `agent-hardener.yaml`.
2. Confirm the new endpoint serves an OpenAI-compatible `GET /models` with your key:
   `curl -s -H "Authorization: Bearer $KEY" "$BASE_URL/models" | head`. The pre-run check uses the same call.
3. Note the exact model id the provider lists — ids differ per provider (OpenRouter's are `vendor/model`).

## Step 1: Know which value wins

Every setting resolves from the first of these that has a value, highest first:

1. **Explicit config** — a field in the session config or manifest: the benign validator's per-component
   `model` / `base_url`, the `garak:` block's model keys, `safety_llm`.
2. **The process environment** — a shell `export`, CI, or `just`, whose `set dotenv-load` loads the repo's
   `.env` before Python starts.
3. **`--env-file`** (default `.env`, loaded by `run`, `up`, `synth-benign` and `serve`). `load_local_env` in
   `agent_hardener/tools/openshell.py` only fills variables that are unset or empty. When an export differs from
   the file it warns `<KEY> is set in the environment and differs from <file>; using the environment value.`
   — the key, never the value. `unset` the export to use the file.
4. **The built-in default** — `nvidia/nemotron-3-ultra-550b-a55b` at `https://integrate.api.nvidia.com/v1`.
   An empty value counts as unset at every level.

`AGENT_HARDENER_MODEL` / `AGENT_HARDENER_BASE_URL` are read on each call (`default_model()` /
`default_base_url()` in `agent_hardener/llm.py`), so an `--env-file` value reaches every group. Keys
(`INFERENCE_API_KEY`, `NIM_API_KEY`) follow the same order with no default.

**The attack group has one more layer on top:** when `garak-scan.yaml` exists (every `init` project has one),
a run uses its model fields as they are, and the `garak:` block and `GARAK_*` variables no longer change the
attack model. Without it: `garak:` block, then `GARAK_*` (environment, then `--env-file`), then the default
(`_resolve` in `agent_hardener/agents/attackers/agent_breaker/config.py`).

So to change a model, set it once at the highest level you control and check nothing above it still holds an
old value: `env | grep -E '^(AGENT_HARDENER_|GARAK_)'` for stale exports, and `garak-scan.yaml` for the attack
model.

## Step 2: Move the analysis group

```bash
export AGENT_HARDENER_MODEL=<model id from /models>
export AGENT_HARDENER_BASE_URL=https://openrouter.ai/api/v1
export INFERENCE_API_KEY=<provider key>
```

Exporting or putting them in `--env-file` both work. The benign validator can instead take a per-component
`model` / `base_url` in its validator `config` (`agent_hardener/agents/validators/smart_benign/config.py`),
which beats the env pair for that component only; the env pair is the one lever for everything.

## Step 3: Move the attack group

If the project has a `garak-scan.yaml` (every project made with `init` does), edit `red_team_model_name`,
`red_team_model_config.uri`, `detector_model_name` and `detector_model_config.uri` there: a run uses that file
as its base and only forces the target URI and reporting fields onto it. Still export the variables below — the
pre-run check reads `GARAK_RED_TEAM_MODEL_NAME`/`_URI`, and the replay validator's detectors read
`GARAK_DETECTOR_*` rather than the scaffold.

```bash
export GARAK_RED_TEAM_MODEL_NAME=<model id>
export GARAK_RED_TEAM_MODEL_URI=https://openrouter.ai/api/v1/
export GARAK_DETECTOR_MODEL_NAME=<model id>
export GARAK_DETECTOR_MODEL_URI=https://openrouter.ai/api/v1/
export NIM_API_KEY=<provider key>   # garak's key; otherwise it inherits INFERENCE_API_KEY
```

garak runs in its own venv with an allow-listed environment. Anything else it needs (e.g. `HF_TOKEN`) goes in
`AGENT_HARDENER_GARAK_ENV_PASSTHROUGH` as a comma-separated list of names.

## Step 4: Move the safety judge

Set `AGENT_HARDENER_SAFETY_BASE_URL` and `AGENT_HARDENER_SAFETY_API_KEY`. Unset, the judge follows
`AGENT_HARDENER_BASE_URL` and `INFERENCE_API_KEY`. The judge runs inside the sandbox; Agent Hardener
allow-lists exactly the host `judge_endpoint()` returns, so no egress edit is needed. Choose the judge model
with `safety_llm` in the `defender-guardrails` defender `config` (a full session config, or `overrides:` in
the manifest — `overrides` replaces lists wholesale, so restate the whole `defenders` list).

## Step 5: Tune concurrency for the provider's rate limits

Each fan-out defaults to 2 in flight (`agent_hardener/concurrency.py`). Env wins over config:
`AGENT_HARDENER_DEFENDER_CONCURRENCY`, `AGENT_HARDENER_VALIDATOR_CONCURRENCY`,
`AGENT_HARDENER_ATTACK_VALIDATOR_CONCURRENCY`, `AGENT_HARDENER_BENIGN_VALIDATOR_CONCURRENCY`. Lower them for a
tightly rate-limited key. Slow reasoning models: raise `garak.model_timeout` and `defenders.timeout_s`.

## Verify

1. `uv run agent-hardener run --config agent-hardener.yaml` — the pre-run check
   (`preflight_configured_models` in `agent_hardener/model_check.py`) fails in seconds with the list of
   reachable models if a name, URL or key is wrong.
2. `uv run pytest tests/agent_hardener/test_model_check.py -q` after changing anything in the check itself.
3. In the run log, confirm calls go to the new host.

## Gotchas

- **garak model settings in env or the manifest are ignored when `garak-scan.yaml` exists.** The pre-run check
  then validates a model garak never uses. Edit the scaffold (Step 3).
- **"Differs from .env; using the environment value".** An old export (often from an earlier shell or CI)
  shadows the file. `unset` it, or change the export instead.
- **Restating `defenders:` in `overrides` pins their timeouts.** Lists replace, so a copied entry's
  `timeout_seconds` wins over a later `defenders.timeout_s`.
- **The pre-run check only covers env overrides.** It checks `AGENT_HARDENER_MODEL` and
  `GARAK_RED_TEAM_MODEL_NAME`, nothing else. A model set in the manifest `garak:` block, the detector model,
  and `safety_llm` are not checked and fail later, mid-run.
- **Missing key on the default endpoint fails fast.** No `INFERENCE_API_KEY` and no `AGENT_HARDENER_BASE_URL`
  raises `ModelUnavailableError` before the sandbox builds. A custom base URL skips the presence check, so a
  keyless local server works.
- **Nemotron gets a reasoning-off body field.** Any model whose name contains `nemotron` is sent
  `chat_template_kwargs.enable_thinking=false` (`no_reasoning_body` in `agent_hardener/relay_plugin/config.py`).
  A provider that rejects unknown fields will fail on Nemotron-named models only.
- **garak's generator type stays `nim`** (`nim.NVOpenAIChat` for the red team) unless you set
  `GARAK_RED_TEAM_MODEL_TYPE` / `GARAK_DETECTOR_MODEL_TYPE`.
- **The judge config says `provider = "nvidia"` whatever the endpoint.** It is a label; the judge client
  (`agent_hardener/relay_plugin/judge.py`) is `ChatOpenAI` against `base_url`.
