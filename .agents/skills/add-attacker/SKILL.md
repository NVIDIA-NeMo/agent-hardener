---
name: add-attacker
description: Add a new attack source to Agent Hardener — a live attacker, or recorded hits replayed instead. Use when asked to add a new attacker, plug in another red-team tool, replay saved attacks, or change the garak agent_breaker attacker. Not for defenders — use `add-defender`.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add an attacker

An attacker turns "attack this victim" into a list of hits: prompts that got the victim to do something
it should not. Every later stage — the defenders and the attack replay validator — works only from those
hit records, so the record shape is the contract that matters.

## Scope

In scope:

- A new **live attacker**: an async `run(request, agent)` that returns an `AttackRecord`.
- Changing the built-in **garak agent_breaker** attacker (its config, override fields, or a garak fork).
- **Preloaded** attacks: replaying recorded hits instead of attacking live.

Adjacent work, other skills:

- Writing the mitigation for a hit → `add-defender`.
- Choosing the red-team or detector model, or switching provider → `models-and-providers`.
- Collecting execution traces of the attacks → `add-trace-provider`.

## Step 0: Preflight

```bash
uv sync
just test-fast            # the baseline must be green before you change anything
just garak-venv           # only if you will run the garak attacker; provisions ~/.agent-hardener/garak-venv
```

Read these first — they are short:

- `agent_hardener/runtime/stages/attack.py` — how the attack stage loads preloaded hits, fans out over
  `config.attackers`, writes `attacks.json`, and fails the round when a live attacker returns `ok=False`.
- `agent_hardener/agents/attackers/agent_breaker/run_agent_breaker.py` — the reference live attacker.
- `agent_hardener/agents/attackers/preloaded.py` and `agent_hardener/agents/attackers/hits.py` — the
  replay path and the shared record normalizer.

**Done means green.** Your last actions before reporting done are `just test-fast` and `just lint-check`;
a passing test file of your own is not enough.

## Step 1: Decide live or preloaded

| You have | Build |
|---|---|
| A tool or model that generates attacks against a running victim | A live attacker (Steps 2–4) |
| A file of hits recorded elsewhere (JSON or JSONL) | Nothing new — use preloaded attacks (Step 5) |
| A garak probe change | Edit garak, not Agent Hardener — overlay your fork with `just garak-dev` (Step 6) |

Preloaded is the faster loop for defender work: deterministic, no red-team model calls.

## Step 2: Write the live attacker

Create a module under `agent_hardener/agents/attackers/` (a package if it needs more than one file, as
`agent_hardener/agents/attackers/agent_breaker/` does). Expose an async callable:

```python
async def run(request: AgentRunInput, agent: AgentConfig) -> AttackRecord: ...
```

How it is called (`agent_hardener/runtime/agent_loader.py`): the runner imports the
`implementation` string `module.path:run`, and passes `request`, `agent` and optionally `ctx` by
parameter name. A sync function is run in a thread.

What it receives:

- `request.target.base_url` — the victim's chat-completions URL.
- `request.context[ARTIFACT_DIR_KEY]` — a round-scoped directory for anything you write (reports, logs).
- `agent.config` — your attacker's own settings, straight from the session config.
- `ctx.emit_progress(msg, current, total)` — optional live progress for the UI.

## Step 3: Return records the rest of the pipeline can read

Build each hit with `normalize_hit_record(raw, index, source)` from
`agent_hardener/agents/attackers/hits.py`, then return
`AttackRecord(agent_id=agent.agent_id, agent_name=agent.name, ok=..., summary=..., records=[...])`
(model in `agent_hardener/models/contracts.py`).

Keys downstream code reads from each record:

| Key | Read by | Notes |
|---|---|---|
| `prompt` | defenders, trace matching, replay validator | A plain string, or garak's `{"turns": [{"content": {"text": ...}}]}` |
| `response` or `output` | defenders (the victim's transcript) | String, or `{"text": ...}` |
| `source` / `probe` / `detector` | replay validator | See the gotcha on attack-type detection |

Set `ok=False` with an `error` when the scan did not finish. The stage then raises `AttackerError` rather
than report "0 hits" — a false "the victim resisted". Mirror agent_breaker: a crash that still produced
hits is `ok=True` with a "partial run" summary.

An explicit zero-hit result is `ok=True`, `records=[]` and `metadata={"hit_count": 0}`; the defenders
manager treats that as "nothing to defend", not as a failure.

## Step 4: Wire it into a run

Manifests (`agent-hardener.yaml`) get their attacker list from `_default_attackers` in
`agent_hardener/manifest.py`. Two ways in:

- **Try it on one run** — add an `overrides:` block to the manifest. Lists *replace*, so include
  garak's entry too if you want both:

  ```yaml
  overrides:
    attackers:
      - name: my-attacker               # role is filled in from the section
        implementation: agent_hardener.agents.attackers.my_attacker:run
        timeout_seconds: 1800
        config: {max_prompts: 50}
  ```

  A full session config (see `tests/fixtures/pipeline_e2e/pipeline.yaml`) takes an `attackers:` list
  directly.
- **Make it a default** — append it in `_default_attackers` and cover it in `tests/agent_hardener/test_manifest.py`.

Keep `timeout_seconds` (the wrapper's cap) above any cap the attacker enforces itself, so your own
timeout fires first and you return a partial result instead of being killed.

## Step 5: Preloaded attacks instead of a live attacker

- From the CLI: `agent-hardener run --replay` replays the latest run's garak hitlog;
  `agent-hardener run --replay path/to/hitlog.jsonl` replays a specific one, and `--with-traces` loads
  the attack traces recorded with it. Replay drops the live attackers.
- In a full session config: list `preloaded_attacks` entries (`name`, `path`, optional `source`,
  `metadata`, `traces_path` — `PreloadedAttackConfig` in `agent_hardener/models/infra.py`) and set
  `attackers: []`.

Each hit becomes its own `AttackRecord` named `preloaded-<name>-<index>`.

## Step 6: Changing the garak attacker

- Settings live in `GarakSettings` (`agent_hardener/models/infra.py`). Every field maps one-to-one to
  a key the attacker reads, so a new field reaches `agent.config` without extra plumbing. Model fields
  also read `GARAK_<FIELD>` env vars (resolution in
  `agent_hardener/agents/attackers/agent_breaker/config.py`).
- `agent-hardener init` writes an editable `garak-scan.yaml`; each run overlays only the target URI and
  reporting keys on it.
- To test a garak fork: `just garak-dev ../garak` installs it editable into the garak venv.
  `AGENT_HARDENER_GARAK_PYTHON` or `GARAK_COMMAND` point the attacker at another interpreter or command.

## Verify

Run all of these before reporting done — your own test file passing is not enough.

1. Unit tests next to the existing ones:
   - Live attacker → model on `tests/agent_hardener/test_garak_attackers.py` (mock the subprocess or
     client; assert the `AttackRecord` shape and the `ok` rules).
   - Stage behaviour → `tests/agent_hardener/test_attack_stage.py`.
   - Preloaded parsing → `tests/agent_hardener/test_preloaded_attacks.py`.
   - Default wiring → `tests/agent_hardener/test_manifest.py`.
   - Selectable from a session config → a test that loads a copy of
     `tests/fixtures/pipeline_e2e/pipeline.yaml` with your entry under `attackers:` through
     `load_config` (`agent_hardener/config.py`) and resolves its `implementation` with `load_run_callable`.
2. Pipeline wiring without a sandbox: point a copy of `tests/fixtures/pipeline_e2e/pipeline.yaml` at
   your attacker (the fixture uses `run_attacker` in `tests/fixtures/pipeline_e2e/mock_agents.py`)
   and run `just test-pipeline-e2e`.
3. Live: `agent-hardener run --config agent-hardener.yaml` against a real victim, then open the
   round's `attacks.json` and the attacker component file under `.agent-hardener/run-logs/`.
4. `just test-fast`, `just lint-check` and `just test-coverage` (100% on new code).

## Gotchas

- **The replay validator only recognises garak-shaped hits.** `identify_attack_type` in
  `agent_hardener/agents/validators/garak_replay/parsing.py` looks for `agent_breaker` /
  `agent-breaker` (direct) or `indirect_injection` (indirect) in `source`, `probe` or `detector`.
  A record with neither is **skipped silently**: it is never replayed, so it can never count as blocked.
  Tagging a record `probe: agent_breaker` makes the validator replay it as a direct prompt injection —
  right for a prompt-list attacker, but keep your own name in `source` so reports still show where the
  hit came from. For any other attack type, extend that function and its tests instead.
- **The defenders read `prompt` as a string.** `defenders_manager.py` builds each defender's
  `attack_prompt` with `str(record.get("prompt"))`, so a dict prompt reaches defenders as its repr.
  Prefer a plain string `prompt` for new attackers.
- **The attacked tool is parsed from the prompt** (`Use the <tool> tool to|for …`) unless the
  `AttackRecord.metadata` carries `attacked_tool`. Set it when your prompts don't follow that phrasing.
- **Attack traces are matched by prompt.** The guardrails defender needs each attack's trace; traces
  are keyed by the normalized prompt (`attack_key` in `agent_hardener/traces/attacks.py`). The prompt
  you record must be the text you actually sent.
- **A killed scan scores nothing.** garak writes its hitlog at the end; raise `garak.timeout_s` in the
  manifest for agents with many tools rather than let the wrapper kill it mid-probe.
- The README's "Preloaded Versus Live Attacks" section names a vulnerable_demo_e2e.yaml example and a
  use-config-attackers flag; neither exists on main. Use `--replay` or a full session config as above.
