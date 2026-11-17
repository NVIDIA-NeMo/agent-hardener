---
name: add-defender
description: Add a defender to Agent Hardener, or a mitigation to an existing one — code that turns an attack hit into an OpenShell policy change or a Relay guardrail. Use when adding a defender or mitigation, or asking how defender output gets deployed. Not for attackers — use `add-attacker`.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add a defender

A defender gets one attack hit at a time and answers with a mitigation: a stricter OpenShell sandbox
policy, or a guardrail in the victim's NeMo Relay plugin config. Defenders differ from every other agent
role — they all edit the *same* victim state, so the manager merges their output before anything is
deployed.

## Scope

In scope:

- A new **defender**: `run(DefenderInput) -> DefenderOutput`, registered in a run's `defenders` list.
- A new **mitigation node** inside the OpenShell defender (`agent_hardener/agents/defenders/openshell_defender_v2/nodes/`).
- Changing the **traces guardrails defender** (`agent_hardener/agents/defenders/traces_guardrails_defender/`).

Adjacent work, other skills:

- Where the attack hits come from → `add-attacker`.
- The execution traces the guardrails defender reads → `add-trace-provider`.
- How Relay loads the guardrails a defender writes → `nemo-relay`.
- The models defenders call → `models-and-providers`.

## Step 0: Preflight

```bash
uv sync
just test-fast        # baseline must be green
```

Read, in this order:

1. `agent_hardener/runtime/stages/defense.py` — builds the manager input (current policy, active-state
   Relay plugins path, benign suite) and calls each defender with a timeout.
2. `agent_hardener/agents/defenders/defenders_manager.py` — routing, one task per (hit × defender),
   the policy merge, and how a `DefenderOutput` becomes a policy patch.
3. `agent_hardener/models/contracts.py` — `DefenderInput`, `DefenderOutput`, `DefenderAnalysis`.
4. The defender closest to yours: `agent_hardener/agents/defenders/openshell_defender_v2/` (sandbox
   policy) or `agent_hardener/agents/defenders/traces_guardrails_defender/` (Relay guardrail).

**Done means green.** Your last actions before reporting done are `just test-fast` and `just lint-check`;
a passing test file of your own is not enough.

## Step 1: Pick the output kind

A defender's output decides what the deploy stage does with it (`_build_policy_patches` in the manager,
consumed by `agent_hardener/runtime/adapters.py`):

| Your defender produces | Return | Becomes |
|---|---|---|
| A stricter OpenShell policy | `ok=True`, `new_policy_yaml=<full policy YAML>` | Merged with the other defenders' candidates, written once to the run's active-state policy, deployed as `openshell_policy_candidate` |
| A Relay guardrail | `ok=True`, `resource_type="relay_guardrail_component"`, plus the rendered config in `new_policy_yaml`; the defender writes the guardrail into `relay_plugins_path` itself | `relay_plugins_candidate`; the sandbox is recreated with the new plugin config |
| Nothing — out of scope, or no safe fix | `ok=False`, `error_message="<code>: <detail>"` | Logged; no patch |

## Step 2: Write the defender

Create a package for it under `agent_hardener/agents/defenders/` with an entry module exposing:

```python
def run(defender_input: DefenderInput) -> DefenderOutput: ...
```

Async works too. Sync defenders run in a thread (`asyncio.to_thread`), several at once, so shared
state must be thread-safe.

What `DefenderInput` gives you, per hit:

- `attack_prompt`, `agent_response`, `attacked_tool` — the hit itself.
- `benign_requests` / `benign_suite` — the synthesized benign suite. Your fix must not break these.
- `attack_trace` — the attack's execution trace, or `None` when none was recorded.
- `relay_plugins_path` — the run's active-state Relay config, for guardrail defenders.
- `relay_victim_spec` — the victim's own env, egress and backends; treat those endpoints as benign.
- `context` — run context **merged with your entry's `config:` dict**. Notable keys: the current
  policy text under `CURRENT_POLICY_KEY`, `victim_policy_path`, and `defender_extraction_cache`
  (an `ExtractionCache` from `agent_hardener/agents/defenders/extraction_cache.py` — use it to share
  one expensive extraction across the parallel defender calls of a round).
- `feedback` — set only when the retry loop is on and a validator rejected your last attempt.

For LLM calls use `build_chat_model` from `agent_hardener/llm.py`, so `AGENT_HARDENER_MODEL` /
`AGENT_HARDENER_BASE_URL` retarget your defender with the others.

**Abstain rather than force a fix.** Every attack is routed to every defender (`_route_attacks`). A
defender that cannot help with a hit must return `ok=False` with a reason code — the OpenShell defender's
`_abstain_output` is the model to copy.

## Step 3: Register it

Each defender entry needs a non-empty `capabilities` string (the model rejects it otherwise).

- **Try it on one run** — manifest `overrides:`. Lists replace, so repeat the built-ins you keep:

  ```yaml
  overrides:
    defenders:
      - name: my-defender               # role is filled in from the section
        implementation: agent_hardener.agents.defenders.my_defender.agent:run
        timeout_seconds: 900
        capabilities: "Mitigates <attack class> by <mechanism>."
        config: {threshold: 0.8}
  ```

  A full session config takes the same list under `defenders:` (see
  `tests/fixtures/pipeline_e2e/pipeline.yaml`).
- **Make it a default** — add it in `_default_defenders` in `agent_hardener/manifest.py` and assert it in
  `tests/agent_hardener/test_manifest.py`. Keep the import path stable afterwards: manifests and the Helix
  plugin name it. `agent_hardener/agents/defenders/guardrails_defender.py` exists only to keep such a path
  fixed while the implementation moves.

### Adding a mitigation to the OpenShell defender instead

Add a node module under `agent_hardener/agents/defenders/openshell_defender_v2/nodes/` implementing the
`MitigationNode` protocol in its `base.py` (`name`, `resource_type`, `allowed_ops`, `synthesize`), and
register it in that package's `__init__.py`. The merge in `policy/merge.py` understands only the two
existing edit kinds (endpoint removal and deny rules); a new edit kind needs a merge rule and a lint rule
in `policy/lint.py`.

## Step 4: Tests

Copy the closest existing test:

- Defender logic, LLM mocked → `tests/agent_hardener/agents/defenders/openshell_defender_v2/test_openshell_defender_v2_agent.py`
  or the files in `tests/agent_hardener/agents/defenders/traces_guardrails_defender/`.
- Concurrent defenders on one policy → `tests/agent_hardener/agents/defenders/test_defenders_manager_policy_merge.py`.
- Routing and zero-hit behaviour → `tests/agent_hardener/agents/defenders/test_defenders_manager_routing.py`.

Cover at least: a hit you fix, a hit you abstain on, and a benign request your fix must keep working.
Also add one test that loads a copy of `tests/fixtures/pipeline_e2e/pipeline.yaml` with your entry under
`defenders:` through `load_config` (`agent_hardener/config.py`), so "selectable from a session config" is proven.

## Verify

Run these before reporting done — your defender's own test file passing is not enough.

1. `just test-fast`, then `just lint-check` and `just test-coverage` (100% on new code).
2. `just test-pipeline-e2e` — proves the patch *types* deploy end to end. It swaps in a fake defenders
   manager, so it does **not** run your defender's code; step 1 must cover that.
3. Live, on a recorded attack so it is cheap: `agent-hardener run --config agent-hardener.yaml --replay`
   (add `--with-traces` for a trace-reading defender). Check the iteration's defender component files
   under `.agent-hardener/run-logs/`, then that the attack validator now reports the hit blocked and the
   benign validator still passes.

**Measuring it.** A passing run shows the defender works once. To know whether it improves outcomes,
and what it costs benign traffic, run it as an arm against a no-defenders control in the separate
agent-eval repo.

## Gotchas

- **One widening candidate blocks everyone's policy change.** The manager merges all policy candidates
  and lints the result as a strict contraction of the original. If it widens access, the merged policy
  is not written at all that round — a single bad defender silently cancels the others.
- **Candidates are computed against the same snapshot.** Never write `victim_policy_path` yourself;
  return the YAML and let the manager merge.
- **Guardrail defenders share one file.** The traces guardrails defender serialises writes with a lock
  in `nodes/component_writer.py`; any defender that appends to `relay_plugins_path` needs the same.
- **No trace, no guardrail.** The traces guardrails defender returns `ok=False` for any hit without an
  `attack_trace`. A replay must load traces (`--with-traces`), and the victim must run under Relay.
- **The timeout discards work.** A defender past `timeout_seconds` (default 900s, manifest
  `defenders.timeout_s`) is cancelled and contributes nothing.
- **Routing is not capability-based yet.** `capabilities` is required and documented, and the defense
  stage's docstring still describes an LLM router — but `_route_attacks` sends every hit to every
  defender. Write your defender to scope itself.
- **Only `resource_type="network"` hot-reloads.** Any other value on a policy candidate makes the deploy
  stage recreate the sandbox (slower, but always applied).
