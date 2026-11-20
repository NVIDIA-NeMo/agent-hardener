---
name: add-trace-provider
description: Add a new source of victim execution traces — a TraceProvider backend beside the Relay one, registered in the factory and contract-tested. Use when adding a tracing backend or reading traces from elsewhere, such as OTLP spans. Not for connecting an agent to Relay — use `nemo-relay`.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add a trace provider

Agent Hardener reads what the victim did on each request (its tool calls, their outcomes, its answer) from a
trace backend. Today there is one, NeMo Relay's ATOF event log. This skill adds another without anything
outside `agent_hardener/traces/` learning which backend recorded a trace.

## Scope

- In: a new `TraceProvider` subclass, its `case` in the factory, its name in `TracesConfig.provider`, its
  harness in the contract suite, and its own fixture tests.
- Out: matching traces to attacks, classifying refusals, or rendering. Those are backend-neutral and already
  shared. A provider that re-implements any of them is wrong.
- Out: getting a victim to *emit* traces. For Relay that is `nemo-relay`.

## The design in one paragraph

A provider returns **all** traces as `AgentTrace` objects and does nothing else. Consumers (the guardrails
defender, the validators, replay) pick what they need with `agent_hardener/traces/matching.py`, which reads
only `AgentTrace` fields. Refusal and denial detection lives once in `agent_hardener/traces/signals.py`. So a
new backend is a translation layer: source records in, `AgentTrace` out.

## Step 0: Preflight

```bash
just test-fast        # green before you start
```

Read, in this order:

1. `agent_hardener/traces/provider.py` — the ABC and its four rules (quoted below).
2. `agent_hardener/models/traces.py` — `AgentTrace`, `LlmCall`, `ToolCall`, `ToolOutcome`. These cross the
   agent boundary as JSON, so they must round-trip exactly.
3. `agent_hardener/traces/relay.py` — the reference implementation.
4. `agent_hardener/traces/factory.py` and `TracesConfig` in `agent_hardener/models/session.py`.

## Step 1: Map your source to `AgentTrace`

Decide, before writing code, what in your backend is:

| `AgentTrace` field | Relay's answer | Yours |
|---|---|---|
| one trace | one completed top-level `agent` scope | ? |
| `trace_id` | the root scope's uuid | must be stable across calls |
| `input` / `output` | the scope's start/end data, else first user message / last LLM output | ? |
| `started_at` / `ended_at` | scope timestamps (the sandbox clock) | must be the victim's clock, not the host's |
| `steps` | nested `llm` and `tool` scopes, ordered by start time | ? |
| `guardrail_latched` | a latch mark under the root, or a blocked canned answer | via `signals.py` |
| tool raised vs returned | the tool scope's error data | your source must record it: the contract suite needs a raised tool to classify as `error` |

Skip requests that are still in flight: a trace with no end is not a trace yet.

## Step 2: Write the provider

Create a new module (named after the backend) in `agent_hardener/traces/` with a `TraceProvider` subclass. Its one method,
`get_traces()`, must keep the four rules in `agent_hardener/traces/provider.py`:

1. **Never raise on missing data** — return `[]`. Wrap the fetch; log a warning instead.
2. **Fresh data on every call.** No caching across calls: callers read a clock anchor before sending traffic
   and filter after (`trace_clock` / `traces_since` in `agent_hardener/traces/matching.py`).
3. **Classify with the shared signals.** Fill `ToolCall.ok`, `outcome` and `rail` with `classify_tool_call`,
   and `guardrail_latched` with `LATCH_EVENT` / `is_blocked_response`, all from
   `agent_hardener/traces/signals.py`. Never add your own refusal patterns.
4. **Never lose history.** If the source can be reset (a sandbox recreate starts a fresh event file), keep
   what you already saw. Relay does this by rotating raw copies into `events.<n>.atof.jsonl` under the run's
   `traces/` directory and reading every generation.

`get_traces()` is synchronous. The attack and benign validators call it from separate worker threads at the
same time, so guard fetch-save-read with a lock as `RelayTraceProvider` does.

Save the raw source under the `raw_dir` you are given (`<run_dir>/traces/`), best-effort: a failed save must
still return traces.

## Step 3: Register it

1. Add the name to `provider: Literal["relay"]` in `TracesConfig` (`agent_hardener/models/session.py`).
2. Add a `case "<backend>":` to `build_trace_provider` in `agent_hardener/traces/factory.py`. Return `None`
   (with a warning) when the victim gives the backend nothing to read.
3. If the backend needs inputs the factory does not take yet (only `relay_spec`, `atof_path`, `sync` and
   `raw_dir` today), add them to the factory signature and pass them from `_build_trace_provider` in
   `agent_hardener/runtime/runner.py`. Keep that the only change outside the package.

Users pick it in the manifest with `traces: {provider: <backend>}`; the manifest passes it through to the
session config. Settings the backend needs (a file path, a URL) go as optional fields on `TracesConfig` next
to `provider`; change the factory signature (point 3) only for data that comes from the runner.

## Step 4: Test it

1. **Contract suite.** Add a harness class to `HARNESSES` in
   `tests/agent_hardener/traces/test_provider_contract.py`. It writes backend-neutral `Request`s into your
   source format, resets the source, and breaks the raw copy. Every contract test (empty source, partial
   writes, in-flight skipped, fresh data, raw saved, history survives a reset, shared signal classification,
   JSON round-trip) then runs against your backend unchanged.
2. **Fixture mapping tests.** Add a captured stream from a real victim next to the Relay ones in
   `tests/fixtures/atof/` (or a sibling directory for a non-ATOF format), and a provider test module in
   `tests/agent_hardener/traces/` that pins the mapping, the way
   `tests/agent_hardener/traces/test_relay_provider.py` pins `langchain_concurrent.atof.jsonl`.
3. **Wiring.** `tests/agent_hardener/traces/test_trace_wiring.py` checks the provider reaches exactly the
   agents that consume traces. Add a case if you changed the factory signature.

For tests of code that *consumes* traces, use `FakeTraceProvider` from `agent_hardener/traces/testing.py`
rather than a real backend.

## Verify

Run all three before reporting done.

```bash
uv run pytest tests/agent_hardener/traces -q -o addopts=''
just test-fast
just test-pipeline-e2e
```

All three green, and the contract suite shows your backend as a separate parameter (`[<backend>]`) on each
test. Then run one real war-game with `traces: {provider: <backend>}` and check that
`<run_dir>/traces/` holds the raw source and that the guardrails defender wrote rails (it writes none when it
gets no traces).

## Gotchas

- **Timestamps come from the victim.** Matching windows use trace timestamps precisely so host/sandbox clock
  skew cannot shift them. Do not stamp traces with the host clock.
- **Python 3.11 parses at most microseconds.** Relay's ATOF carries nanoseconds; see `_timestamp` in
  `agent_hardener/traces/relay.py`.
- **A refused tool call usually "succeeds".** The guardrail returns its refusal as the tool result rather
  than raising, so the backend's own status is not enough. That is why `classify_tool_call` checks the
  result text first.
- **Don't keep the LLM prompt.** `LlmCall` deliberately stores only the answer; storing messages would put
  the victim's full conversation into every saved trace.
- **`None` from the factory is legitimate** (a plain HTTP target has no sandbox to read from). The
  guardrails defender then writes no rails and says why, rather than the run crashing.
