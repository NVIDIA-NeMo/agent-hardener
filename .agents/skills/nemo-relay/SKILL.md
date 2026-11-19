---
name: nemo-relay
description: Connect a LangChain or LangGraph agent to NeMo Relay so its tool calls are traced and guarded; change the Relay guardrail plugin (judge, latch, blocking); bump nemo-relay. Use when wiring Relay into an agent or traces are missing. Not for deploying the victim into the sandbox — use `connect-victim-agent`.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Relay

Agent Hardener sees what a victim did through NeMo Relay's ATOF event stream, and enforces the guardrails it
writes through a Relay plugin running inside the victim. Both only work if the agent's tool calls pass
through Relay. This skill covers the three jobs that touch Relay.

## Scope

- **A. Connect an agent** — LangChain (`create_agent`) or a hand-built LangGraph graph.
- **B. Change the guardrail plugin** — the code under `agent_hardener/relay_plugin/`.
- **C. Move the nemo-relay version** — here and in NeMo Helix, together.
- Out: building the image and deploying it into OpenShell (`connect-victim-agent`); reading traces on the host
  side (`add-trace-provider`).

## Step 0: Preflight

```bash
grep nemo-relay pyproject.toml   # the version range Agent Hardener is built against
just test-fast
```

Read `examples/langgraph-victim/agent.py` and `examples/langgraph-victim/server.py` end to end. They are the
reference victim and every rule below is commented there.

---

## A. Connect a LangChain or LangGraph agent

The agent writes no Agent Hardener code. It owes Relay five things (four for LangChain `create_agent`).

### Step A1: Install Relay with the right extra

In the agent's image: `nemo-relay[langgraph]` (LangGraph) or `nemo-relay[langchain]` (LangChain), inside the
range in this repo's `pyproject.toml`. The guardrail plugin refuses to start on a Relay without
`ToolExecutionContext` (0.9+), and the seeded `plugins.toml` needs 0.10+ (see Gotchas).

The extras raise the framework floor: nemo-relay 0.11's extras need `langgraph>=1.2.14` and
`langchain>=1.3.15`. Loosen exact pins on those before `uv add`, or the resolve fails.

### Step A2: Route tool calls through Relay

This is the line that matters. Without it the agent still emits LLM telemetry, so the start-up probe passes,
but no guardrail can refuse a tool and the run fails its tool-path check.

- **LangChain:** `create_agent(..., middleware=[NemoRelayMiddleware()])`, importing it from
  `nemo_relay.integrations.langchain`.
- **LangGraph, hand-built graph:** `ToolNode(TOOLS, awrap_tool_call=relay.awrap_tool_call, handle_tool_errors=...)`
  with `relay = NemoRelayMiddleware()` from `nemo_relay.integrations.langgraph`. A bare `ToolNode(TOOLS)` runs
  tools outside Relay. Set `handle_tool_errors` so a refusal becomes a tool result instead of an HTTP 500 —
  a 500 is scored as a broken victim, not a working guardrail (see `_refusal` in the example).

The code disagrees with itself on refusals: the example's `_refusal` says a refusal surfaces from
`awrap_tool_call` as `RuntimeError`, while `agent_hardener/relay_plugin/autopatch.py` says Relay's intercept
returns it. Keep the LangGraph handler (it is harmless either way). For LangChain, check once that a blocked
tool call still gets an HTTP 200 answer; if it gets a 500, catch the refusal in a `wrap_tool_call` middleware.

### Step A2b: Route model calls through Relay (hand-built LangGraph only)

LangChain's `create_agent` middleware already covers the model. A hand-built graph's model node must send each
call through `nemo_relay.llm.execute` instead of calling `llm.ainvoke` directly:

```python
import nemo_relay
from langchain_core.messages import SystemMessage, messages_from_dict, messages_to_dict
from nemo_relay.integrations.langchain._serialization import LANGCHAIN_MODEL_RESPONSE_KEY, LangChainCodec

async def model(state: MessagesState) -> dict[str, list]:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    codec = LangChainCodec()
    request = nemo_relay.LLMRequest({}, {"messages": messages_to_dict(messages), "model": MODEL})

    async def call(request: nemo_relay.LLMRequest) -> Any:
        reply = await llm.ainvoke(messages_from_dict(request.content["messages"]))
        return {LANGCHAIN_MODEL_RESPONSE_KEY: {"messages": messages_to_dict([reply])}}

    result = await nemo_relay.llm.execute(MODEL, request, call, model_name=MODEL, codec=codec, response_codec=codec)
    return {"messages": messages_from_dict(result[LANGCHAIN_MODEL_RESPONSE_KEY]["messages"])}
```

With a bare `llm.ainvoke`, Relay records no LLM span. Tool guardrails still fire, but each attack's trace has
no prompt and no model steps, so the traces guardrails defender cannot match the attack to its trace and
writes no rail from it. The callback handler (Step A4) does not record these calls either. Nothing in the run
fails on this; it only shows up as a defender that never writes rails. `examples/langgraph-victim/agent.py`
does not do this yet — follow the snippet, not the example, for the model node. Keep this step in line with
NeMo Helix's `relay-attachment.md` reference, which teaches the same wiring.

### Step A3: Initialise Relay's plugin host at startup

```python
async with await nemo_relay.plugin.initialize(PluginConfig()):
    yield   # e.g. in a FastAPI lifespan
```

`PluginConfig()` is empty on purpose: Relay layers the `plugins.toml` that Agent Hardener uploads to
`/etc/nemo-relay/` over it. That file carries the ATOF sink and, each round, the guardrails. Without this call
neither activates. Keep the `async with`: Relay tears the plugin host down when the handle is dropped.

### Step A4: Open one agent scope per request

```python
with nemo_relay.scope.scope("my-agent", nemo_relay.ScopeType.Agent):
    result = await agent.ainvoke(..., config={"callbacks": [NemoRelayCallbackHandler()]})
```

The host builds one trace per completed top-level `agent` scope (`build_traces` in
`agent_hardener/traces/relay.py`). Tool and LLM scopes outside one are not attributed to any request. The
callback handler adds LLM and graph telemetry; it is observability only and cannot refuse anything.

### Step A5: Declare it in the manifest

In `agent-hardener.yaml`, set `harness: langchain` or `harness: langgraph`, and
`relay_integration_confirmed: true` once Step A2 is done (see `examples/langgraph-victim/agent-hardener.yaml`).
The run still verifies it from real traffic; the flag records who said so. The image also needs
`/etc/nemo-relay/` and `/tmp/agent-hardener/relay/` writable by the `sandbox` user (see
`examples/langgraph-victim/Dockerfile`).

### Confirm traces arrive

Locally first, without a sandbox: copy `agent_hardener/templates/relay-victim/plugins.toml` to
`$XDG_CONFIG_HOME/nemo-relay/plugins.toml` (Relay's user config layer), point its `output_directory` at a temp
dir, start the agent, send one request that uses a tool, and check `events.atof.jsonl` there for a
`category: tool` scope named after your tool and a `category: llm` scope, both inside a completed
`category: agent` scope. No `llm` scope on a hand-built graph means Step A2b is missing.

Then run the war-game against the agent. Two checks fail the run loudly if Relay is not wired:

1. **Start-up probe** (`check_relay_instrumentation` in `agent_hardener/preflight/relay.py`) sends a greeting
   and waits for ATOF records. Failure raises `VictimNotInstrumentedError` with a diagnosis.
2. **Tool-path check** (`check_tool_path`, same file) runs on real attack traffic and raises if Relay saw no
   tool calls — the signature of Step A2 being skipped.

After a run, the run directory's `traces/events.atof.jsonl` holds the raw stream; it should contain `tool`
scopes named after your tools.

---

## B. Change the guardrail plugin

| File | Owns | Imports Relay? |
|---|---|---|
| `agent_hardener/relay_plugin/config.py` | The wire format the defender writes and the plugin reads (`GuardrailsComponentConfig`, `extra="forbid"`), judge endpoint and key resolution | no |
| `agent_hardener/relay_plugin/policy.py` | The refuse/allow decision for one rail | no |
| `agent_hardener/relay_plugin/judge.py` | The LLM safety judge and score parsing | no |
| `agent_hardener/relay_plugin/latch.py` | Remembers blocked requests so later tools and the final answer are refused too | no |
| `agent_hardener/relay_plugin/blocked.py` | The canned answer a blocked request gets | lazily |
| `agent_hardener/relay_plugin/plugin.py` | Registration, one tool intercept per rail (priority `100 + index`), the latch intercepts, `validate()` | yes |
| `agent_hardener/relay_plugin/autopatch.py` | Registers the plugin kind from a `sitecustomize`, before `initialize()` | yes |
| `agent_hardener/relay_plugin/victim.py` | Where the victim writes ATOF | no |

Keep decision logic in the Relay-free modules so it stays unit-testable. The plugin is copied into the
victim image from this same tree by `agent_hardener/openshell/relay_victim.py`, so the defender that writes the
config (`agent_hardener/agents/defenders/traces_guardrails_defender/nodes/component_writer.py`) and the plugin
that reads it are always the same version — a new config field needs both sides in one change.

Contracts with the host side, in `agent_hardener/traces/signals.py`:

- The refusal text `agent-hardener[<rail>]: refused <tool>` (`_refusal` in `policy.py`) is parsed to
  classify a tool call as `guardrail_refused`. Change one, change the other.
- `DEFAULT_BLOCKED_MESSAGE` is copied, not imported (importing the plugin loads Relay);
  `tests/agent_hardener/traces/test_signals.py` keeps the copies equal.
- The latch mark `agent_hardener.latched` is a string literal in both `plugin.py` (`_record_latch`) and
  `LATCH_EVENT` in `signals.py`, and **no test ties them**. Rename both or neither.

`AGENT_HARDENER_DISABLE_LATCH=1` in the victim turns the latch off for debugging.

Tests live in `tests/agent_hardener/relay_plugin/`; `test_plugin_intercepts.py` drives real Relay intercepts.

---

## C. Move the nemo-relay version

Relay is pinned in this repo's `pyproject.toml` and, separately, in **NeMo Helix**, which runs the same victims.
They move together: a victim image built against one and a host or Helix plugin expecting the other is how
schema mismatches reach users.

1. Read the Relay release notes for: the ATOF record shape, the observability component schema version, the
   tool-result wrapper, and the plugin/intercept API.
2. `uv add "nemo-relay>=X.Y.0,<X.(Y+1)"` (always a pinned range; it updates `uv.lock` too).
3. Update what encodes Relay's shape:
   - the observability `version` in `agent_hardener/templates/relay-victim/plugins.toml` (a rejected schema
     fails the whole `initialize()`, so the victim never starts);
   - `TOOL_RESULT_SCHEMA` and `_tool_result` in `agent_hardener/traces/relay.py`;
   - `_require_context_intercepts` in `agent_hardener/relay_plugin/plugin.py` if the intercept API moved;
   - the pin in `examples/langgraph-victim/Dockerfile`.
4. Re-capture a real stream into `tests/fixtures/atof/` if the record shape changed.
5. Open the matching NeMo Helix change for its Relay pin in the same window, and say in both PRs that they
   depend on each other.

## Verify

```bash
uv run pytest tests/agent_hardener/relay_plugin tests/agent_hardener/preflight tests/agent_hardener/traces -q -o addopts=''
just test-fast
just test-openshell-smoke-e2e
agent-hardener run --config examples/langgraph-victim/agent-hardener.yaml --env-file .env
```

The last run must pass both Relay checks and its report must show at least one tool call refused by a rail.
For a LangChain agent, run the same against its manifest.

## Gotchas

- **The plugin kind must be registered before `initialize()`.** Relay rejects an unknown `[[components]]` kind
  and the victim fails to start once the first guardrail is written. `autopatch.py` handles this via
  `sitecustomize`; never move registration to an import side effect.
- **Schema 3 vs 4.** The seed `plugins.toml` uses observability schema 4, which nemo-relay 0.10+ requires.
  Check that the victim's image installs a Relay at least that new.
- **No `[[plugins.dynamic]]`.** Dynamic plugins need CLI-managed lifecycle state a file upload cannot supply.
- **Hermes** is the one harness that needs the `adaptive` component in the seed; other harnesses ignore it.
- **`NemoRelayMiddleware` wraps model calls too and needs a model name.** A custom chat model without
  `model_name` fails with `'None' is not an instance of 'str'`.
- **`start_command` must use full paths.** `openshell sandbox exec` does not pass the image's `PATH`.
