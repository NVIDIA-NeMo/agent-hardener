<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connecting an agent to NeMo Relay

Scope: exactly what Agent Hardener needs and verifies. For other frameworks and attachment modes, use NeMo
Relay's own documentation; do not improvise wiring. NeMo Helix's `relay-attachment.md` reference teaches the
same obligations, so keep the two in step.

## The obligations — all of them, not one

1. **Start Relay and let it discover its config**, once, at startup:

   ```python
   import nemo_relay
   from nemo_relay.plugin import PluginConfig

   async with nemo_relay.plugin.activate(PluginConfig()):
       ...  # serve requests inside the block (e.g. a FastAPI lifespan)
   ```

   The empty `PluginConfig()` means "discover": Relay layers `/etc/nemo-relay/plugins.toml` over it, and that
   uploaded file is how Agent Hardener's guardrails *and* its trace sink reach the agent. Skip this call and
   every guardrail is inert. Keep the block open for the process's life: a bare
   `await nemo_relay.plugin.initialize(...)` whose result is discarded lets the plugin host shut down.

2. **Put Relay in the tool path.** Per framework:

   | Framework | The interception line |
   | --- | --- |
   | LangChain `create_agent` | `create_agent(model=…, tools=…, middleware=[NemoRelayMiddleware()])` — `from nemo_relay.integrations.langchain import NemoRelayMiddleware` |
   | Hand-built LangGraph graph | `graph.add_node("tools", create_tool_node(TOOLS))` — `from nemo_relay.integrations.langgraph import create_tool_node`; or, to keep your own `ToolNode`, `ToolNode(TOOLS, awrap_tool_call=NemoRelayMiddleware().awrap_tool_call)` |
   | DeepAgents | `create_deep_agent(**add_nemo_relay_integration(kwargs))` — `from nemo_relay.integrations.deepagents import add_nemo_relay_integration` |
   | Hermes | no code: set `harness: hermes` in the manifest and Agent Hardener adds the Relay wiring when it stages the image. Do not enable Hermes' old `observability/nemo_relay` plugin; it was removed and fails the build. |

   Install the matching extra: `nemo-relay[langgraph]` or `nemo-relay[langchain]`, at the version Agent
   Hardener's release notes name. `create_tool_node` takes the usual `ToolNode` options. A refused tool call
   comes back as a tool message saying it was refused, so the agent answers instead of crashing.

3. **Put Relay in the model path** (hand-built LangGraph graph only; the `create_agent` middleware and the
   other rows already do it). Send each model call through `nemo_relay.llm.execute` instead of calling the
   model directly:

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

   A bare `await llm.ainvoke(...)` in the model node costs two things. The trace of each attack has no prompt
   and no model steps, so the traces guardrails defender cannot match the attack to its trace and writes no
   guardrail from it. And after a guardrail refuses a tool, Agent Hardener cannot end the turn: it does that
   by answering the agent's next model call itself, which only works when that call goes through Relay.
   Nothing fails loudly; the run just hardens less.

4. *(Attribution, recommended)* open one agent scope per request and pass Relay's callback handler, so a
   turn's events nest under one root:

   ```python
   with nemo_relay.scope.scope("my-agent", nemo_relay.ScopeType.Agent):
       await graph.ainvoke(state, config={"callbacks": [NemoRelayCallbackHandler()]})
   ```

## The trap: instrumented-looking but unguarded

On a hand-built LangGraph graph, `NemoRelayCallbackHandler()` alone gives **visibility only**: graph scopes
appear in the trace while tool calls run outside Relay, so no guardrail is ever consulted. The agent looks
instrumented and every attack scores as unblocked while the defenses were never in the fight. The
interception is the tools node, not the callback.

The half-way version: the tools node is wired but the model node calls the model directly. Tool calls are
guarded, but the callback handler records no model call either, so attacks cannot be matched to their traces
and a refused attack keeps going. Obligation 3 closes it.

## How Agent Hardener verifies the promise

Two checks, at two moments; each failure names its cause:

| Error text contains | It means | Fix |
| --- | --- | --- |
| "wrote no Relay telemetry … the file was never created" | Relay never started | obligation 1: `plugin.activate(PluginConfig())` at startup, held open |
| "still empty after 30s; Relay is not attached, or its ATOF sink is disabled" | Relay started but emits nothing | the uploaded `plugins.toml` was not discovered: check `/etc/nemo-relay/` exists and is writable in the image |
| "emitted no new Relay events in 30s (N older records present)" | this request went untraced | the serving path bypasses the instrumented agent object |
| "Relay recorded no tool call at all" (after round 1) | tool calls bypass Relay — the trap above | obligation 2 |

Obligation 3 has no check. On a hand-built graph, confirm it yourself: after a run, the trace stream should
hold `llm` events beside the `tool` ones.

## The image contract

The agent's Dockerfile must provide:

- a `sandbox` user and group, and `iproute2`;
- every binary the start command needs at a full path (OpenShell replaces `PATH`; symlink venv binaries into
  `/usr/local/bin` if needed);
- the agent serving `GET /health` and OpenAI-style `POST /v1/chat/completions` on the manifest `port`.

When Agent Hardener stages the build it appends its egress shim and creates `/etc/nemo-relay` and the trace
directory, writable by the runtime user; do not add these yourself. An image built outside `agent-hardener stage`
misses them.

Then set in the manifest:

```yaml
agent:
  harness: langgraph               # or langchain, deepagents, hermes
  relay_integration_confirmed: true
```
