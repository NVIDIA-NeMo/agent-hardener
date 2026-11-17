<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connecting an agent to NeMo Relay

Agent Hardener needs two things from the agent, and checks both:

1. **Relay is started.** Call `nemo_relay.plugin.initialize(PluginConfig())` once at startup and keep the
   returned handle alive for the process's life. Relay layers the guardrails Agent Hardener delivers
   (`/etc/nemo-relay/plugins.toml`) over the empty base config. Without this call, delivered guardrails never
   activate. The run's first probe fails with "victim not instrumented" if Relay emits nothing.
2. **Every tool call passes through Relay.** Proven later, from real attack traffic. An agent whose LLM calls
   are traced but whose tool calls bypass Relay cannot be guarded, and the run fails instead of reporting
   weak defenders.

Install `nemo-relay` with the extra for your framework (`nemo-relay[langgraph]` or `nemo-relay[langchain]`),
at the version Agent Hardener's release notes name.

## LangGraph (hand-built graph)

```python
import nemo_relay
from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler, NemoRelayMiddleware
from nemo_relay.plugin import PluginConfig
from langgraph.prebuilt import ToolNode

relay = NemoRelayMiddleware()
graph.add_node("tools", ToolNode(TOOLS, awrap_tool_call=relay.awrap_tool_call))  # the interception line

# at server startup, for the life of the process:
async with await nemo_relay.plugin.initialize(PluginConfig()):
    ...

# per request:
with nemo_relay.scope.scope("my-agent", nemo_relay.ScopeType.Agent):
    await graph.ainvoke(state, config={"callbacks": [NemoRelayCallbackHandler()]})
```

A bare `ToolNode(TOOLS)` runs tools outside Relay. A refused tool call surfaces from `awrap_tool_call` as an
exception; turn it into a tool message (`handle_tool_errors=`) so the agent answers rather than crashes.

## LangChain (`create_agent`)

LangChain agents take Relay as agent middleware:

```python
from nemo_relay.integrations.langchain import NemoRelayMiddleware

agent = create_agent(model, tools, middleware=[NemoRelayMiddleware()])
```

Start Relay with `nemo_relay.plugin.initialize(...)` at startup exactly as above.

## The image contract

The agent's Dockerfile must provide:

- a `sandbox` user and group, and `iproute2`;
- every binary the start command needs at a full path (OpenShell replaces `PATH`; symlink venv binaries into
  `/usr/local/bin` if needed);
- `/etc/nemo-relay` and the trace directory created and owned by `sandbox` before start. A missing
  `/etc/nemo-relay` fails at deploy with `Permission denied`.

Agent Hardener appends its own egress shim to your Dockerfile when it stages the build; do not add it yourself.

Then set in the manifest:

```yaml
agent:
  harness: langgraph               # or langchain
  relay_integration_confirmed: true
```
