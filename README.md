# Agent Hardener

## Operating safely

## How it works

### The hardening loop

### Attackers

### Defenders

### Validators

### Victim control

## Prerequisites

## Setup

### Install Docker

### Install OpenShell

### Install Agent Hardener

### Credentials

### Verify the setup

## Quickstart: run on the example agent

## Run on your own agent

### The manifest

### What your agent needs

### Generate the benign suite

### Run the hardening loop

## Configuration

### Credentials and environment variables

### Choosing models

### Model providers

Agent Hardener works with any OpenAI-compatible endpoint. The three providers below were each tested with one
full hardening run (`synth-benign`, then `run --rounds 1`) against the bundled example agent, using Nemotron
models throughout.

| Provider | Base URL | Attack, analysis and judge model | Example agent's model | Attacks landed | Blocked after hardening | Benign requests OK |
|---|---|---|---|---|---|---|
| NVIDIA Build | `https://integrate.api.nvidia.com/v1` | `nvidia/nemotron-3-ultra-550b-a55b` (default) | `nvidia/nemotron-3.5-lightning-30b-a3b` (default) | 8 / 24 | 8 / 8 | 5 / 5 |
| NVIDIA Inference | `https://inference-api.nvidia.com/v1` | `nvidia/nvidia/nemotron-3-ultra` | `nvidia/nvidia/nemotron-3.5-super-vl` | 10 / 25 | 10 / 10 | 6 / 6 |
| OpenRouter | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-ultra-550b-a55b` | `nvidia/nemotron-3-ultra-550b-a55b` | 8 / 23 | 8 / 8 | 6 / 6 |

Results are from one run per provider (2026-11-26, agent-hardener `741b3ec`, OpenShell 0.1.4, garak 0.17.0).

Models are set in three places, and all three must point at the same provider:

| Model group | Covers | Set with |
|---|---|---|
| Attack | garak red-team model and detector, and the replay validator's detector | `GARAK_RED_TEAM_MODEL_URI`, `GARAK_RED_TEAM_MODEL_NAME`, `GARAK_DETECTOR_MODEL_URI`, `GARAK_DETECTOR_MODEL_NAME` |
| Analysis | defenders, benign suite generation and judging, and the guardrail judge inside the agent | `AGENT_HARDENER_BASE_URL`, `AGENT_HARDENER_MODEL` |
| Your agent | the agent's own LLM | `agent.env` in the manifest (for the example agent: `INFERENCE_BASE_URL`, `INFERENCE_MODEL`) |

Your shell's environment never reaches the sandbox, so the agent's own model must be set under `agent.env` in
the manifest.

`INFERENCE_API_KEY` holds the provider's key. When `NIM_API_KEY` is unset, Agent Hardener copies
`INFERENCE_API_KEY` into it for garak. If you do set `NIM_API_KEY`, give it the same value: a stale value wins
over the copy.

#### NVIDIA Build

The defaults already point at Build, so only the key is needed. Use a key that can call chat models; some Build
keys can list models but are refused on chat calls.

```bash
# .env
INFERENCE_API_KEY=nvapi-...
```

#### NVIDIA Inference

Model IDs on this endpoint carry a double `nvidia/nvidia/` prefix, and the key starts with `sk-`.

```bash
# .env
INFERENCE_API_KEY=sk-...
AGENT_HARDENER_BASE_URL=https://inference-api.nvidia.com/v1
AGENT_HARDENER_MODEL=nvidia/nvidia/nemotron-3-ultra
GARAK_RED_TEAM_MODEL_URI=https://inference-api.nvidia.com/v1/
GARAK_RED_TEAM_MODEL_NAME=nvidia/nvidia/nemotron-3-ultra
GARAK_DETECTOR_MODEL_URI=https://inference-api.nvidia.com/v1/
GARAK_DETECTOR_MODEL_NAME=nvidia/nvidia/nemotron-3-ultra
```

```yaml
# agent-hardener.yaml
agent:
  env:
    INFERENCE_BASE_URL: "https://inference-api.nvidia.com/v1"
    INFERENCE_MODEL: "nvidia/nvidia/nemotron-3.5-super-vl"
```

#### OpenRouter

Ultra is the only Nemotron model available on OpenRouter, so it serves both the Agent Hardener model groups and
the example agent.

```bash
# .env
INFERENCE_API_KEY=sk-or-...
AGENT_HARDENER_BASE_URL=https://openrouter.ai/api/v1
AGENT_HARDENER_MODEL=nvidia/nemotron-3-ultra-550b-a55b
GARAK_RED_TEAM_MODEL_URI=https://openrouter.ai/api/v1/
GARAK_RED_TEAM_MODEL_NAME=nvidia/nemotron-3-ultra-550b-a55b
GARAK_DETECTOR_MODEL_URI=https://openrouter.ai/api/v1/
GARAK_DETECTOR_MODEL_NAME=nvidia/nemotron-3-ultra-550b-a55b
```

```yaml
# agent-hardener.yaml
agent:
  env:
    INFERENCE_BASE_URL: "https://openrouter.ai/api/v1"
    INFERENCE_MODEL: "nvidia/nemotron-3-ultra-550b-a55b"
```

#### Known issues

- **OpenRouter ignores the flag that turns reasoning off**, so every call reasons. Chat, JSON mode and tool
  calls still work.
- **OpenRouter can return an empty attack plan.** About a quarter of its replies to garak's planning prompt came
  back empty, and the run then stops with `no_attacks_generated`. No setting fixes this; rerun.
- **Switching providers on one gateway:** the OpenShell provider is named after `agent.name`, so give each
  provider's agent a different name to keep their keys apart.
- **A credential placeholder in the conversation is refused.** If an attack makes the agent print its
  environment, OpenShell's key placeholder enters the conversation and the gateway refuses to forward later model
  calls that contain it (HTTP 403). This can happen on any provider. The run still completes.

### Garak overrides

## Reading the results

## Troubleshooting

## Development

### Repository layout

### Tests

## Contributions
