# Smart Benign Validator

## Working on this feature
- Commit messages: one short imperative line — no "Co-Authored-By", no mention of Claude

## What it does

The smart benign validator answers one question: *does this victim agent over-block legitimate traffic?*

It works in two stages:

1. **Synth DAG** — infers what the victim can do (tools, personas, out-of-scope behaviour) from three optional input sources, then generates a tiered suite of benign request payloads.
2. **Replay + judge** — sends those payloads to the live victim endpoint and classifies each response as complied or refused via an LLM judge.

A finding is raised for every request the victim refuses. Zero findings means the victim does not over-block for this profile.

## Architecture

```
START → [nl_parser, github_analyzer, api_prober]   ← parallel fan-in
              ↓
        profile_synthesizer   (merge + LLM normalize)
              ↓
        gap_detector           (rules + LLM boundary pass when interactive)
              ↓ if gaps + interactive + budget
        interviewer            (LLM picks question → stdin → loop back)
              ↓
        request_generator → critic → profile_writer → END
              ↓
        replay_judge           (HTTP replay + LLM verdict per row)
```

Nine subgraphs. Each lives under `subgraphs/<name>/` with its own `state.py`, `adapters.py`, `agent.py`, and `nodes/`.

## Running it

### Pre-flight only (synth DAG, no replay)

```bash
agent-hardener synth-benign -c examples/vulnerable_demo_e2e.yaml
agent-hardener synth-benign -c examples/vulnerable_demo_e2e.yaml --no-interactive   # CI / no TTY
agent-hardener synth-benign -c examples/vulnerable_demo_e2e.yaml --validator my-name  # if config has multiple
```

Writes `profile.json`, `requests.csv`, and per-source notes under
`<storage.root_dir>/benign_profiles/<target>/`.

At the end of an interactive run the CLI prints newly collected Q&A pairs:

```
Add these to your YAML's `interview_answers:` to skip re-asking next run:
  - ["What commands are typical?", "ls, grep, find"]
```

Copy those into the YAML so the next run starts from a known profile without the interview loop.

### Full run (supplied-suite seed + in-loop replay + judge)

Runs via the orchestrator — add the validator to `benign_validators:` in a session YAML and call `just run`. `run` is a pure **consumer**: it does not synthesize. Its pre-flight step only **seeds a supplied suite** (from `--benign-suite <csv>` / `benign_suite_path`) into the target's artifact dir via `validator.synthesize(source_suite=...)`; with no suite configured it skips benign validation. Generate the suite out of band with `agent-hardener synth-benign` (self-contained: brings the sandbox up, synthesizes, tears down) or the `serve` HITL. The per-attempt `validator.run()` is **replay-only**: it loads the seeded `requests.csv` and hard-fails with guidance if no suite exists — it never runs the synth DAG inside the loop.

## Session YAML config block

```yaml
benign_validators:
  - name: smart-benign-validator
    implementation: agent_hardener.agents.validators.smart_benign:run
    timeout_seconds: 1200
    config:
      # Synth inputs — any combination; the DAG branches on what is present.
      description: |
        NAT-style research assistant. Has bash_executor and python_executor tools.
      # github_url: https://github.com/org/repo
      api_endpoint: http://localhost:8000/v1/chat/completions
      interview_answers: []         # pre-seed to skip re-asking

      # Per-source skip flags — set true to bypass a source entirely.
      # skip_nl_parser: true
      # skip_github_analysis: true
      # skip_api_probe: true

      # Replay + judge.
      replay_url: http://localhost:8000/v1/chat/completions
      replay_mode: openai_chat      # or "json"
      model: smart_benign_validator
      confidence_cutoff: 0.5
      max_interview_questions: 10   # default from defaults.yaml
```

See `examples/vulnerable_demo_e2e.yaml` for a complete working example.

## Ingestion sources and skip flags

| Source | Activated by | Skip flag | Skip mechanism |
|--------|-------------|-----------|---------------|
| `nl_parser` | `description` present | `skip_nl_parser: true` | passes `description=None` → node no-ops |
| `github_analyzer` | `github_url` present | `skip_github_analysis: true` | passes `repo_url=""` → node no-ops |
| `api_prober` | always (falls back to `target.base_url`) | `skip_api_probe: true` | passes `max_probes=0` → node no-ops |

All three nodes remain in the LangGraph graph regardless — they just return an empty delta immediately when skipped. The graph topology never changes.

## Key invariants — do not break

**Source merge priority** (in `subgraphs/profile_synthesizer/nodes/merge.py`):

```
github (3) > nl (2) > api_probe (1) > interview (0)
```

Structural sources outrank inferential ones. When the same tool name appears in multiple sources the highest-priority source wins; `example_inputs` are unioned across all sources.

**`gap_detector` runs rules first, then an interactive-only LLM boundary pass.** The deterministic `detect_gaps` rules check that every tool has a non-empty description, at least one example input, and a confidence above threshold, and that the profile has `system_role`, `personas`, and `out_of_scope` set — keep these LLM-free. The `analyze_boundaries` node then makes one LLM call per tool to judge whether the benign/attack line is clear and, if not, emits a `boundary[<tool>]:` gap into the interview. It no-ops (no LLM call) when `interactive=False`, so non-TTY runs stay rules-only. The `boundary[<tool>]:` prefix is a stable per-tool key: a later round skips any tool that already has an answered boundary gap, which is what keeps the interview loop from re-asking when the LLM rephrases the question.

**Interview loop exits when** any of these is true: no gaps remain, `interactive=False`, or `interview_questions_asked >= max_interview_questions`. One question is asked per interviewer invocation (LLM picks the single best one).

**`api_endpoint` always resolves** — `validator.py._build_inputs` falls back to `target.base_url` if `api_endpoint` is not set in the agent config. This means `api_prober` will always have a non-empty URL unless you set `skip_api_probe: true`.

**Borderline generation is on by default but interactive-only.** `borderline_per_tool: 2` in `defaults.yaml`, yet `request_generator`'s entry adapter forces it to `0` when `interactive=False`. Borderline rows probe a benign/attack line that is only confirmed via the boundary interview, so non-interactive runs (orchestrator pre-flight, CI) generate zero of them and replay an unconfirmed boundary against nobody. `negative_control_per_tool` stays `0`.

## Defaults knobs (`defaults.yaml`)

| Key | Default | Effect |
|-----|---------|--------|
| `synth_llm.model` | `nvidia/nemotron-3-super-120b-a12b` | All synthesis LLM calls |
| `judge_llm.model` | `nvidia/nemotron-3-super-120b-a12b` | Judge verdicts |
| `generation.requests_per_tool` | `3` | Benign rows generated per tool |
| `generation.borderline_per_tool` | `2` | Borderline rows per tool — interactive-only (forced to 0 with no TTY); see above |
| `generation.negative_control_per_tool` | `0` | Intentionally zero |
| `pipeline.max_interview_questions` | `10` | Hard ceiling on interview loop |
| `pipeline.api_probe_budget` | `5` | Max meta-probes sent to victim |
| `replay.confidence_cutoff` | `0.5` | Judge threshold for "refused" verdict |

Override any of these per-validator in the session YAML `config` block. Never duplicate defaults in code — this file is the single source of truth for fallback values.

## Langfuse tracing

Opt-in. Set these env vars to enable:

```
LANGFUSE_ENABLED=1
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...          # optional
```

Missing any one silently disables tracing. Optionally, `agent-hardener synth-benign --upload-dataset` exports the generated requests to a Langfuse dataset.

## File map

```
smart_benign/
  validator.py          — synthesize() (pre-flight, cache-aware), replay-only run(), _build_inputs(), _build_config()
  graph.py              — LangGraph parent graph, routing logic
  state.py              — SynthState, SynthInputs
  models.py             — ToolSpec, VictimCapabilityProfile, GeneratedRequest
  defaults.yaml         — all default knobs
  wizard.py             — agent-hardener synth-benign command (synth-only)
  tracing.py            — Langfuse opt-in
  subgraphs/
    nl_parser/          — description → ToolSpec list (1 LLM call)
    github_analyzer/    — git clone + YAML scan + README summarize (1 LLM call)
    api_prober/         — meta-probes to live endpoint + extract (1 LLM call)
    profile_synthesizer/— deterministic merge + LLM normalize
    gap_detector/       — rules gaps + interactive-only LLM boundary pass
    interviewer/        — LLM composes question + stdin ask
    request_generator/  — per-tool tiered request generation
    critic/             — filter + deduplicate requests
    profile_writer/     — write profile.json + requests.csv to disk
  replay_judge/         — HTTP replay + LLM judge verdict per row
```
