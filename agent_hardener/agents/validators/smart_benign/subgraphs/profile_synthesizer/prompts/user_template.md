---
name: profile_synthesizer.user
version: 3
last_updated: 2026-05-24
inputs:
  - target_name
  - tool_summary
  - tool_gaps
  - source_notes
  - interview_qa
---

# Target system: {{ target_name }}

## Merged tools (authoritative — do not modify)

{{ tool_summary }}

## Tool-level gaps (fields currently empty)

{{ tool_gaps }}

## Per-source notes

{{ source_notes }}

## User clarifications from interview

Each entry below may begin with `Gap:` — the exact gap string this question
was generated to fill. Use that label to map the answer directly to the right
tool or profile field.

{{ interview_qa }}

# Task

Produce the `NormalizedProfileGlobals` JSON object. Fill `system_role`,
`personas`, and `out_of_scope` from the inputs. For any tool listed in
"Tool-level gaps", check whether the interview Q&A contains an answer that
fills it (look for a `Gap:` label matching that tool's gap string) and emit a
`ToolPatch` for that tool. Do not modify the tool list beyond what the
interview answers explicitly provide.
