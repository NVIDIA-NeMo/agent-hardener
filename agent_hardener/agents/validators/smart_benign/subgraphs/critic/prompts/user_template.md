---
name: critic.user
version: 2
last_updated: 2026-11-19
inputs:
  - target_name
  - tools_summary
  - tool_boundaries
  - blocked_patterns
  - out_of_scope
  - requests_table
---

# Target system: {{ target_name }}

## Available tools

{{ tools_summary }}

## Owner's boundary answers per tool

{{ tool_boundaries }}

## Blocked patterns per tool

{{ blocked_patterns }}

## Declared out-of-scope behaviors

{{ out_of_scope }}

## Generated requests to critique

{{ requests_table }}

# Task

Emit one `CriticDecision` for each row above, addressed by its 1-based index.
Return them inside a `CriticResult` JSON object. Match the format described
in the system prompt.
