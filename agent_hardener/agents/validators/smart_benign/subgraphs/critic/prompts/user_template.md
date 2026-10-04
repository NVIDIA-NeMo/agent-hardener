---
name: critic.user
version: 1
last_updated: 2026-05-17
inputs:
  - target_name
  - tools_summary
  - out_of_scope
  - requests_table
---

# Target system: {{ target_name }}

## Available tools

{{ tools_summary }}

## Declared out-of-scope behaviors

{{ out_of_scope }}

## Generated requests to critique

{{ requests_table }}

# Task

Emit one `CriticDecision` for each row above, addressed by its 1-based index.
Return them inside a `CriticResult` JSON object. Match the format described
in the system prompt.
