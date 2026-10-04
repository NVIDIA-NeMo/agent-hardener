---
name: github_analyzer.user
version: 1
last_updated: 2026-05-17
inputs:
  - target_name
  - tool_summary
  - readme
---

# Target system: {{ target_name }}

## Tools already extracted from NAT workflow YAMLs (authoritative)

{{ tool_summary }}

## Repository README

{{ readme }}

# Task

Emit a `SummarizerOutput` JSON object — `system_role`, `personas`,
`out_of_scope` — based on the README content. Do not modify the tool list.
