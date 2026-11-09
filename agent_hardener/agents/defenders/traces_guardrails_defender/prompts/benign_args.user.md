---
name: traces_guardrails_defender.benign_args.user
version: 1
last_updated: 2026-10-11
inputs:
  - tool_name
  - arg_shape
  - requests
---

# Tool
`{{ tool_name }}` with arguments {{ arg_shape }} (argument name: type)

# Requests
{% for request in requests -%}
- {{ request }}
{% endfor %}
