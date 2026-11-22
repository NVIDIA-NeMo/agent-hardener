---
name: observed_tool.user
version: 1
last_updated: 2026-11-22
inputs:
  - tool
  - system_role
  - description
  - calls
---

# Agent

{{ system_role }}

# Tool

`{{ tool }}`

# Extracted description

{{ description }}

# Observed calls

{{ calls }}

# Task

Refine the description and add example requests. Respond with the `ToolRefinement` JSON object only.
