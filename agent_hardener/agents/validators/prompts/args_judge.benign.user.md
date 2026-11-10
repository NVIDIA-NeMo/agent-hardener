---
name: args_judge.benign.user
version: 1
last_updated: 2026-11-10
inputs:
  - prompt
  - tool
  - arguments
---

User request:
{{ prompt }}

Tool called: {{ tool }}
Arguments: {{ arguments }}
