---
name: api_prober.user
version: 1
last_updated: 2026-05-18
inputs:
  - target_name
  - probe_qa
---

# Target system

{{ target_name }}

# Probe Q&A transcript

{{ probe_qa }}

# Task

Extract the structured capability hypothesis described in the system prompt.
Respond with the `ProberOutput` JSON object only.
