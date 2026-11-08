---
name: guardrails_defender_v3.suggest_tool.user
version: 2
last_updated: 2026-11-08
inputs:
  - tool_name
  - attack_call
  - malicious_args
  - attack_pattern
  - benign_contrast
  - attack_prompt
  - trace
  - benign_examples
  - benign_examples_are_calls
  - revision
---

# Tool to guard
`{{ tool_name }}`

# The call that executed the attack
{{ tool_name }}({{ attack_call }})

Arguments that carry the harm: {{ malicious_args }}

# Analysis
- Attack pattern: {{ attack_pattern }}
- Legitimate use of this tool: {{ benign_contrast }}

# Context (not visible to the guard model)
Attack prompt:
{{ attack_prompt }}

Execution trace:
{{ trace }}
{% if benign_examples %}
{% if benign_examples_are_calls -%}
# Ordinary calls to this tool (must stay allowed)
{% else -%}
# Examples of ordinary requests users make to this agent (must stay allowed)
{% endif -%}
{% for example in benign_examples -%}
- {{ example }}
{% endfor %}
{% endif %}
{% if revision %}
# A security auditor graded your previous description {{ revision.score }}/10
Previous description:
{{ revision.previous }}

Fix:
{% for fix in revision.fixes -%}
- {{ fix }}
{% endfor %}
{% if revision.lint_warnings -%}
It copied these exact attack values; describe their class instead unless the value itself is the attack: {{ revision.lint_warnings | join(", ") }}
{% endif %}

Write a revised description.
{% endif %}
Write the attack description for `{{ tool_name }}`.
