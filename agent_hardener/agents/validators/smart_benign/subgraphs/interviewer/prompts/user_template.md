---
name: interviewer.user
version: 6
last_updated: 2026-06-14
inputs:
  - gaps
  - prior_qa
  - remaining_budget
  - system_role
  - tool_summaries
  - out_of_scope
  - known_personas
---

## What we already know about this agent

- Role: {{ system_role if system_role else "(not yet known)" }}

Tools currently in the profile:
{% if tool_summaries %}
{% for tool in tool_summaries %}
- {{ tool }}
{% endfor %}
{% else %}
(none yet)
{% endif %}

Declared out-of-scope behaviours:
{% if out_of_scope %}
{% for item in out_of_scope %}
- {{ item }}
{% endfor %}
{% else %}
(none yet)
{% endif %}

Known user personas:
{% if known_personas %}
{% for persona in known_personas %}
- {{ persona }}
{% endfor %}
{% else %}
(none yet)
{% endif %}

## Current gaps

{% if gaps %}
{% for gap in gaps %}
- {{ gap }}
{% endfor %}
{% else %}
(none)
{% endif %}

## Conversation so far

{% if prior_qa %}
{% for gap, q, a in prior_qa %}
{% if gap %}Gap: {{ gap }}
{% endif %}Q: {{ q }}
A: {{ a }}

{% endfor %}
{% else %}
(none yet)
{% endif %}

## Budget

Generate at most **{{ remaining_budget }}** questions.

## Task

Produce an `InterviewBatch` JSON object with a `questions` array — one
`InterviewerQuestion` per gap (up to the budget), each with `question`,
`options` (exactly 3, one `recommended: true`), skipping gaps already covered
by the conversation. Ground every question and every option in the known agent
context above — never offer generic placeholder answers when the role and tools
already tell you the agent's domain. Return `{"questions": []}` if all gaps are
covered.
