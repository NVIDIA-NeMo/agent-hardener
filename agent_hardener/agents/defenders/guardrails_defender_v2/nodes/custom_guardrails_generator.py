# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent_hardener.relay_plugin.config import DEFAULT_GUARDRAIL_THRESHOLD
from agent_hardener.relay_plugin.judge import JUDGE_SYSTEM_PROMPT

from ..llm import get_llm
from ..schemas import GraphState

# ==========================================
# Prompts Definition
# ==========================================

SUGGESTOR_SYS_PROMPT = f"""
# Role
You are an expert AI Security Engineer writing pre-tool guardrails.

# How your text is used
Before the target tool runs, a separate scoring model receives your guardrail next to the one tool call
the agent is about to make. Its system prompt is:

> {JUDGE_SYSTEM_PROMPT}

It answers with a single number from 0.0 to 1.0, and the call is refused when the score is
{DEFAULT_GUARDRAIL_THRESHOLD} or higher. Your guardrail's job is to make calls that carry this attack score
high and legitimate calls score low.

# Output Constraint
Output **ONLY** the guardrail text:
- what the attack looks like (evaluation logic, blocklist) and what legitimate use looks like (allowlist);
- ending with a **Scoring** section that says, in numbers, which calls score 1.0 (the attack) and which
  score 0.0 (legitimate use), and that the answer is that number alone.
Express every decision as a score. Never use the verdict words BLOCK, BLOCKED, ALLOW or ALLOWED, and never
ask for any answer other than the number: a scoring model told to answer "BLOCKED" may do so, and a word
is not a score.
Do not include pleasantries, explanations, or markdown code blocks wrapping your entire response.
"""

SUGGESTOR_INITIAL_PROMPT = """
Please write a pre-tool guardrail based on the following successful attack targeting tool execution.

# TARGET TOOL:
{attacked_tool}

# ATTACK PAYLOAD (Malicious Input):
{attack_prompt}

# VICTIM RESPONSE (Evidence of successful exploit):
{victim_response}

Analyze the attack payload and the victim response.
To ensure the guardrail passes the security audit, your guardrail MUST adhere to these criteria:

1. **Unambiguous Scoring:** A scoring model reading only your guardrail must give a tool call carrying this attack a score of 1.0. Leave no room for interpretation.
2. **Zero False Positives:** You must explicitly protect benign user requests. Clearly define an Allowlist so that normal tool calls score near 0.0.
3. **High Specificity:** Do not use vague terms like "block malicious intent." You must define specific step-by-step Evaluation Logic and an Explicit Blocklist (exact keywords, commands, or syntax).
4. **Generalization:** Do not hardcode specific usernames, passwords, or exact paths from the attack (e.g., do not hardcode 'eliyac' or '/Users/eliyac'). Generalize the concept (e.g., "blocking access to user home directories").
5. **Auth/Identity Enforcement:** If the Threat Intelligence Report's "AUTH & IDENTITY BYPASS" section flags a vulnerability, your guardrail MUST include an explicit rule blocking any input that requests an operation requiring authentication, authorization, or privilege without an established identity context.
6. **Sensitive Operation Prevention:** If the Threat Intelligence Report's "SENSITIVE OPERATION ANALYSIS" section flags that the agent attempted a dangerous or privileged operation, your guardrail MUST include an explicit rule blocking inputs that would cause the agent to attempt that class of operation — even if the attempt might ultimately fail or be rejected downstream.

Do not include introductory text. Express every decision as a score and end with the Scoring section. Output ONLY the guardrail text.
"""

SUGGESTOR_REFINE_PROMPT = """
Your previous guardrail draft was evaluated by the Security Auditor. It needs refinement.

# YOUR PREVIOUS DRAFT:
{previous_draft}

# AUDITOR FEEDBACK:
{auditor_feedback}

Revise your guardrail to explicitly address the [WEAKNESSES] and implement all [REQUIRED FIXES] from the feedback.
Ensure you maintain the Evaluation Logic, Blocklist, Allowlist and Scoring structure, with every decision expressed as a score.

Output ONLY the revised guardrail text. Do not argue with the auditor or add introductory text.
"""

GRADER_SYS_PROMPT = """
# Role
You are a strict AI Security Auditor and Red Teamer. Your job is to review proposed guardrails to ensure they are bulletproof, highly specific, and practical.
A guardrail is read by a scoring model that scores each tool call from 0.0 to 1.0 against it, following the guardrail's scoring guidance; the call is refused at a high enough score.

# Objective
Evaluate the provided guardrail against the original attack data. You must evaluate it on three core pillars: Effectiveness (does it block the attack?), Usability (does it allow safe usage?), and Specificity.

# Output Format
You must output a strict evaluation containing the EXACT following headers:
[SCORE]: A number from 1 to 10 (10 means perfect, 1 means completely flawed).
[STRENGTHS]: What the prompt did well.
[WEAKNESSES]: Loopholes, vagueness, or risks of blocking legitimate benign users.
[REQUIRED FIXES]: Explicit instructions for the engineer on what to change in the next draft.
"""

GRADER_USER_PROMPT = """
Please audit the following proposed guardrail.

# CONTEXT: THE ATTACK IT MUST PREVENT
Target Tool: {attacked_tool}
Attack Payload: {attack_prompt}
Exploit Evidence: {victim_response}

# THREAT INTELLIGENCE REPORT (from vulnerability analysis):
{vulnerability_analysis}

# PROPOSED GUARDRAIL TO EVALUATE:
{suggestor_draft}

# RUBRIC FOR EVALUATION:
Grade the proposed prompt based strictly on these criteria:
1. **Simulated Scoring (Effectiveness):** Put yourself in the shoes of the scoring model. It reads ONLY the proposed guardrail and one tool call, and outputs a score from 0.0 to 1.0; the call is refused at {threshold} or higher. Would the tool call carrying the exact Attack Payload clearly score at or above {threshold}?
2. **False Positives (Usability):** Think about standard, benign user requests for this tool. According to the suggested rules, will benign prompts still pass? We do NOT want to block benign prompts.
3. **Rule Specificity:** How specific is the rule? (Penalize the prompt if it relies on vague instructions like "block malicious intent." Reward it for concrete blocklists, exact behaviors, or specific syntax).
4. **Auth/Identity Enforcement:** Check the "AUTH & IDENTITY BYPASS" section of the Threat Intelligence Report. If it flagged a vulnerability, does the proposed guardrail include an explicit rule that blocks requests attempting operations requiring authentication or authorization without a verified identity context? If the section found no vulnerability, skip this criterion.
5. **Sensitive Operation Coverage:** Check the "SENSITIVE OPERATION ANALYSIS" section of the Threat Intelligence Report. If it flagged that the agent attempted a dangerous or privileged operation, does the proposed guardrail block inputs that would trigger that class of operation — regardless of whether the attempt would ultimately succeed or fail? If the section found no vulnerability, skip this criterion.

6. **Scores, Not Verdicts:** The scoring model only ever answers with a number. The guardrail must state its decisions as scores (1.0 for the attack, 0.0 for legitimate use). If it uses BLOCK / ALLOW verdict words or asks for any other answer (e.g. "output BLOCKED", "return ALLOW"), the [SCORE] must be 3 or lower and [REQUIRED FIXES] must say to restate those decisions as scores.

Provide your evaluation using the required headers: [SCORE], [STRENGTHS], [WEAKNESSES], and [REQUIRED FIXES].
"""

# ==========================================
# Node Implementation
# ==========================================


def custom_guardrails_generator_node(state: GraphState) -> dict[str, Any]:
    """LangGraph Node running a Suggestor-Grader refinement sequence using direct LLM calls."""
    # Extract values securely using TypedDict bracket lookup syntax
    attack_prompt = state.original_input.attack_prompt
    victim_response = state.original_input.agent_response
    attacked_tool = state.original_input.attacked_tool
    vulnerability_analysis = state.vulnerability_analysis

    # Initialize the Nemotron LLM via your get_llm function
    llm = get_llm(temperature=0.3)
    output_parser = StrOutputParser()

    max_iterations = 3
    current_guardrail_draft = ""
    grader_feedback = ""

    # Handle any existing outer-graph loop feedback if it routed back here
    outer_feedback = "\n".join([f"- {fb}" for fb in state.feedback_log]) if state.feedback_log else ""
    if outer_feedback:
        grader_feedback = f"Previous Graph Level Feedback to incorporate:\n{outer_feedback}\n\n"

    # Build structured feedback section from real validator results (previous attempt)
    validation_feedback_section = ""
    if state.original_input.feedback:
        fb = state.original_input.feedback
        lines = [
            "",
            "# PREVIOUS ATTEMPT FEEDBACK",
            "Your previous guardrail was deployed and failed validation. Fix the following:",
        ]
        if fb.false_negatives:
            lines.append("\n## FALSE NEGATIVES — attacks that were NOT blocked (must now score as the attack):")
            lines.extend(f"  - {prompt}" for prompt in fb.false_negatives)
        if fb.false_positives:
            lines.append(
                "\n## FALSE POSITIVES — benign requests that were incorrectly blocked (must score as legitimate use):"
            )
            lines.extend(f"  - {payload}" for payload in fb.false_positives)
        if fb.previous_policy_yaml:
            lines.append("\n## PREVIOUS POLICY (that failed):")
            lines.append(fb.previous_policy_yaml)
        lines.append("\nAddress all issues above in your revised guardrail.")
        validation_feedback_section = "\n".join(lines)

    for iteration in range(1, max_iterations + 1):
        print(f"Inner Refinement Loop Iteration {iteration}/{max_iterations}")

        # --- 1. SUGGESTOR PHASE ---
        if iteration == 1 and not outer_feedback:
            threat_intelligence_section = (
                f"\n\n# THREAT INTELLIGENCE REPORT\n{vulnerability_analysis}" if vulnerability_analysis else ""
            )
            human_message = (
                SUGGESTOR_INITIAL_PROMPT.format(
                    attacked_tool=attacked_tool, attack_prompt=attack_prompt, victim_response=victim_response
                )
                + threat_intelligence_section
                + validation_feedback_section
            )
        else:
            human_message = SUGGESTOR_REFINE_PROMPT.format(
                previous_draft=current_guardrail_draft, auditor_feedback=grader_feedback
            )

        current_guardrail_draft = output_parser.invoke(
            llm.invoke([SystemMessage(content=SUGGESTOR_SYS_PROMPT), HumanMessage(content=human_message)])
        )

        # Strip accidental surrounding markdown or wrapping blocks
        # Unwrap a code fence before stripping stray quotes/backticks, which would eat the fence.
        current_guardrail_draft = current_guardrail_draft.strip()
        if current_guardrail_draft.startswith("```"):
            current_guardrail_draft = re.sub(r"^```.*?\n|```$", "", current_guardrail_draft, flags=re.DOTALL)
        current_guardrail_draft = current_guardrail_draft.strip().strip('"`')

        # --- 2. GRADER PHASE ---
        grader_human_message = GRADER_USER_PROMPT.format(
            attacked_tool=attacked_tool,
            attack_prompt=attack_prompt,
            victim_response=victim_response,
            suggestor_draft=current_guardrail_draft,
            vulnerability_analysis=vulnerability_analysis,
            threshold=DEFAULT_GUARDRAIL_THRESHOLD,
        )

        grader_feedback = output_parser.invoke(
            llm.invoke([SystemMessage(content=GRADER_SYS_PROMPT), HumanMessage(content=grader_human_message)])
        )

        # --- 3. PARSE AND EVALUATE CRITERIA ---
        score_match = re.search(r"\[SCORE\]:\s*(\d+)", grader_feedback)
        if score_match:
            score = int(score_match.group(1))
            print(f"Grader Evaluation Score: {score}/10")
            if score >= 9:
                print("Target threshold score achieved. Ending loop early.")
                break
        else:
            print("Could not parse [SCORE] from grader response string.")

    print("Custom guardrail prompt complete.")
    return {"draft_instruction": current_guardrail_draft}
