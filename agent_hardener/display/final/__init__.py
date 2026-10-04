# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final report renderers for Agent Hardener runs."""

from agent_hardener.display.final.render_plain import render_final_run_log
from agent_hardener.display.final.render_rich import (
    build_summary_renderable,
    print_final_summary,
)
from agent_hardener.display.renderers.garak_attacker import attack_transcript_for_attacks, attacker_table_for_attacks
from agent_hardener.display.renderers.openshell_policy import render_defender_detail

__all__ = [
    "attack_transcript_for_attacks",
    "attacker_table_for_attacks",
    "build_summary_renderable",
    "print_final_summary",
    "render_defender_detail",
    "render_final_run_log",
]
