# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support modules for the garak attack-replay validator.

The validator's entry point + orchestration + monkeypatched I/O (``run``, the hit-validation loop, the
persistent garak worker, ``_replay_prompt``/``_fetch_issue_comments``/``_detector_for``) stay in
``agent_hardener.agents.validators.garak_attack_replay``; this package holds the pure, dependency-free
helpers it composes so that module stays focused.
"""
