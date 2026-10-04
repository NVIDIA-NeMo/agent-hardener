# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refusal judge — LangChain ``with_structured_output``, no agent framework."""

from __future__ import annotations

from .client import judge_one
from .schema import JudgeVerdict

__all__ = ["JudgeVerdict", "judge_one"]
