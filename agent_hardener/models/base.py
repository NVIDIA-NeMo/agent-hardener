# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared strict model base and the role/kind literal aliases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

AgentRole = Literal["attacker", "defender", "victim", "validator"]
ValidatorKind = Literal["attack", "benign"]
VictimControlType = Literal["file", "openshell"]


class AgentHardenerModel(BaseModel):
    """Shared strict model base."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
