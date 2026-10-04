# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import side effect: registers every mitigation node into ``base.NODE_REGISTRY``."""

from __future__ import annotations

from . import add_deny_rule, remove_endpoint
from .base import get_node, register

for _module in (remove_endpoint, add_deny_rule):
    register(_module.NODE)

__all__ = ["get_node", "register"]
