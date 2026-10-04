# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local HTTP synth service — drives the interrupt-based benign-suite synth for a UI (interview + review)."""

from __future__ import annotations

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
