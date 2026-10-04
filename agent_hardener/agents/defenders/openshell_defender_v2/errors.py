# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception vocabulary for the deterministic pipeline.

``run()`` never raises these (or anything else) out — ``openshell_defender_v2_agent.py`` catches
them at the top level and converts every one into ``DefenderOutput(ok=False, error_message=...)``.
They exist so each stage can fail loudly and specifically during development/testing without
every caller needing its own ad-hoc error string.
"""

from __future__ import annotations


class DefenderError(Exception):
    """Base class for all errors raised inside the openshell_defender_v2 pipeline."""


class PolicyParseError(DefenderError):
    """The current policy YAML could not be parsed into ``policy.schema.Policy``."""


class ExtractionError(DefenderError):
    """Attack or benign request tuples could not be recovered from the given evidence."""


class NoFeasibleNodeError(DefenderError):
    """Every feasibility predicate in ``NODE_PRIORITY`` returned ``None``."""


class LintViolationError(DefenderError):
    """``policy.lint.assert_contraction`` found the candidate patch widened access."""
