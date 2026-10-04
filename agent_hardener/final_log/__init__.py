# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public data/parse contract for the final log, shared with the display layer.

The parsing/aggregation helpers live in :mod:`agent_hardener.final_log.collect`,
:mod:`agent_hardener.final_log.garak`, and :mod:`agent_hardener.final_log.helpers` with underscore-prefixed
(module-private) names. This package root re-exports the subset that the presentation layer
(:mod:`agent_hardener.display.final`) legitimately shares, under public names — so display imports a stable
facade instead of reaching into private internals. Rendering itself lives entirely in ``display``;
``final_log`` never imports ``display``.
"""

from __future__ import annotations

from agent_hardener.final_log.collect import (
    _artifact_paths_from_reports as artifact_paths_from_reports,
)
from agent_hardener.final_log.collect import (
    _collect_attack_stats as collect_attack_stats,
)
from agent_hardener.final_log.collect import (
    _collect_attempt_transcripts as collect_attempt_transcripts,
)
from agent_hardener.final_log.collect import (
    _collect_final_log_data as collect_final_log_data,
)
from agent_hardener.final_log.garak import (
    _is_hitlog_path as is_hitlog_path,
)
from agent_hardener.final_log.garak import (
    _merge_garak_artifacts as merge_garak_artifacts,
)
from agent_hardener.final_log.helpers import (
    _counter_keys as counter_keys,
)
from agent_hardener.final_log.helpers import (
    _counter_summary as counter_summary,
)
from agent_hardener.final_log.helpers import (
    _defenders as defenders,
)
from agent_hardener.final_log.helpers import (
    _fmt_bool as fmt_bool,
)
from agent_hardener.final_log.helpers import (
    _fmt_value as fmt_value,
)
from agent_hardener.final_log.helpers import (
    _iterations as iterations,
)
from agent_hardener.final_log.helpers import (
    _load_json as load_json,
)
from agent_hardener.final_log.helpers import (
    _path_mtime as path_mtime,
)
from agent_hardener.final_log.helpers import (
    _policy_patches as policy_patches,
)
from agent_hardener.final_log.helpers import (
    _round_dir_from_report as round_dir_from_report,
)
from agent_hardener.final_log.helpers import (
    _scan_has_data as scan_has_data,
)
from agent_hardener.final_log.helpers import (
    _short_text as short_text,
)
from agent_hardener.final_log.models import (
    MAX_EXAMPLES_PER_SCANNER,
    MAX_POLICY_ITEMS,
    SCANNER_LABELS,
    SCANNERS,
    SCANNERS_WITH_OTHER,
    AttemptTranscript,
    FinalLogData,
    HitExample,
    ScanStats,
)

__all__ = [
    "MAX_EXAMPLES_PER_SCANNER",
    "MAX_POLICY_ITEMS",
    "SCANNERS",
    "SCANNERS_WITH_OTHER",
    "SCANNER_LABELS",
    "AttemptTranscript",
    "FinalLogData",
    "HitExample",
    "ScanStats",
    "artifact_paths_from_reports",
    "collect_attack_stats",
    "collect_attempt_transcripts",
    "collect_final_log_data",
    "counter_keys",
    "counter_summary",
    "defenders",
    "fmt_bool",
    "fmt_value",
    "is_hitlog_path",
    "iterations",
    "load_json",
    "merge_garak_artifacts",
    "path_mtime",
    "policy_patches",
    "round_dir_from_report",
    "scan_has_data",
    "short_text",
]
