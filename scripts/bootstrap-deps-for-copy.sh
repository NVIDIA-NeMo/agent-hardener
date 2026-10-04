#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "ERROR: unexpected argument: $1" >&2
    echo "Usage: scripts/bootstrap-deps-for-copy.sh" >&2
    exit 1
fi

if uvx --from rust-just just bootstrap-deps; then
    exit 0
else
    status="$?"
fi

cat >&2 <<'EOF'

Dependency bootstrap failed while preparing the initial commit.

This step generates uv.lock and requirements.txt before git-init so the
initial commit and first push include dependency artifacts.

Fix the uv/python/package-index issue and rerun copier copy.
EOF
exit "$status"
