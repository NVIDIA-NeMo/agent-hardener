#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

remote_url="${1:-}"

if [ -z "$remote_url" ]; then
    echo "ERROR: No remote URL provided. Usage: scripts/configure-git-origin.sh <url>" >&2
    exit 1
fi

shift
if [ "$#" -ne 0 ]; then
    echo "ERROR: unexpected argument: $1" >&2
    echo "Usage: scripts/configure-git-origin.sh <url>" >&2
    exit 1
fi

# Prevent git from operating on a parent repo if run from a subdirectory.
export GIT_CEILING_DIRECTORIES="$(cd .. && pwd)"

if [ ! -d ".git" ]; then
    echo "ERROR: Not a git repository. Run 'just git-init' first." >&2
    exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' already configured, skipping."
else
    git remote add origin "$remote_url"
    git push --set-upstream origin main
fi
