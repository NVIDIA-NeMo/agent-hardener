#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Prevent git from operating on a parent repo if run from a subdirectory.
export GIT_CEILING_DIRECTORIES="$(cd .. && pwd)"

if [ -d ".git" ]; then
    echo "Git already initialized, skipping."
else
    git init --initial-branch=main
    if [ ! -d ".git" ]; then
        echo "ERROR: git init failed - .git directory not found. Aborting." >&2
        exit 1
    fi
    git add .
    git commit -m "Initial commit"
fi
