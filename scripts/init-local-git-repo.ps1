# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"

# Prevent git from operating on a parent repo if run from a subdirectory.
$env:GIT_CEILING_DIRECTORIES = (Resolve-Path "..").Path

if (Test-Path ".git") {
    Write-Host "Git already initialized, skipping."
} else {
    git init --initial-branch=main
    if (-not (Test-Path ".git")) {
        Write-Error "git init failed - .git directory not found. Aborting."
        exit 1
    }
    git add .
    git commit -m "Initial commit"
}
