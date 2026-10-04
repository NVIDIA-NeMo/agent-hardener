# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"

$remoteUrl = $args[0]

if ([string]::IsNullOrEmpty($remoteUrl)) {
    Write-Error "No remote URL provided. Usage: scripts/configure-git-origin.ps1 <url>"
    exit 1
}

if ($args.Count -ne 1) {
    Write-Error "Unexpected argument count. Usage: scripts/configure-git-origin.ps1 <url>"
    exit 1
}

# Prevent git from operating on a parent repo if run from a subdirectory.
$env:GIT_CEILING_DIRECTORIES = (Resolve-Path "..").Path

if (-not (Test-Path ".git")) {
    Write-Error "Not a git repository. Run 'just git-init' first."
    exit 1
}

$exists = git remote get-url origin 2>$null
if ($exists) {
    Write-Host "Remote 'origin' already configured, skipping."
} else {
    git remote add origin $remoteUrl
    git push --set-upstream origin main
}
