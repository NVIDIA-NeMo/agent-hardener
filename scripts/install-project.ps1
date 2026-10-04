# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"

if ($args.Count -ne 0) {
    Write-Error "Unexpected argument count. Usage: scripts/install-project.ps1"
    exit 1
}

$depFiles = @("uv.lock", "requirements.txt")

function Get-FileFingerprint($Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    } else {
        "__missing__"
    }
}

$before = @{}
foreach ($path in $depFiles) {
    $before[$path] = Get-FileFingerprint $path
}

& just bootstrap-deps
if ($LASTEXITCODE -ne 0) {
    $status = $LASTEXITCODE
    [Console]::Error.WriteLine("")
    [Console]::Error.WriteLine("Dependency setup failed while running 'just install'.")
    [Console]::Error.WriteLine("")
    [Console]::Error.WriteLine("This step installs Python, syncs dependencies, and exports requirements.txt.")
    [Console]::Error.WriteLine("Fix the uv/python/package-index issue above and rerun 'just install'.")
    exit $status
}

& git rev-parse --is-inside-work-tree >$null 2>$null
if ($LASTEXITCODE -ne 0) {
    exit 0
}

& uv run pre-commit install --hook-type pre-commit --hook-type pre-push
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$changed = @()
foreach ($path in $depFiles) {
    $after = Get-FileFingerprint $path
    $status = @(& git status --porcelain -- $path)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($before[$path] -ne $after -and $status.Count -gt 0) {
        $changed += $path
    }
}

if ($changed.Count -eq 0) {
    exit 0
}

Write-Host ""
Write-Host "just install updated dependency files:"
foreach ($path in $changed) {
    Write-Host "  - $path"
}
Write-Host ""
Write-Host "Your repo is now dirty because generated dependency files changed."
Write-Host "Commit them on a work branch (create one with 'git switch -c <branch-name>'"
Write-Host "first if you are still on the default branch):"
Write-Host "  git add uv.lock requirements.txt"
Write-Host '  git commit -m "Update dependency lock files"'
