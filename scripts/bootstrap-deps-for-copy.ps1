# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$ErrorActionPreference = "Stop"

if ($args.Count -ne 0) {
    Write-Error "Unexpected argument count. Usage: scripts/bootstrap-deps-for-copy.ps1"
    exit 1
}

try {
    & uvx --from rust-just just bootstrap-deps
    $status = $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine($PSItem.Exception.Message)
    $status = 1
}
if ($status -eq 0) {
    exit 0
}

[Console]::Error.WriteLine("")
[Console]::Error.WriteLine("Dependency bootstrap failed while preparing the initial commit.")
[Console]::Error.WriteLine("")
[Console]::Error.WriteLine("This step generates uv.lock and requirements.txt before git-init so the")
[Console]::Error.WriteLine("initial commit and first push include dependency artifacts.")
[Console]::Error.WriteLine("")
[Console]::Error.WriteLine("Fix the uv/python/package-index issue and rerun copier copy.")
exit $status
