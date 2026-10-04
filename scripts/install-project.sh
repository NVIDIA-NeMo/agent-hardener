#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "ERROR: unexpected argument: $1" >&2
    echo "Usage: scripts/install-project.sh" >&2
    exit 1
fi

fingerprint_file() {
    if [ -e "$1" ]; then
        git hash-object -- "$1"
    else
        printf '__missing__'
    fi
}

uv_lock_before="$(fingerprint_file uv.lock)"
requirements_before="$(fingerprint_file requirements.txt)"

if just bootstrap-deps; then
    :
else
    status="$?"
    cat >&2 <<'EOF'

Dependency setup failed while running 'just install'.

This step installs Python, syncs dependencies, and exports requirements.txt.
Fix the uv/python/package-index issue above and rerun 'just install'.
EOF
    exit "$status"
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    exit 0
fi

uv run pre-commit install --hook-type pre-commit --hook-type pre-push

changed=()
check_changed() {
    path="$1"
    before="$2"
    after="$(fingerprint_file "$path")"
    if [ "$before" != "$after" ] && [ -n "$(git status --porcelain -- "$path")" ]; then
        changed+=("$path")
    fi
}

check_changed "uv.lock" "$uv_lock_before"
check_changed "requirements.txt" "$requirements_before"

if [ "${#changed[@]}" -eq 0 ]; then
    exit 0
fi

printf '\njust install updated dependency files:\n'
for path in "${changed[@]}"; do
    printf '  - %s\n' "$path"
done
printf '\nYour repo is now dirty because generated dependency files changed.\n'
printf 'Commit them on a work branch (create one with '\''git switch -c <branch-name>'\''\n'
printf 'first if you are still on the default branch):\n'
printf '  git add uv.lock requirements.txt\n'
printf '  git commit -m "Update dependency lock files"\n'
