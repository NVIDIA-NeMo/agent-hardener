# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sourced by the cycle scripts. Picks one OpenAI-compatible endpoint for both the victim and
# Agent Hardener's own models from the INFERENCE_API_KEY in the operator dotenv (as the entrypoint
# wrote it), without printing or exporting it.
# AH_BASE_URL / AH_MODEL override the choice; AH_EGRESS_HOST is derived from the base URL.

ah_inference_key() {
  local operator_env="$1" key
  key="$(sed -n 's/^INFERENCE_API_KEY=//p' "${operator_env}")"
  [[ -n "${key}" ]] || { printf 'ah-room: no INFERENCE_API_KEY in %s\n' "${operator_env}" >&2; return 1; }
  printf '%s' "${key}"
}

# Creates or updates the NeMo Secret NAME in the default workspace from stdin, which is the only
# place the value passes through.
ah_put_secret() {
  local name="$1" action=create
  nemo secrets get "${name}" --workspace default </dev/null >/dev/null 2>&1 && action=update
  nemo secrets "${action}" "${name}" --from-file - --workspace default >/dev/null \
    || { printf 'ah-room: nemo secrets %s %s failed\n' "${action}" "${name}" >&2; return 1; }
}

# The key as the NeMo Secret the agent-hardener plugin reads first, so Helix jobs take it from the
# Secrets store.
ah_put_inference_secret() {
  local operator_env="$1" key
  key="$(ah_inference_key "${operator_env}")" || return 1
  printf '%s' "${key}" | ah_put_secret agent-hardener-inference-key
}

ah_select_models() {
  local operator_env="$1" key default_base default_model
  key="$(ah_inference_key "${operator_env}")" || return 1
  case "${key}" in
    nvapi-*)
      default_base=https://integrate.api.nvidia.com/v1
      default_model=nvidia/nemotron-3.5-lightning-30b-a3b
      ;;
    sk-or-*)
      default_base=https://openrouter.ai/api/v1
      default_model=deepseek/deepseek-v4.1-flash-1017
      ;;
    *)
      [[ -n "${AH_BASE_URL:-}" && -n "${AH_MODEL:-}" ]] || {
        printf 'ah-room: the key is neither nvapi- nor sk-or-; set AH_BASE_URL and AH_MODEL for its endpoint\n' >&2
        return 1
      }
      ;;
  esac
  AH_BASE_URL="${AH_BASE_URL:-${default_base}}"
  AH_MODEL="${AH_MODEL:-${default_model}}"
  AH_EGRESS_HOST="$(python3 -c 'import sys, urllib.parse; h = urllib.parse.urlsplit(sys.argv[1]).hostname; assert h, sys.argv[1]; print(h)' "${AH_BASE_URL}")"
  export AH_BASE_URL AH_MODEL AH_EGRESS_HOST
}
