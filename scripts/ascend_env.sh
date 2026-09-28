#!/usr/bin/env bash
# Shared Ascend env for TIGER 910B launchers / probes.
#
# 修法2（关键）:
#   source toolkit set_env.sh 之后，toolkit 的 lib64 常排在 LD_LIBRARY_PATH 最前，
#   会盖住 driver 里的 libascend_hal.so 等，导致 zeros(1).npu() / set_device Abort。
#   本脚本在 set_env 之后：从 PATH 中摘掉 driver 目录，再强制把
#     driver/lib64/driver → common → lib64
#   插到最前面（不是 toolkit lib64）。
#
# Usage (from other scripts):
#   source "$(dirname "$0")/ascend_env.sh"
#
# shellcheck shell=bash

_tiger_ascend_source_set_env() {
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/script/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/script/set_env.sh
  elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
  else
    echo "[ascend-env] WARNING: no set_env.sh found under /usr/local/Ascend" >&2
  fi
}

_tiger_ascend_force_driver_libs_first() {
  # Candidate dirs (exist only). Order = final prepend order.
  local -a drivers=()
  local d
  for d in \
    /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64/common \
    /usr/local/Ascend/driver/lib64; do
    if [[ -d "$d" ]]; then
      drivers+=("$d")
    fi
  done

  if [[ ${#drivers[@]} -eq 0 ]]; then
    echo "[ascend-env] WARNING: no /usr/local/Ascend/driver/lib64* dirs found" >&2
    return 0
  fi

  # Rebuild LD_LIBRARY_PATH without these driver entries (wherever they sat).
  local new_path=""
  local part
  local skip
  local dd
  # shellcheck disable=SC2086
  IFS=':' read -r -a _ld_parts <<< "${LD_LIBRARY_PATH:-}"
  for part in "${_ld_parts[@]}"; do
    [[ -z "$part" ]] && continue
    skip=0
    for dd in "${drivers[@]}"; do
      if [[ "$part" == "$dd" ]]; then
        skip=1
        break
      fi
    done
    if [[ "$skip" -eq 0 ]]; then
      if [[ -z "$new_path" ]]; then
        new_path="$part"
      else
        new_path="$new_path:$part"
      fi
    fi
  done

  local prefix=""
  for d in "${drivers[@]}"; do
    if [[ -z "$prefix" ]]; then
      prefix="$d"
    else
      prefix="$prefix:$d"
    fi
  done

  if [[ -n "$new_path" ]]; then
    export LD_LIBRARY_PATH="$prefix:$new_path"
  else
    export LD_LIBRARY_PATH="$prefix"
  fi
}

_tiger_ascend_verify_ld_head() {
  local head
  head="$(echo "${LD_LIBRARY_PATH:-}" | cut -d: -f1-4)"
  echo "[ascend-env] LD_LIBRARY_PATH head=${head}"
  local first
  first="$(echo "${LD_LIBRARY_PATH:-}" | cut -d: -f1)"
  if [[ "$first" == *"/ascend-toolkit/"* ]] || [[ "$first" == *"/cann/"* && "$first" != *"/driver/"* ]]; then
    echo "[ascend-env] ERROR: toolkit/cann still first in LD_LIBRARY_PATH — driver prepend failed" >&2
    echo "[ascend-env] first=$first" >&2
    return 1
  fi
  if [[ "$first" != *"/driver/"* ]]; then
    echo "[ascend-env] WARNING: first entry is not under driver/: $first" >&2
  else
    echo "[ascend-env] OK: driver lib64 is ahead of toolkit"
  fi
  return 0
}

# --- main when sourced ---
_tiger_ascend_source_set_env
_tiger_ascend_force_driver_libs_first
_tiger_ascend_verify_ld_head || true

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"
# Do NOT default VISIBLE_DEVICES here — leave to caller.
