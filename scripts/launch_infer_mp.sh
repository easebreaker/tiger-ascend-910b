#!/usr/bin/env bash
# Launch N independent single-card processes (NO HCCL / torchrun).
# Used for inference replicas & overalloc scheduling experiments.
#
# Usage:
#   DEVICES=4,5,6,7 ROLE=replicas bash scripts/launch_infer_mp.sh --strategy replicas ...
#   DEVICES=4,5 ROLE=overalloc bash scripts/launch_infer_mp.sh --strategy overalloc ...
#
# Env:
#   DEVICES   comma-separated physical NPU ids (required)
#   ROLE      replicas | overalloc (default replicas)
#   OUT_DIR   shared output directory (required via --out_dir in args too)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICES="${DEVICES:?set DEVICES=0,1,2,3}"
ROLE="${ROLE:-replicas}"
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
N="${#DEV_ARR[@]}"
if [[ "$N" -lt 1 ]]; then
  echo "DEVICES empty" >&2
  exit 1
fi

# Remaining args passed to sched_bench.py (must include --strategy/--out_dir/...)
ARGS=("$@")

PIDS=()
cleanup() {
  local p
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "[launch-mp] role=${ROLE} n=${N} devices=${DEVICES}"
echo "[launch-mp] args: ${ARGS[*]}"

i=0
for dev in "${DEV_ARR[@]}"; do
  export ASCEND_RT_VISIBLE_DEVICES="$dev"
  # Each process only sees one card → logical npu:0
  unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE || true
  LOG="${OUT_DIR:-/tmp}/mp_rank${i}_dev${dev}.log"
  if [[ -n "${OUT_DIR:-}" ]]; then
    mkdir -p "$OUT_DIR"
    LOG="$OUT_DIR/mp_rank${i}_dev${dev}.log"
  fi
  # Tag encodes rank for later aggregation
  python -u scripts/sched_bench.py \
    "${ARGS[@]}" \
    --mp_rank "$i" \
    --mp_world_size "$N" \
    --mp_role "$ROLE" \
    --tag "mp${N}_r${i}" \
    >"$LOG" 2>&1 &
  PIDS+=("$!")
  echo "[launch-mp] started rank=$i device=$dev pid=${PIDS[-1]} log=$LOG"
  i=$((i + 1))
done

ec=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    ec=1
  fi
done
trap - EXIT INT TERM
if [[ "$ec" -ne 0 ]]; then
  echo "[launch-mp] one or more ranks failed; check logs under ${OUT_DIR:-.}" >&2
  exit "$ec"
fi
echo "[launch-mp] all ranks finished OK"
