#!/usr/bin/env bash
# Inference-first multi-chip scheduling matrix (TIGER / Ascend 910B).
#
# Multi-card uses independent processes (NO torchrun / NO HCCL) to avoid
# hcclCommInitRootInfoConfig error code 1 on many single-node boxes.
#
#   export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#   bash scripts/run_sched_bench.sh
#   bash scripts/run_sched_bench.sh --quick
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/ascend_env.sh"

REQUESTS="${REQUESTS:-64}"
WARMUP="${WARMUP_REQUESTS:-5}"
DATA="${DATA_DIR:-data/amazon_beauty/subset_512u}"
OUT="${OUT_DIR:-artifacts/sched_bench/$(date +%Y%m%d_%H%M%S)}"
BATCH="${BATCH_SIZE:-8}"
BEAM="${BEAM_SIZE:-10}"
INFER_OP="${INFER_OP:-generate}"
CKPT="${CKPT:-}"
DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
TRAIN_PROFILE="${TRAIN_PROFILE:-0}"

if [[ "${1:-}" == "--quick" ]]; then
  REQUESTS=24
  WARMUP=3
  BEAM=5
  shift
fi

mkdir -p "$OUT"
export OUT_DIR="$OUT"
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
N_DEV="${#DEV_ARR[@]}"
echo "[sched-matrix] devices=${DEVICES} n=${N_DEV} requests=${REQUESTS} op=${INFER_OP} out=${OUT}"
echo "[sched-matrix] multi-card launcher: independent procs (no HCCL)"

CKPT_ARGS=()
if [[ -n "$CKPT" && -f "$CKPT" ]]; then
  CKPT_ARGS=(--ckpt "$CKPT")
  echo "[sched-matrix] using ckpt=${CKPT}"
fi

COMMON=(
  --device npu
  --data_dir "$DATA"
  --out_dir "$OUT"
  --requests "$REQUESTS"
  --warmup_requests "$WARMUP"
  --batch_size "$BATCH"
  --beam_size "$BEAM"
  --infer_op "$INFER_OP"
)

run_py() {
  # single-card: pin first visible device
  export ASCEND_RT_VISIBLE_DEVICES="${DEV_ARR[0]}"
  python -u scripts/sched_bench.py "$@"
}

run_mp() {
  local n="$1"
  local strategy="$2"
  shift 2
  local subset=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -n "$subset" ]]; then
      subset+=",${DEV_ARR[$i]}"
    else
      subset="${DEV_ARR[$i]}"
    fi
  done
  echo "[sched-matrix] MP strategy=${strategy} devices=${subset}"
  # wipe overalloc done flag if any
  rm -f "$OUT/.overalloc_done_n${n}"
  DEVICES="$subset" ROLE="$strategy" OUT_DIR="$OUT" \
    bash scripts/launch_infer_mp.sh \
      --strategy "$strategy" \
      "${COMMON[@]}" \
      "$@"
}

for s in tp pp ep sp; do
  run_py --strategy "$s" --out_dir "$OUT" --device npu || true
done

echo "[sched-matrix] === single inference baseline ==="
run_py \
  --strategy single \
  "${COMMON[@]}" \
  --tag baseline \
  "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

BASE="$(ls -1 "$OUT"/single_n1_baseline.json 2>/dev/null | head -1 || true)"
if [[ -z "$BASE" ]]; then
  BASE="$(ls -1 "$OUT"/single_*.json 2>/dev/null | head -1 || true)"
fi
echo "[sched-matrix] baseline=${BASE}"

if [[ "$N_DEV" -ge 2 ]]; then
  echo "[sched-matrix] === overalloc n=2 ==="
  run_mp 2 overalloc --baseline_json "${BASE}" "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
  echo "[sched-matrix] === replicas n=2 ==="
  run_mp 2 replicas --baseline_json "${BASE}" "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
fi

if [[ "$N_DEV" -ge 4 ]]; then
  echo "[sched-matrix] === overalloc n=4 ==="
  run_mp 4 overalloc --baseline_json "${BASE}" "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
  echo "[sched-matrix] === replicas n=4 ==="
  run_mp 4 replicas --baseline_json "${BASE}" "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
fi

if [[ "$TRAIN_PROFILE" == "1" ]]; then
  echo "[sched-matrix] === optional train_profile ==="
  run_py \
    --strategy train_profile \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps 20 \
    --batch_size 16 \
    --tag secondary \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
fi

python -u scripts/summarize_sched_bench.py --in_dir "$OUT" --out_dir "$OUT"
echo "[sched-matrix] DONE → ${OUT}/summary.md"
ls -la "$OUT"
