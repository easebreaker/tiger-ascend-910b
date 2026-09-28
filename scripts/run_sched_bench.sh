#!/usr/bin/env bash
# Full multi-chip scheduling experiment matrix for TIGER / Ascend 910B.
#
# Usage:
#   export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7   # pick FREE cards
#   bash scripts/run_sched_bench.sh            # full matrix
#   bash scripts/run_sched_bench.sh --quick    # fewer steps
#
# Collects: P50/P90/P99 step latency, samples/s, peak HBM, phase times,
# speedup vs single, DP efficiency, overalloc waste_factor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/ascend_env.sh"

STEPS="${STEPS:-50}"
WARMUP="${WARMUP_STEPS:-5}"
DATA="${DATA_DIR:-data/amazon_beauty/subset_512u}"
OUT="${OUT_DIR:-artifacts/sched_bench/$(date +%Y%m%d_%H%M%S)}"
BATCH="${BATCH_SIZE:-32}"
DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

if [[ "${1:-}" == "--quick" ]]; then
  STEPS=20
  WARMUP=3
  shift
fi

mkdir -p "$OUT"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

# How many visible cards?
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
N_DEV="${#DEV_ARR[@]}"
echo "[sched-matrix] devices=${DEVICES} n=${N_DEV} steps=${STEPS} out=${OUT}"

run_py() {
  python -u scripts/sched_bench.py "$@"
}

run_torch() {
  local np="$1"
  shift
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --standalone --nproc_per_node="$np" scripts/sched_bench.py "$@"
  else
    python -m torch.distributed.run --standalone --nproc_per_node="$np" scripts/sched_bench.py "$@"
  fi
}

# --- boundary markers (no NPU needed for JSON, but ok to run here) ---
for s in tp pp ep sp; do
  run_py --strategy "$s" --out_dir "$OUT" --device npu || true
done

# --- 1) single baseline ---
echo "[sched-matrix] === single baseline ==="
run_py \
  --strategy single \
  --device npu \
  --data_dir "$DATA" \
  --out_dir "$OUT" \
  --steps "$STEPS" \
  --warmup_steps "$WARMUP" \
  --batch_size "$BATCH" \
  --grad_accum 1 \
  --tag baseline

BASE="$(ls -1 "$OUT"/single_n1_baseline.json 2>/dev/null | head -1 || true)"
if [[ -z "$BASE" ]]; then
  BASE="$(ls -1 "$OUT"/single_*.json 2>/dev/null | head -1 || true)"
fi
echo "[sched-matrix] baseline=${BASE}"

# --- 2) overalloc: hold 2 / min(4,N) cards, train on 1 ---
if [[ "$N_DEV" -ge 2 ]]; then
  echo "[sched-matrix] === overalloc nproc=2 ==="
  run_torch 2 \
    --strategy overalloc \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" \
    --baseline_json "${BASE}" \
    --tag oa2
fi
if [[ "$N_DEV" -ge 4 ]]; then
  echo "[sched-matrix] === overalloc nproc=4 ==="
  run_torch 4 \
    --strategy overalloc \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" \
    --baseline_json "${BASE}" \
    --tag oa4
fi

# --- 3) DP strong scaling (fixed global batch) ---
if [[ "$N_DEV" -ge 2 ]]; then
  echo "[sched-matrix] === dp nproc=2 fixed global batch ==="
  run_torch 2 \
    --strategy dp \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" \
    --global_batch_mode fixed \
    --target_global_batch "$BATCH" \
    --baseline_json "${BASE}" \
    --tag dp2_fixed
fi
if [[ "$N_DEV" -ge 4 ]]; then
  echo "[sched-matrix] === dp nproc=4 fixed global batch ==="
  run_torch 4 \
    --strategy dp \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" \
    --global_batch_mode fixed \
    --target_global_batch "$BATCH" \
    --baseline_json "${BASE}" \
    --tag dp4_fixed
fi

# --- 4) DP weak scaling (optional) ---
if [[ "$N_DEV" -ge 2 ]]; then
  echo "[sched-matrix] === dp nproc=2 scale (weak) ==="
  run_torch 2 \
    --strategy dp \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --steps "$STEPS" \
    --warmup_steps "$WARMUP" \
    --batch_size "$BATCH" \
    --global_batch_mode scale \
    --baseline_json "${BASE}" \
    --tag dp2_weak
fi

python -u scripts/summarize_sched_bench.py --in_dir "$OUT" --out_dir "$OUT"
echo "[sched-matrix] DONE → ${OUT}/summary.md"
echo "[sched-matrix] key files:"
ls -la "$OUT"
