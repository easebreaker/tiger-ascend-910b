#!/usr/bin/env bash
# Inference-first multi-chip scheduling matrix (TIGER / Ascend 910B).
#
#   export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#   bash scripts/run_sched_bench.sh
#   bash scripts/run_sched_bench.sh --quick
#
# Primary: single / replicas / overalloc (+ tp|pp|ep|sp boundary JSON)
# Optional: TRAIN_PROFILE=1 to append a short train_profile run
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
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
N_DEV="${#DEV_ARR[@]}"
echo "[sched-matrix] devices=${DEVICES} n=${N_DEV} requests=${REQUESTS} op=${INFER_OP} out=${OUT}"

CKPT_ARGS=()
if [[ -n "$CKPT" && -f "$CKPT" ]]; then
  CKPT_ARGS=(--ckpt "$CKPT")
  echo "[sched-matrix] using ckpt=${CKPT}"
fi

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

for s in tp pp ep sp; do
  run_py --strategy "$s" --out_dir "$OUT" --device npu || true
done

echo "[sched-matrix] === single inference baseline ==="
run_py \
  --strategy single \
  --device npu \
  --data_dir "$DATA" \
  --out_dir "$OUT" \
  --requests "$REQUESTS" \
  --warmup_requests "$WARMUP" \
  --batch_size "$BATCH" \
  --beam_size "$BEAM" \
  --infer_op "$INFER_OP" \
  --tag baseline \
  "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

BASE="$(ls -1 "$OUT"/single_n1_baseline.json 2>/dev/null | head -1 || true)"
if [[ -z "$BASE" ]]; then
  BASE="$(ls -1 "$OUT"/single_*.json 2>/dev/null | head -1 || true)"
fi
echo "[sched-matrix] baseline=${BASE}"

if [[ "$N_DEV" -ge 2 ]]; then
  echo "[sched-matrix] === overalloc nproc=2 (waste) ==="
  run_torch 2 \
    --strategy overalloc \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --requests "$REQUESTS" \
    --warmup_requests "$WARMUP" \
    --batch_size "$BATCH" \
    --beam_size "$BEAM" \
    --infer_op "$INFER_OP" \
    --baseline_json "${BASE}" \
    --tag oa2 \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

  echo "[sched-matrix] === replicas nproc=2 ==="
  run_torch 2 \
    --strategy replicas \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --requests "$REQUESTS" \
    --warmup_requests "$WARMUP" \
    --batch_size "$BATCH" \
    --beam_size "$BEAM" \
    --infer_op "$INFER_OP" \
    --baseline_json "${BASE}" \
    --tag rep2 \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
fi

if [[ "$N_DEV" -ge 4 ]]; then
  echo "[sched-matrix] === overalloc nproc=4 ==="
  run_torch 4 \
    --strategy overalloc \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --requests "$REQUESTS" \
    --warmup_requests "$WARMUP" \
    --batch_size "$BATCH" \
    --beam_size "$BEAM" \
    --infer_op "$INFER_OP" \
    --baseline_json "${BASE}" \
    --tag oa4 \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

  echo "[sched-matrix] === replicas nproc=4 ==="
  run_torch 4 \
    --strategy replicas \
    --device npu \
    --data_dir "$DATA" \
    --out_dir "$OUT" \
    --requests "$REQUESTS" \
    --warmup_requests "$WARMUP" \
    --batch_size "$BATCH" \
    --beam_size "$BEAM" \
    --infer_op "$INFER_OP" \
    --baseline_json "${BASE}" \
    --tag rep4 \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
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
