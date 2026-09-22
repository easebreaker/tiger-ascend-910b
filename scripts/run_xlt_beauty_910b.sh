#!/usr/bin/env bash
# XiaoLongtaoo/TIGER-aligned Beauty on Ascend 910B.
#
# Smoke (subset + probe-friendly):
#   bash scripts/run_xlt_beauty_910b.sh --smoke
# Isolate crash step:
#   python -u scripts/npu_xlt_probe.py
# Full:
#   bash scripts/run_xlt_beauty_910b.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/script/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/script/set_env.sh
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

OUT="${OUT_DIR:-artifacts/ckpt_beauty_xlt}"
DATA="${DATA_DIR:-data/amazon_beauty}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
INFER_BS="${INFER_BATCH_SIZE:-32}"
LAYOUT="${LAYOUT:-npu_safe}"
SKIP_EVAL=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      DATA="${SMOKE_DATA:-data/amazon_beauty/subset_512u}"
      OUT="${OUT_DIR:-artifacts/ckpt_beauty_xlt_smoke}"
      BATCH_SIZE=2
      GRAD_ACCUM=1
      INFER_BS=2
      LAYOUT=npu_safe
      EXTRA+=(--max_train_steps 10 --skip_valid --epochs 1 --batch_size 2 --grad_accum 1 --layout npu_safe)
      SKIP_EVAL=1
      shift
      ;;
    --probe)
      echo "[xlt-910b] running NPU probe (last [probe N] = crash site)"
      python -u scripts/npu_xlt_probe.py
      exit 0
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

echo "[xlt-910b] sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[xlt-910b] data=${DATA} layout=${LAYOUT} workers=${NUM_WORKERS} batch=${BATCH_SIZE} accum=${GRAD_ACCUM}"

python -u scripts/train_tiger_xlt.py \
  --mode train \
  --device npu \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --infer_batch_size "${INFER_BS}" \
  --beam_size 30 \
  --epochs 200 \
  --early_stop 10 \
  --lr 1e-4 \
  --num_workers "${NUM_WORKERS}" \
  --layout "${LAYOUT}" \
  "${EXTRA[@]+"${EXTRA[@]}"}"

if [[ "${SKIP_EVAL}" -eq 1 ]]; then
  echo "[xlt-910b] skip eval (smoke / --skip-eval)"
  exit 0
fi

echo "[xlt-910b] eval"
python -u scripts/train_tiger_xlt.py \
  --mode eval \
  --device npu \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  --infer_batch_size "${INFER_BS}" \
  --beam_size 30 \
  --num_workers "${NUM_WORKERS}" \
  --layout "${LAYOUT}"

echo "[xlt-910b] ${OUT}/eval_metrics.json"
