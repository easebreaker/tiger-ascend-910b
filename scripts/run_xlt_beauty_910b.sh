#!/usr/bin/env bash
# XiaoLongtaoo/TIGER-aligned Beauty on Ascend 910B.
# Upstream Beauty: R@5=0.0392 N@5=0.0257 R@10=0.0594 N@10=0.0321
#
# NPU tip: default num_workers=0 (workers>0 often causes "Aborted" with torch_npu).
# Smoke:  bash scripts/run_xlt_beauty_910b.sh --smoke
# Full:   bash scripts/run_xlt_beauty_910b.sh
# OOM:    bash scripts/run_xlt_beauty_910b.sh --batch_size 32 --grad_accum 8
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
# Avoid tokenizer/thread forks fighting NPU runtime
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

OUT="${OUT_DIR:-artifacts/ckpt_beauty_xlt}"
DATA="${DATA_DIR:-data/amazon_beauty}"
# 0 is safest on Ascend; set NUM_WORKERS=2 only after smoke passes
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
INFER_BS="${INFER_BATCH_SIZE:-32}"
SKIP_EVAL=0
EXTRA=()

# Parse leading --smoke; remaining args go to python
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      BATCH_SIZE=8
      GRAD_ACCUM=1
      INFER_BS=8
      EXTRA+=(--max_train_steps 20 --skip_valid --epochs 1 --batch_size 8 --grad_accum 1)
      SKIP_EVAL=1
      shift
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

echo "[xlt-910b] device=npu workers=${NUM_WORKERS} batch=${BATCH_SIZE} accum=${GRAD_ACCUM} out=${OUT}"
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
  --num_workers "${NUM_WORKERS}"

echo "[xlt-910b] ${OUT}/eval_metrics.json"
