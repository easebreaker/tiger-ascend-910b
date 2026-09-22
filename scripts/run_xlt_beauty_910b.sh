#!/usr/bin/env bash
# XiaoLongtaoo/TIGER-aligned Beauty on Ascend 910B.
# Upstream Beauty: R@5=0.0392 N@5=0.0257 R@10=0.0594 N@10=0.0321
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
OUT="${OUT_DIR:-artifacts/ckpt_beauty_xlt}"
DATA="${DATA_DIR:-data/amazon_beauty}"

echo "[xlt-910b] train → ${OUT}"
python scripts/train_tiger_xlt.py \
  --mode train \
  --device npu \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  --batch_size 64 \
  --grad_accum 4 \
  --infer_batch_size 64 \
  --beam_size 30 \
  --epochs 200 \
  --early_stop 10 \
  --lr 1e-4 \
  --num_workers 4 \
  "$@"

echo "[xlt-910b] eval"
python scripts/train_tiger_xlt.py \
  --mode eval \
  --device npu \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  --infer_batch_size 64 \
  --beam_size 30

echo "[xlt-910b] ${OUT}/eval_metrics.json"
