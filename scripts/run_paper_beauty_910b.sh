#!/usr/bin/env bash
# Paper-faithful TIGER Beauty training + eval on Ascend 910B.
# Target metrics (Table 1): R@5=0.0454 N@5=0.0321 R@10=0.0648 N@10=0.0384
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
OUT="${OUT_DIR:-artifacts/ckpt_beauty_paper}"
DATA="${DATA_DIR:-data/amazon_beauty}"

echo "[paper-910b] train Beauty → ${OUT}"
python scripts/train_tiger.py \
  --mode train \
  --device npu \
  --recipe paper \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  "$@"

echo "[paper-910b] eval Beauty"
python scripts/train_tiger.py \
  --mode eval \
  --device npu \
  --recipe paper \
  --data_dir "${DATA}" \
  --output_dir "${OUT}" \
  --skip_tf_diag

echo "[paper-910b] metrics: ${OUT}/eval_metrics.json"
