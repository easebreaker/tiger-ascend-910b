#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/ascend_env.sh"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
python3 -m tiger_ascend.ops.probe --device npu --out artifacts/op_probe_npu.json
python3 scripts/train_tiger.py --mode all --device npu --epochs 2 --batch_size 8
