#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m tiger_ascend.ops.probe --device cpu
python3 scripts/train_tiger.py --mode all --device cpu --epochs 1 --batch_size 4 \
  --toy_users 16 --toy_items 32 --d_model 64 --num_layers 1 --num_heads 2
