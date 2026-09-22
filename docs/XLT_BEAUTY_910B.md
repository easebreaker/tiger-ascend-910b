# 参考实现：XiaoLongtaoo/TIGER（910B 测算）

选定上游：[XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)（~252★，MIT）

本仓入口：`scripts/train_tiger_xlt.py` · `scripts/run_xlt_beauty_910b.sh`

## 对齐清单

| 项 | 上游 | 本仓 |
|---|---|---|
| 编码 | level offset，vocab=1025，pad/eos=0 | `tiger_ascend/data/xlt.py` 从 `semantic_ids.json` 转换 |
| 模型 | d_model=128, d_ff=1024, L=4, H=6, d_kv=64 | `build_xlt_t5_config` |
| 优化 | Adam `1e-4`，batch 256，最多 200 epoch | 同；910B 默认 `64×4` accum |
| 早停 | valid NDCG@20，patience=10 | 同 |
| 生成 | `max_length=5`，beam=30，无 Trie | 同 |
| 指标 | 四码完全匹配 → Recall/NDCG | 同 |
| 用户 hash | 无 | 无 |

## Beauty 对照

| | XiaoLongtaoo Ours | Paper Table 1 |
|---|---:|---:|
| Recall@5 | **0.0392** | 0.0454 |
| NDCG@5 | **0.0257** | 0.0321 |
| Recall@10 | **0.0594** | 0.0648 |
| NDCG@10 | **0.0321** | 0.0384 |

910B 优先对齐左列。SID 仍用本仓社区预处理（与上游自训 RQ-VAE 可能略有差异）。

## 910B

```bash
bash scripts/run_xlt_beauty_910b.sh
# OOM:  ... --batch_size 32 --grad_accum 8
# 冒烟: ... --max_train_steps 20 --skip_valid --epochs 1
```

结果：`artifacts/ckpt_beauty_xlt/eval_metrics.json`
