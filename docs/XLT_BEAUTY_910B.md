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
# 0) 先确认 commit（日志会打印 sha=... layout=npu_safe）
git log -1 --oneline

# 1) 分步探测（最后一行 [probe N] 即崩溃点）
bash scripts/run_xlt_beauty_910b.sh --probe

# 1b) 若 probe 死在 matmul：先跑更细的基础探针
export ASCEND_LAUNCH_BLOCKING=1
python -u scripts/npu_basic_probe.py

# 1c) basic 通过但 smoke 死在 model.to(npu)：
python -u scripts/npu_model_to_probe.py
# 看最后停在 m1 / m2 / m3

# 2) 冒烟（自动用 subset_512u + batch=2 + layout=npu_safe）
bash scripts/run_xlt_beauty_910b.sh --smoke

# 3) 正式全量
bash scripts/run_xlt_beauty_910b.sh
```

**重要：** 若日志仍是 `params=4.59M` 且 `d_model=128 heads=6 d_kv=64`，说明还在跑旧代码/旧 layout（`128≠6*64`）。新版本应看到：

`layout=npu_safe d_model=128 heads=4 d_kv=32 aligned=True`

`stack smashing` / 无堆栈 `Aborted`：先跑 `--probe`，把完整输出贴回。
