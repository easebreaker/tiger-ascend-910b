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

# 1c) basic 通过但 smoke 死在 model.to / safe move：
#     - launcher 会自动做 修法2（driver lib64 前置）
#     - 训练改为：warmup → copy_ 去重 H2D → 再读数据（与探针顺序一致）
python -u scripts/npu_model_to_probe.py
# 期望：[m3] safe move ok；若 [m4] Abort 可忽略
# smoke 若死在 p2 set_device：已默认跳过 set_device，只用 device='npu:0' 分配。
#   git pull && export ASCEND_RT_VISIBLE_DEVICES=7
#   bash scripts/run_xlt_beauty_910b.sh --probe-device   # 应看到 skip set_device 后 s7/s8 ok
# 若死在 s7a / w2a zeros(1).npu()：先确认 launcher 打印
#   [ascend-env] OK: driver lib64 is ahead of toolkit
# 旧写法若 driver 已在 PATH 中间会跳过 prepend，toolkit 仍抢第一 → Abort。
# 现 scripts/ascend_env.sh 会摘掉再强制 driver/lib64/{driver,common,} 置顶。
#   bash scripts/run_xlt_beauty_910b.sh --probe-acl
#   bash scripts/run_xlt_beauty_910b.sh --smoke

# 2) 冒烟（自动用 subset_512u + batch=2 + layout=npu_safe）
bash scripts/run_xlt_beauty_910b.sh --smoke
# 若仍中断：把最后一条 [warmup]/[npu-move]/[step] 行贴回

# 3) 正式全量
bash scripts/run_xlt_beauty_910b.sh
```

**重要：** 若日志仍是 `params=4.59M` 且 `d_model=128 heads=6 d_kv=64`，说明还在跑旧代码/旧 layout（`128≠6*64`）。新版本应看到：

`layout=npu_safe d_model=128 heads=4 d_kv=32 aligned=True`

`stack smashing` / 无堆栈 `Aborted`：先跑 `--probe`，把完整输出贴回。
