# Paper-faithful TIGER Beauty on Ascend 910B

对照论文 Rajput et al., *Recommender Systems with Generative Retrieval*（NeurIPS 2023）在 **Amazon Beauty** 上的训练/评测配方，便于在 910B 上测算吞吐与指标。

## 论文配方（本仓 `--recipe paper`）

| 项 | 论文 | 本仓实现 |
|---|---|---|
| 数据 | Amazon Beauty，leave-one-out，max history 20 | `data/amazon_beauty/`（RQ-VAE SID 已预计算） |
| 模型 | 4×enc/dec，6 head×dim64，MLP=1024，dropout 0.1，~13M | `d_model=384,d_ff=1024,d_kv=64`（约 14–15M） |
| 用户 ID | 2000 hash token 拼在历史前 | `--user_hash_size 2000` |
| 优化 | batch 256；lr=0.01 恒定 10k step，再 inv-sqrt；Beauty 200k step | `batch 64 × accum 4`；`AdamW` + 同 LR 日程（论文 T5X 常用 Adafactor，NPU 上用 AdamW） |
| 指标 | Recall@5/10，NDCG@5/10 | 同；leave-one-out 下 Recall@K=Hit@K |
| Beam | 受限 beam | 默认 `num_beams=20` |

**Beauty 论文 Table 1 参考值：**

| Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0.0454 | 0.0321 | 0.0648 | 0.0384 |

> 语义 ID 来自社区预处理（NonameUntitled/tiger RQ-VAE），与论文自训 RQ-VAE 可能略有差异；指标接近即可，不保证 bit-exact。

## 910B 一键跑

```bash
# 可选：OOM 时改小 micro-batch
#   bash scripts/run_paper_beauty_910b.sh --batch_size 32 --grad_accum 8

bash scripts/run_paper_beauty_910b.sh
```

等价命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按本机路径
export ASCEND_RT_VISIBLE_DEVICES=0

python scripts/train_tiger.py --mode train --device npu --recipe paper \
  --data_dir data/amazon_beauty \
  --output_dir artifacts/ckpt_beauty_paper

python scripts/train_tiger.py --mode eval --device npu --recipe paper \
  --data_dir data/amazon_beauty \
  --output_dir artifacts/ckpt_beauty_paper
```

结果写在 `artifacts/ckpt_beauty_paper/eval_metrics.json`，日志会打印 `[vs paper Beauty Table 1]`。

## 与「跑通骨架」的区别

| | `--recipe smoke`（默认） | `--recipe paper` |
|---|---|---|
| 目的 | 链路验证 | 测算 / 对齐论文数 |
| 步数 | 数个 epoch | 200k updates |
| 有效 batch | 8 | 256 |
| AMP / workers | 关 / 0 | 开 / 4 |
| 用户 token | 无 | 2000 hash |

**不要**对真实 Beauty 目录使用 `--mode toy` / `all`（会覆盖 `inter.json`）。
