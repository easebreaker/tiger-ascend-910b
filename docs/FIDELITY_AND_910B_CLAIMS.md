# TIGER 复刻保真度说明（能否代表 910B「真实性能」）

## 直接结论

| 问题 | 答案 |
|---|---|
| 能否**确保**与论文 bit-exact 复刻？ | **不能。** 原作者未开源；本仓是按论文描述 + 社区实现整理的可运行复现。 |
| `--recipe paper` 是否按论文**方法**做？ | **大体是**（生成式 SID + T5 seq2seq + Trie beam + Beauty LOO）。 |
| 910B 上跑出的数能否当「该方法真实性能」？ | **可以代表本移植在 910B 上的实测**（吞吐 + 该设定下的 Recall/NDCG）；**不能自动等同**论文 Table 1 的 GPU/TPU 数字。 |

要用 910B 结果对外引用，请同时报：**本仓 commit、CANN/torch_npu 版本、`--recipe paper`、SID 来源、最终 `eval_metrics.json`**。

---

## 与论文对齐情况

### 已对齐（方法层面）

| 项 | 论文 | 本仓 |
|---|---|---|
| 任务 | 历史 → 自回归生成下一 item 的 Semantic ID | 同 |
| 切分 | leave-one-out；history≤20 | 同 |
| SID 形态 | 4 级 codebook（含碰撞消歧位） | Beauty 数据为 4 级 `<a/b/c/d_*>` |
| 骨干 | T5 encoder-decoder，随机初始化 | HF `T5ForConditionalGeneration` 子类，随机初始化 |
| 规模 | ~13M；4 层；6 head×dim64；d_ff≈1024 | `d_model=384,d_ff=1024,d_kv=64` → ~14.5M |
| 用户 token | 2000 hash 拼在序列前 | `--user_hash_size 2000` |
| 训练量 | Beauty 200k step，batch 256 | `max_steps=200000`，`64×4` accum |
| LR | 0.01 恒定 10k，再 inv-sqrt | 同形状（`LambdaLR`） |
| 约束解码 | 合法 SID 前缀 Trie / beam | `Trie` + `prefix_allowed_tokens_fn` |
| 指标 | Recall@K / NDCG@K（K=5,10） | 同；LOO 下 Recall@K=Hit@K |

### 已知偏差（会影响「是否贴上 Table 1」）

| 项 | 风险 | 说明 |
|---|---|---|
| **RQ-VAE / SID 本身** | **高** | 论文用 Sentence-T5 + 自训 RQ-VAE；本仓用社区预处理 SID（NonameUntitled/tiger）。SID 不同 → 指标不可强制对齐。 |
| **优化器** | 中 | 论文走 T5X（多为 **Adafactor**）；本仓 **AdamW**（910B/torch_npu 更稳）。同 LR 日程 ≠ 同轨迹。 |
| **框架** | 中 | T5X（JAX/TPU）vs HuggingFace T5（PyTorch/NPU）；相对位置、初始化、数值路径不同。 |
| **混合精度** | 中 | paper recipe 默认 **AMP fp16**；论文未写死精度。测「精度上限」建议加 `--no_amp` 对比。 |
| **Beam 宽度** | 中低 | Table 1 未给确切 beam；本仓默认 20。beam 过小会压 Recall。 |
| **「input dim=128」表述** | 低 | 文中含糊；本仓按 **6×64→d_model=384** 且参数量贴 ~13M 选取（非 Cornac 的 d_model=128≈5M）。 |
| **weight decay** | 低 | 本仓 AdamW `0.01`；论文未写。 |
| **temperature CE** | 低 | 继承 rechub 的 `/T`；paper 默认 `T=1` 即标准 CE。 |

### 本环境未替你完成的事

- 未在本 Cloud Agent 机器上跑满 **200k step 全量 Beauty NPU 训练**（需你在 910B 上跑 `run_paper_beauty_910b.sh`）。
- 因此：**仓库本身不附带「已验证的 910B 精度数字」**；数字以你机器上的 `eval_metrics.json` 为准。

---

## 「910B 真实性能」应怎么理解

拆成两件事，不要混谈：

1. **系统性能（硬件/栈）**  
   用 paper recipe 在 910B 上测：`updates/s`、step 耗时、评测 wall time、是否 OOM。  
   → 这能诚实反映 **TIGER 这类工作负载在 910B + 当前 CANN/torch_npu 上的吞吐与稳定性**。

2. **算法精度（方法）**  
   同一 recipe 下的 Recall/NDCG，并与 Table 1 对照打印。  
   → 反映的是 **「该移植 + 该 SID + AdamW/AMP」在 910B 上的精度**，不是「论文原文在未知加速器上的官方数」。

若 910B 精度明显低于论文，优先排查顺序：SID 质量 → 关 AMP → 加大 beam → 确认 200k 跑满 → 再谈 NPU 算子/数值。

---

## 建议你怎么报结果

```text
TIGER Ascend port (commit <sha>), recipe=paper, Beauty
CANN x.y / torch_npu z / AMP=on|off / beams=20
SID: community RQ-VAE (NonameUntitled/tiger)
Recall@5=… NDCG@5=… Recall@10=… NDCG@10=…
Paper Table1: 0.0454 / 0.0321 / 0.0648 / 0.0384
Train: updates/s=…  Eval: …s / 22363 users
```

一键脚本：`bash scripts/run_paper_beauty_910b.sh`（见同目录 `PAPER_BEAUTY_910B.md`）。
