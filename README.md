# TIGER → Ascend 910B Adaptation Workspace

基于 [torch-rechub TIGER](https://github.com/datawhalechina/torch-rechub) 源码整理的 **910B 适配试验仓库**。  
目标：在 CPU 上先跑通 toy 流程，再到 Atlas A2 / Ascend 910B 上用 `torch_npu` 做设备与算子对齐。

> 说明：昇腾生态目前 **没有官方 TIGER 一键样例**。本仓库把 T5 版 TIGER 收成可改写骨架；**不依赖 RecSDK 的 HSTU 融合算子**。

## 仓库结构

```text
tiger_ascend/                 # 模型 / 数据 / NPU 工具
data/amazon_beauty/           # Amazon Beauty 预处理数据（论文同款）
scripts/train_tiger.py        # toy / train / eval 入口
scripts/convert_tiger_amazon_index.py
docs/ASCEND_910B_ENV_AND_OPS.md
docs/DEPLOY_STEP_BY_STEP_910B.md
```

## 真实数据（Amazon Beauty）

已内置论文常用的 **Amazon Beauty** 预处理数据（交互 + RQ-VAE 语义 ID）：

- 说明：[`data/amazon_beauty/README.md`](data/amazon_beauty/README.md)
- 全量：`data/amazon_beauty/`（约 2.2 万用户 / 1.2 万 item）
- 快速试跑：`data/amazon_beauty/subset_512u/`

```bash
# 子集上 NPU 训练（不要用 --mode all，以免覆盖真实数据）
python scripts/train_tiger.py --mode train --device npu \
  --data_dir data/amazon_beauty/subset_512u \
  --output_dir artifacts/ckpt_beauty_subset --epochs 3 --batch_size 16

python scripts/train_tiger.py --mode eval --device npu \
  --data_dir data/amazon_beauty/subset_512u \
  --output_dir artifacts/ckpt_beauty_subset
```

## 指标怎么读（loss ↓ 但 Hit@K≈0）

Leave-one-out 下 **train loss 和 test Hit@K 不是同一任务**：

- 训练：对每个用户 `items[:-2]` 做滑窗 next-SID（teacher forcing）
- 测试：用 `items[:-1]` 预测从未当过该用户训练目标的 `items[-1]`，再 **自回归 + Trie beam** 生成整条 SID

因此常见现象：`train_loss≈0.2` 但 `token_acc(test TF)≈0.35`、`Hit@10≈0`。  
**同一次 eval 会打印 `[diag/train]` 与 `[diag/test]`**：若 train TF 很高、test TF 低 → 过拟合/LOO 难度，不是 CE 对齐写错；若同一 split 上 `ce < ce_lb(acc)` 才会报 WARNING。

随机基线（Beauty subset ≈3175 items）：`Hit@10 ≈ 10/3175 ≈ 0.003`。

## 快速开始（CPU 冒烟）

```bash
pip install -e .
python scripts/train_tiger.py --mode all --device cpu --epochs 1 --batch_size 4
python -m tiger_ascend.ops.probe --device cpu
```

## 910B 上跑

逐步部署请看：**[docs/DEPLOY_STEP_BY_STEP_910B.md](docs/DEPLOY_STEP_BY_STEP_910B.md)**  
环境与算子清单：**[docs/ASCEND_910B_ENV_AND_OPS.md](docs/ASCEND_910B_ENV_AND_OPS.md)**

```bash
# 1) 配好 CANN + torch / torch_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径按本机安装调整

# 2) 算子探测
python3 -m tiger_ascend.ops.probe --device npu --out artifacts/op_probe.json

# 3) toy 训练
ASCEND_RT_VISIBLE_DEVICES=0 \
python3 scripts/train_tiger.py --mode all --device npu --epochs 2 --batch_size 8
```

## 与 HSTU / RecSDK 的关系

| 项目 | TIGER（本仓） | HSTU / Meta GR |
|---|---|---|
| 骨干 | T5 encoder-decoder | HSTU |
| 语义 ID | RQ-VAE / 预计算 semantic ids | 通常直接 item id / 其他索引 |
| 昇腾依赖 | `torch` + `torch_npu` + 通用 CANN 算子 | RecSDK + 自定义融合算子 |
| 官方样例 | 无 | RecSDK / cann-recipes-infer |

## 源码来源

- 模型 / 数据逻辑改编自 Datawhale [torch-rechub](https://github.com/datawhalechina/torch-rechub)（Apache-2.0）
- 论文：Rajput et al., *Recommender Systems with Generative Retrieval*（TIGER）

详见 [docs/ASCEND_910B_ENV_AND_OPS.md](docs/ASCEND_910B_ENV_AND_OPS.md)。
