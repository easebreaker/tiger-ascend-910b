# TIGER → Ascend 910B Adaptation Workspace

基于 [torch-rechub TIGER](https://github.com/datawhalechina/torch-rechub) 源码整理的 **910B 适配试验仓库**。  
目标：在 CPU 上先跑通 toy 流程，再到 Atlas A2 / Ascend 910B 上用 `torch_npu` 做设备与算子对齐。

> 说明：昇腾生态目前 **没有官方 TIGER 一键样例**。本仓库把 T5 版 TIGER 收成可改写骨架；**不依赖 RecSDK 的 HSTU 融合算子**。

## 仓库结构

```text
tiger_ascend/
  model/tiger.py          # 自 torch-rechub 改编的 TIGERModel（去硬编码 CUDA）
  data/dataset.py         # TigerSeqDataset + Trie 受限解码
  data/tokenizer.py       # 离线 Semantic-ID tokenizer（无需下载 T5 词表）
  utils/device.py         # auto/npu/cuda/cpu 设备选择
  ops/probe.py            # 910B 必要算子探测
scripts/train_tiger.py    # toy → train → eval 入口
docs/ASCEND_910B_ENV_AND_OPS.md
```

## 快速开始（CPU 冒烟）

```bash
pip install -e .
python scripts/train_tiger.py --mode all --device cpu --epochs 1 --batch_size 4
python -m tiger_ascend.ops.probe --device cpu
```

## 910B 上跑

```bash
# 1) 配好 CANN + torch / torch_npu（见 docs/ASCEND_910B_ENV_AND_OPS.md）
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径按本机安装调整

# 2) 算子探测
python -m tiger_ascend.ops.probe --device npu --out artifacts/op_probe.json

# 3) toy 训练
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/train_tiger.py --mode all --device npu --epochs 2 --batch_size 8
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
