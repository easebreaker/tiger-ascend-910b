# Amazon Beauty（TIGER 论文同款预处理数据）

本目录供 `scripts/train_tiger.py` 直接使用，包含：

| 文件 | 说明 |
|---|---|
| `inter.json` | 用户交互序列 `{user_id: [item_id, ...]}` |
| `semantic_ids.json` | RQ-VAE 语义 ID `{item_id: ["<a_..>", "<b_..>", "<c_..>", "<d_..>"]}` |
| `subset_512u/` | 512 用户子集，方便 910B 快速试跑 |
| `manifest.json` | 规模与来源元信息 |
| `SOURCE_LICENCE` | 上游仓库许可证副本 |

## 来源与引用

- 预处理文件来自社区实现：[NonameUntitled/tiger](https://github.com/NonameUntitled/tiger)（`data/Beauty/inter.json` + `index_rqvae.json`）
- 论文：Rajput et al., *Recommender Systems with Generative Retrieval* (NeurIPS 2023)  
  https://arxiv.org/abs/2305.05065
- 原始 Amazon Reviews：https://jmcauley.ucsd.edu/data/amazon/
- Sports / Toys 等更大集合可从 Zenodo 获取：https://zenodo.org/records/17351848  
  转换脚本：`scripts/convert_tiger_amazon_index.py`

## 规模（全量 Beauty）

- users: **22363**
- items: **12101**
- 语义 ID：4 层 × codebook 0–255 → token 形如 `<a_12><b_3><c_40><d_7>`

## 论文配方测算（推荐）

**选定参考：XiaoLongtaoo/TIGER**

```bash
bash scripts/run_xlt_beauty_910b.sh
```

说明：[`docs/XLT_BEAUTY_910B.md`](../../docs/XLT_BEAUTY_910B.md)  
上游 Beauty：R@5=0.0392，N@5=0.0257，R@10=0.0594，N@10=0.0321。

按论文文字的另一配方：[`docs/PAPER_BEAUTY_910B.md`](../../docs/PAPER_BEAUTY_910B.md)。

## 子集冒烟

```bash
python scripts/train_tiger.py \
  --mode train --device npu \
  --data_dir data/amazon_beauty/subset_512u \
  --output_dir artifacts/ckpt_beauty_subset \
  --epochs 3 --batch_size 16
```

> 注意：不要对真实 `data_dir` 使用 `--mode all` / `toy`，那会覆盖 `inter.json`。
