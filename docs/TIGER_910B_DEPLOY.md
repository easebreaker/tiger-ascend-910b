# TIGER 在 Ascend 910B 上的部署说明

## 结论先说

**目前没有找到华为/昇腾官方的「TIGER → 910B」一键样例仓。**  
TIGER（*Transformer Index for GEnerative Recommenders*）与上次的 HSTU/GR 不是同一套模型：

| 模型 | 核心思路 | 昇腾官方适配 |
|---|---|---|
| **TIGER** | RQ-VAE 语义 ID + T5 自回归生成 + 前缀受限 beam search | **无官方样例** |
| **HSTU / Meta GR** | HSTU 序列建模 + 检索/排序 | 有（RecSDK / cann-recipes-infer） |

910B 上要跑 TIGER，现实路径是：  
**选一个开源 TIGER 实现 → 在 Ascend PyTorch（torch_npu）环境把 `cuda` 换成 `npu` → 验证训练/推理。**

---

## 1. 相关库地址（按推荐优先级）

### A. 推荐起步：torch-rechub（有完整 TIGER 复现）

| 项 | 地址 |
|---|---|
| 仓库 | https://github.com/datawhalechina/torch-rechub |
| TIGER 复现说明 | https://datawhalechina.github.io/torch-rechub/zh/blog/tiger_reproduction.html |
| 模型代码 | `torch_rechub/models/generative/tiger.py` |
| 脚本 | `examples/generative/run_tiger_movielens.py` / `run_tiger_amazon_books.py` |
| RQ-VAE | `examples/generative/run_rqvae_amazon_books.py` |

适合：快速在 910B 上验证「语义 ID + T5 生成」链路。

### B. 备选实现：PreferredAI / Cornac

| 项 | 地址 |
|---|---|
| 仓库 | https://github.com/PreferredAI/cornac |
| TIGER 封装 | `cornac/models/tiger/recom_tiger.py` |
| Seq2Seq 主体 | `cornac/models/tiger/tiger.py` |

适合：想要库级 API（`fit` / `score`）而不是独立训练脚本。

### C. 昇腾侧「生成式推荐」官方/半官方参考（不是 TIGER，但可复用环境）

| 项 | 地址 | 说明 |
|---|---|---|
| RecSDK | https://gitcode.com/Ascend/RecSDK （镜像：https://gitee.com/ascend/RecSDK） | 昇腾推荐 SDK；含 Meta GR / HSTU，**不含 TIGER** |
| RecSDK GR(Meta) 样例 | `torch2.6.0_examples_benchmark/develop/gr/gr_meta` | Meta `generative-recommenders` NPU 训练适配 |
| HSTU 推理样例 | https://gitcode.com/cann/cann-recipes-infer → `models/hstu` | Atlas A2/910B 推理 |
| 社区 GR 适配汇总 | https://gitcode.com/raintBN/Generative_Recommendation_on_Ascend | GR / OneRec / RankMixer，**仍无 TIGER** |
| torch_npu | https://github.com/Ascend/pytorch | Ascend PyTorch 插件 |
| RecSDK-Torch 镜像 | AscendHub `rec_sdk-torch` | 推荐场景基础镜像 |

### D. 论文与原版概念

- 论文关键词：*Recommender Systems with Generative Retrieval*（TIGER / Semantic ID）
- Meta 开源的是 **HSTU/GR**（https://github.com/meta-recsys/generative-recommenders），不是 Google TIGER 的官方代码仓

---

## 2. 910B 环境基线（建议）

与 HSTU/RecSDK 路线可共用一套机器环境：

| 组件 | 建议 |
|---|---|
| 硬件 | Atlas A2 / Ascend 910B |
| 驱动 | 主机 `npu-smi info` 正常 |
| CANN | 8.5.x（与 torch_npu 配套；社区 GR 常用 ≥8.5.1） |
| 算子包 | `Ascend-cann-910b-ops` |
| PyTorch | **与 torch_npu 严格同版本**（社区常用 `2.6.0`） |
| torch_npu | 与 PyTorch 同版本 |
| Python | 3.11 推荐 |
| 基础镜像 | AscendHub `rec_sdk-torch`，或 `quay.io/ascend/cann:*-910b-*` |

自检：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或 cann-8.5.0/set_env.sh
python3 - <<'PY'
import torch, torch_npu
print(torch.__version__, torch_npu.__version__)
print(torch.npu.is_available(), torch.npu.device_count())
x = torch.randn(2, 2).npu()
print(x.device, (x @ x).sum().item())
PY
```

---

## 3. 操作说明：用 torch-rechub 在 910B 跑 TIGER

### 3.1 拉代码并安装

```bash
mkdir -p /data/code && cd /data/code
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub

pip install -e ".[generative]"
pip install sentencepiece
# 若环境还没有匹配的 torch/torch_npu，按 CANN 配套安装，例如：
# pip install torch==2.6.0 torch_npu==2.6.0
```

### 3.2 改设备：CUDA → NPU（必须）

当前脚本默认：

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

在 `examples/generative/run_tiger_movielens.py`（及 amazon 脚本）中改为类似：

```python
import torch
import torch_npu  # noqa: F401

def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
```

同步检查：

- `pin_memory=(device.type == "cuda")` → NPU 上建议改为 `False`，或写成 `device.type in ("cuda",)`
- HuggingFace `Trainer` / `TrainingArguments`：确认最终 `.to(device)` 或 `device` 参数落到 `npu`
- 多卡时用 `ASCEND_RT_VISIBLE_DEVICES`，分布式后端用 **HCCL**（不是 NCCL）

### 3.3 Toy 端到端（先验证 NPU 能跑通）

需能访问 HuggingFace，或把 `t5-small` 的 tokenizer/config 提前缓存到本地，再用 `--base_model /path/to/local`。

```bash
cd /data/code/torch-rechub/examples/generative
export ASCEND_RT_VISIBLE_DEVICES=0

python run_tiger_movielens.py --mode all \
  --data_inter_path ./tmp/tiger-toy/ml/inter.json \
  --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json \
  --output_dir ./tmp/tiger-toy/ml/ckpt \
  --toy_num_users 16 --toy_num_items 20 \
  --epochs 2 --per_device_batch_size 4 \
  --num_beams 4 --test_batch_size 2 --num_workers 0
```

期望日志出现：`Added N semantic-id tokens` → `Model saved` → `Test results: {...}`。

### 3.4 真实数据（MovieLens-1M 示意）

```bash
# 1) 交互序列
python run_tiger_movielens.py --mode prepare-data \
  --ratings_path ./data/ml-1m/ratings.dat \
  --data_inter_path ./data/ml-1m/tiger/inter.json \
  --min_seq_len 5 --max_his_len 20

# 2) 语义 ID：用 RQ-VAE 对 item embedding 量化，导出 semantic_ids.json
#    （参考 run_rqvae_amazon_books.py；务必保证 item id 与 inter.json 对齐）

# 3) 训练 / 测试
python run_tiger_movielens.py --mode train \
  --data_inter_path ./data/ml-1m/tiger/inter.json \
  --data_indice_path ./data/ml-1m/tiger/semantic_ids.json \
  --output_dir ./ckpt/tiger_ml

python run_tiger_movielens.py --mode test \
  --ckpt_path ./ckpt/tiger_ml \
  --data_inter_path ./data/ml-1m/tiger/inter.json \
  --data_indice_path ./data/ml-1m/tiger/semantic_ids.json
```

---

## 4. 910B 上要注意的点

1. **没有 RecSDK 专用 TIGER 融合算子**  
   主体是 T5 + embedding + beam search，一般走 torch_npu 通用算子；性能优化空间在后续（图模式、自定义算子），不是开箱即用。

2. **受限 beam search / Trie**  
   多为 CPU 侧逻辑 + 模型前向在 NPU；首次跑通时不要急着改解码，先保证 `generate` 在 NPU 上结果正确。

3. **词表扩展**  
   训练必须 `add_tokens(semantic_ids)` + `resize_token_embeddings`；否则语义 ID 被切成子词，指标会假低。

4. **与 HSTU 部署不要混用同一套脚本**  
   HSTU 依赖 RecSDK 的 `hstu_paged` / jagged / FBGEMM NPU；TIGER 不依赖这些，但可共用 CANN + torch_npu 环境。

5. **若目标其实是「生成式推荐上 910B」而不是纸面 TIGER**  
   更稳妥的官方路径仍是：
   - 训练：RecSDK `gr_meta` / `gr_nv`
   - 推理：`cann-recipes-infer/models/hstu`

---

## 5. 建议落地顺序

```text
确认 910B + CANN + torch_npu
  → clone torch-rechub
  → 脚本 device 改为 npu
  → toy --mode all 跑通
  → 准备 inter.json + semantic_ids.json
  → train / test
  →（可选）对照 RecSDK GR 环境做性能/工程化
```

---

## 6. 如果你指的不是这个 TIGER

若「Tiger」是内部项目名、其他模型或商业组件，需要补充全称或仓库链接；当前公开昇腾生态里**没有**与 HSTU 同级的官方 TIGER 910B 部署包。
