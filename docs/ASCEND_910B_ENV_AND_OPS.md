# Ascend 910B：TIGER 环境依赖与算子要求

本文给出在 **Atlas A2 / Ascend 910B** 上尝试适配本仓库 TIGER 实现时，需要准备的软件环境与算子清单。

## 1. 硬件与驱动

| 项 | 要求 |
|---|---|
| 硬件 | Ascend 910B（Atlas 800T A2 / 等价 A2 训练服务器） |
| 驱动 / 固件 | 与所选 CANN 版本匹配的 Ascend HDK（以昇腾兼容性矩阵为准） |
| 自检 | `npu-smi info` 可见 910B；存在 `/dev/davinci*` |

## 2. 必备软件栈（训练 / 前向）

TIGER 主体是 **T5 seq2seq + Embedding + CE + generate**，不需要 RecSDK 的 HSTU 融合算子。推荐组合：

| 组件 | 建议 | 说明 |
|---|---|---|
| OS | openEuler / Ubuntu（与镜像一致） | 优先用昇腾官方或 RecSDK-Torch 镜像 |
| Python | 3.9–3.11（推荐 3.10/3.11） | 与 torch_npu wheel 一致 |
| CANN | **8.5.x**（常用 8.5.0 / 8.5.1） | 必须与 torch_npu 配套 |
| cann-toolkit | 与 CANN 同版本 | 开发套件 |
| cann-910b-ops | 与 CANN 同版本 | **910B 算子包**（不要装错 A3-ops） |
| PyTorch | 与 torch_npu **同版本**（社区常用 2.6.0 / 2.1–2.5 视站点包而定） | 版本错配会直接找不到 NPU |
| torch_npu | 与 PyTorch / CANN 匹配 | Ascend PyTorch 插件 |
| transformers | `>=4.38,<5`（本仓 pin） | T5ForConditionalGeneration |
| numpy / tqdm / sentencepiece | 见 `requirements.txt` | tokenizer 与工具 |

### 推荐安装顺序

```bash
# 主机已装驱动后，容器/环境内：
bash Ascend-cann-toolkit_<ver>_linux-<arch>.run --install
bash Ascend-cann-910b-ops_<ver>_linux-<arch>.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或 cann-x.y/set_env.sh

pip install torch==<ver>
pip install torch-npu==<ver>     # 必须与 torch、CANN 对齐
pip install -r requirements.txt
pip install -e .
```

版本号请以当前机器文档为准：  
https://www.hiascend.com/developer/download/community/result?module=cann  
https://github.com/Ascend/pytorch

### 环境变量（最小集）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
# 可选：减少碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

自检：

```bash
python - <<'PY'
import torch, torch_npu
print(torch.__version__, torch_npu.__version__)
print('npu?', torch.npu.is_available(), 'count', torch.npu.device_count())
x = torch.randn(2, 2).npu()
print(x.device, (x @ x).sum().item())
PY
```

## 3. 算子要求（相对 HSTU 更轻）

### 3.1 必需（TIGER train / generate）

用 `python -m tiger_ascend.ops.probe --device npu` 逐项验证：

| 算子 / 原语 | 用途 | 失败时影响 |
|---|---|---|
| Tensor alloc / H2D / D2H | 张量与搬运 | 无法上卡 |
| `aten::embedding` | token embedding | 无法建模型 |
| `aten::addmm` / `linear` / `matmul` | QKV、FFN、lm_head | 前向失败 |
| `aten::bmm` | attention | 前向失败 |
| `aten::softmax` | attention 归一化 | 前向失败 |
| `aten::gelu` 或 `relu` | T5 FFN（本仓小配置默认 relu） | 前向失败 |
| LayerNorm / T5LayerNorm | 各 block 归一化 | 前向失败 |
| `aten::dropout` | 训练 | 训练失败 |
| `cross_entropy` + backward | teacher forcing | 无法训练 |
| SDPA 或手工 attention | self/cross-attn | 前向失败 |
| `torch.autocast(device_type="npu")` | fp16/bf16（可选但建议） | 只能 FP32，吞吐差 |

`transformers` T5 在 NPU 上通常走以上 ATen 映射；**不要求**自定义 AscendC 融合算子即可先跑通。

### 3.2 非必需（本仓第一阶段不要装也能改）

| 组件 | 说明 |
|---|---|
| RecSDK / `hstu_paged` | HSTU 专用 |
| FBGEMM NPU / jagged ops | TorchRec 稀疏路径 |
| FlashAttention | 可选加速 |
| ACLGraph | 可选图模式优化 |
| NNAL `_npu_reshape_and_cache` | HSTU KV cache 路径 |

### 3.3 解码侧说明

受限 beam search 的 **Trie / prefix_allowed_tokens_fn 在 Host（CPU）**；NPU 只跑逐步 `forward`。  
因此 Trie 本身 **不是 NPU 算子依赖**。

### 3.4 RQ-VAE（完整 TIGER 流水线第二阶段）

若从 item embedding 生成 semantic id，额外常见依赖：

| 项 | 要求 |
|---|---|
| `Linear` / `MSE` / codebook lookup | 通用 NPU 算子即可 |
| KMeans 初始化 | 可放 CPU（sklearn） |
| Sinkhorn（若使用） | 需确认 `exp` / `sum` 归约在 NPU 正常 |

本仓 toy 模式 **跳过 RQ-VAE**，直接写好 `semantic_ids.json`。

## 4. 多卡（可选）

| 项 | 要求 |
|---|---|
| 集合通信 | HCCL（不是 NCCL） |
| 启动 | `torchrun` + `ASCEND_RT_VISIBLE_DEVICES` |
| 后端 | `backend="hccl"` |

第一阶段建议先单卡 `device=npu` 跑通，再上分布式。

## 5. 建议适配顺序

```text
CPU toy (本仓 --device cpu)
  → ops.probe --device npu 全绿
  → train_tiger.py --device npu
  → 换真实 inter.json + semantic_ids.json
  →（可选）AMP / 多卡 / 更大 T5 配置
  →（可选）接官方 T5Tokenizer + 预训练 config
```

## 6. 常见失败

| 现象 | 排查 |
|---|---|
| `torch.npu.is_available()==False` | CANN 未 source、驱动未挂载、torch/torch_npu/CANN 版本不一致 |
| 某 Aten op `ACL_ERROR` / not implemented | 对照 `ops.probe` 定位；降 transformers 版本或改手写 attention |
| autocast 报错 | 先关 `--amp` 用 FP32 |
| generate 极慢 | 正常：受限 decoding 多步 host-device；先保证正确性 |
| 与 HSTU 环境冲突 | TIGER 不必加载 RecSDK so；分开 conda/容器更干净 |

## 7. 参考链接

- 本仓算子探测：`python -m tiger_ascend.ops.probe --device npu`
- torch-rechub TIGER：https://github.com/datawhalechina/torch-rechub
- Ascend PyTorch：https://github.com/Ascend/pytorch
- CANN 下载：https://www.hiascend.com/developer/download/community/result?module=cann
- （对照）HSTU 官方样例：https://gitcode.com/cann/cann-recipes-infer/tree/master/models/hstu
