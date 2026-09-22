# 在 Ascend 910B 上逐步部署本仓库（TIGER 适配骨架）

目标：在 Atlas A2 / 910B 上跑通  
`算子探测 → toy 训练 → toy 评估`。  
这是 **适配试验仓**，不是生产推荐服务；完整 TIGER 还需要 RQ-VAE 语义 ID 与真实数据。

---

## 0. 你需要准备什么

| 项 | 说明 |
|---|---|
| 机器 | 带 Ascend 910B 的服务器（能跑 `npu-smi info`） |
| 权限 | 能装 CANN / 能进 Docker（二选一） |
| 本仓库 | `https://github.com/easebreaker/tiger-ascend-910b` |
| 软件基线 | CANN 8.5.x + 匹配的 `torch` / `torch_npu` + Python 3.9–3.11 |

> 本仓 **不需要** RecSDK / HSTU 融合算子。

---

## 1. 主机侧：确认 NPU 可见

在 910B 机器上：

```bash
npu-smi info
ls /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
```

能看到 910B 卡信息再继续。若在容器里跑，启动时要挂载这些设备与驱动目录。

---

## 2. 安装 CANN + torch_npu（最关键）

### 2.1 安装 CANN 8.5.x

从昇腾社区下载（版本号按你机器文档微调）：

- `Ascend-cann-toolkit_8.5.x_linux-*.run`
- `Ascend-cann-910b-ops_8.5.x_linux-*.run`

```bash
bash Ascend-cann-toolkit_8.5.x_linux-*.run --install
bash Ascend-cann-910b-ops_8.5.x_linux-*.run --install

# 常见路径二选一
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source /usr/local/Ascend/cann-8.5.0/set_env.sh
```

### 2.2 安装匹配的 PyTorch + torch_npu

**三者版本必须对齐**：CANN ↔ torch ↔ torch_npu。  
以你现场兼容性矩阵为准，示意：

```bash
pip install torch==2.6.0
pip install torch-npu==2.6.0
```

自检：

```bash
python3 - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available(), "count", torch.npu.device_count())
x = torch.randn(2, 3).npu()
print(x.device, (x @ x.T).shape)
PY
```

全部正常再往下。

---

## 3. 拉取本仓库并安装依赖

```bash
git clone https://github.com/easebreaker/tiger-ascend-910b.git
cd tiger-ascend-910b

pip install -e .
pip install -r requirements.txt
# requirements-npu.txt 目前是注释清单；torch-npu 请按上一步单独安装
```

---

## 4. 算子探测（上卡前必做）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 -m tiger_ascend.ops.probe --device npu --out artifacts/op_probe_npu.json
```

期望：`ok: true`。  
若某个算子 `fail`，先把失败项对照 `docs/ASCEND_910B_ENV_AND_OPS.md` 排查，再训练。

一键脚本：

```bash
bash scripts/run_npu_smoke.sh
```

（脚本会尝试 `source` CANN 环境变量；路径不对时请先手动 `source`。）

---

## 5. 跑通 toy 训练 / 评估

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python3 scripts/train_tiger.py \
  --mode all \
  --device npu \
  --epochs 2 \
  --batch_size 8 \
  --toy_users 32 \
  --toy_items 64 \
  --d_model 128 \
  --num_layers 2 \
  --num_heads 4
```

成功标志：

- 打印 `[device] kind=npu ...`
- 每个 epoch 有 `train_loss=...`
- `saved checkpoint -> ./artifacts/ckpt`
- 有 `exact_match=...`（toy + 很少 epoch 时接近 0 **正常**）

可选混合精度：

```bash
python3 scripts/train_tiger.py --mode all --device npu --epochs 2 --batch_size 8 --amp
```

若 `--amp` 报错，先去掉，用 FP32 跑通。

**吞吐提示（默认配置偏“跑通”不是论文 batch）：**

按论文 Beauty 测算请用：

```bash
bash scripts/run_paper_beauty_910b.sh
# 说明：docs/PAPER_BEAUTY_910B.md
```

OOM 时缩小 micro-batch：`bash scripts/run_paper_beauty_910b.sh --batch_size 32 --grad_accum 8`

---

## 6. 换成你自己的数据（第二阶段）

准备两个 JSON（路径用 `--data_dir`）：

**`inter.json`**

```json
{
  "0": [1, 5, 9, 12, 3],
  "1": [2, 4, 8, 8, 7, 1]
}
```

- 每个用户按时间排序的 item id 列表  
- **至少 4 个交互**（leave-one-out：train/valid/test）

**`semantic_ids.json`**

```json
{
  "1": ["<a_0>", "<b_1>", "<c_2>"],
  "5": ["<a_0>", "<b_3>", "<c_1>"]
}
```

- key 必须覆盖 `inter.json` 里出现的所有 item id  
- 本仓 **尚未内置 RQ-VAE**；语义 ID 需离线生成后放入该文件

然后：

```bash
python3 scripts/train_tiger.py \
  --mode train \
  --device npu \
  --data_dir /path/to/your_data \
  --output_dir ./artifacts/ckpt_real \
  --epochs 5 \
  --batch_size 16

python3 scripts/train_tiger.py \
  --mode eval \
  --device npu \
  --data_dir /path/to/your_data \
  --output_dir ./artifacts/ckpt_real
```

---

## 7. 建议验收清单

- [ ] `npu-smi info` 正常  
- [ ] `torch.npu.is_available() == True`  
- [ ] `ops.probe --device npu` → `ok: true`  
- [ ] toy `--mode all --device npu` 跑通  
- [ ]（可选）真实 `inter.json` + `semantic_ids.json` 训练不报错  

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `preferred=npu but Ascend NPU is unavailable` | 未 `source` CANN；或 torch/torch_npu/CANN 版本不匹配 |
| probe 某算子 fail | 对照失败算子名；先 FP32；必要时降 `transformers` 版本 |
| `train dataset is empty` | 用户序列太短（需 ≥4） |
| `semantic_ids.json missing ...` | item id 与语义 ID 表不对齐 |
| `exact_match` 很低 | toy + 少 epoch 正常；真实指标需 Hit@K/NDCG，后续可加 |
| 想做完整论文复现 | 还需补 RQ-VAE、T5Tokenizer 扩词表、更大配置与标准评测 |

---

## 9. 推荐推进顺序

```text
本机/CPU：bash scripts/run_cpu_smoke.sh
    ↓
910B：装好 CANN + torch_npu 自检
    ↓
910B：python3 -m tiger_ascend.ops.probe --device npu
    ↓
910B：toy train/eval
    ↓
准备真实 inter + semantic_ids
    ↓
再谈 RQ-VAE / 多卡 / 吞吐优化
```
