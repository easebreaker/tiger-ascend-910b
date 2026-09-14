# Ascend 910B 生成式推荐部署笔记

- **HSTU / Meta GR**：本文下方 checklist（有昇腾官方样例）
- **TIGER（语义 ID + T5）**：[docs/TIGER_910B_DEPLOY.md](docs/TIGER_910B_DEPLOY.md)（无官方一键样例，需自行迁 NPU）

---

# Ascend 910B 上部署 HSTU（生成式推荐）Checklist

基于 [cann-recipes-infer/models/hstu](https://github.com/hicann/cann-recipes-infer/tree/master/models/hstu) 官方样例整理。  
硬件目标：**Atlas A2 / Ascend 910B**；软件目标：**CANN 8.5.0 + RecSDK-Torch**。

> 说明：单独的 `hstu_processor.py` 不能完成部署。需要整套 HSTU 样例代码、CANN/NPU 算子环境，以及 `recsys-examples` 基线代码。

---

## 0. 部署前确认

| 项 | 要求 |
|---|---|
| 硬件 | Atlas A2 系列（910B），单卡或多卡 |
| 驱动 | 主机已安装 Ascend 驱动，`npu-smi` 可用 |
| CANN | **8.5.0**（与样例绑定，勿随意换大版本） |
| 算子包 | `Ascend-cann-910b-ops_8.5.0`（不是 A3-ops） |
| 加速库 | `Ascend-cann-nnal_8.5.0` |
| 基础镜像 | AscendHub **rec_sdk-torch** Docker |
| 数据盘 | 建议准备 `/data`，用于代码、权重、数据集 |

主机快速自检：

```bash
npu-smi info
ls /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
```

---

## 1. 拉起 RecSDK-Torch 容器

1. 从 [AscendHub rec_sdk-torch](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5) 下载镜像。
2. 查看本地镜像名：

```bash
docker images | grep rec_sdk-torch
```

3. 启动容器（按机器卡数增减 `--device=/dev/davinciN`）：

```bash
docker run -u root -itd --name rec_gr --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/:/home \
    -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /root/.pip:/root/.pip \
    -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host \
    --shm-size=128g \
    --privileged \
    $REPOSITORY:$TAG \
    /bin/bash
```

4. 进入容器并设置 Python：

```bash
docker exec -it rec_gr bash
export PATH=/usr/local/python3.11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib:$LD_LIBRARY_PATH
```

---

## 2. 安装 CANN 8.5.0（容器内）

从 [CANN 8.5.0 下载页](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0) 获取：

- `Ascend-cann-toolkit_8.5.0_linux-${arch}.run`
- `Ascend-cann-910b-ops_8.5.0_linux-${arch}.run`
- `Ascend-cann-nnal_8.5.0_linux-${arch}.run`

`${arch}` 为 `aarch64` 或 `x86_64`。安装参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html)。

```bash
# toolkit + 910B ops（按实际包名执行）
bash Ascend-cann-toolkit_8.5.0_linux-${arch}.run --install
bash Ascend-cann-910b-ops_8.5.0_linux-${arch}.run --install
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# nnal（KV Cache 依赖 _npu_reshape_and_cache）
bash Ascend-cann-nnal_8.5.0_linux-${arch}.run --install
source /usr/local/Ascend/nnal/atb/set_env.sh
```

建议写入 `~/.bashrc`，避免每次进容器丢环境。

---

## 3. 准备代码

样例依赖 NVIDIA `recsys-examples` 基线 + Ascend patch。

```bash
mkdir -p /data/code
cd /data/code

# 1) Ascend 推理样例
git clone https://gitcode.com/cann/cann-recipes-infer.git
# 若用 GitHub 镜像：https://github.com/hicann/cann-recipes-infer.git

# 2) 拷贝 HSTU 基线（非覆盖，保留 Ascend 侧已有文件）
git clone --branch v25.11 --depth 1 https://github.com/NVIDIA/recsys-examples.git
cp -an recsys-examples/examples/hstu/* cann-recipes-infer/models/hstu/

# 3) 应用 NPU 适配 patch
cd cann-recipes-infer/models/hstu
git apply hstu_cann.patch
```

完成后应能看到：`hstu_processor.py`、`inference/`、`hstu_cann.patch`、`build_install_ops.sh` 等。

---

## 4. 编译并安装 RecSDK 算子

### 4.1 拉取 RecSDK

```bash
cd /data/code/cann-recipes-infer/models/hstu
git clone https://gitcode.com/Ascend/RecSDK.git
```

### 4.2 准备外部依赖

1. 下载 [json v3.9.1](https://github.com/nlohmann/json/archive/v3.9.1.tar.gz)，**重命名为 `v3.9.1.tar.gz`**，放到：

```text
RecSDK/cust_op/ascendc_op/build/scripts/onnx_plugin/
```

2. 下载并解压 [catlass v1.3.0](https://raw.gitcode.com/cann/catlass/archive/refs/heads/v1.3.0.zip)，设置：

```bash
export CATLASS_HOME=<解压后的 catlass 路径>
```

### 4.3 适配 `concat_nd_jagged`（多 jagged concat）

修改：

```text
RecSDK/cust_op/framework/torch_plugin/torch_library/concat_2d_jagged/concat_jagged_tensor.cpp
```

按官方 README 增加 `concat_nd_jagged_npu` 实现，并在 `TORCH_LIBRARY_FRAGMENT(mxrec, m)` / `TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)` 中注册。  
（完整补丁片段见上游 `models/hstu/README.md`。）

### 4.4 一键编译安装（A2 / 910B）

```bash
cd /data/code/cann-recipes-infer/models/hstu
chmod +x build_install_ops.sh
bash build_install_ops.sh A2 ./RecSDK
```

### 4.5 Python 依赖

```bash
pip uninstall torchrec -y
pip install torchrec==1.1.0
pip install rich einops

export LIB_FBGEMM_NPU_API_SO_PATH=/usr/local/python3.11.0/lib/python3.11/site-packages/libfbgemm_npu_api.so
```

---

## 5. 数据与推理

### 5.1 准备数据集（样例：KuaiRand-1K）

```bash
cd /data/code/cann-recipes-infer/models/hstu
python3 ./preprocessor.py --dataset_name "kuairand-1k" --inference
```

### 5.2 单卡推理

```bash
# eval
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --mode eval \
  --enable_fused_ops all

# profiling
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --mode eval \
  --enable_fused_ops all \
  --enable_profiler

# simulate
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --mode simulate \
  --enable_fused_ops all
```

### 5.3 单机多卡（示例：2 卡）

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:6000 \
  --nnodes=1 \
  --nproc-per-node=2 \
  ./inference/inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --embed_tp_size 2 \
  --mode eval \
  --enable_fused_ops all
```

---

## 6. 验收标准（建议按序勾选）

- [ ] `npu-smi info` 能看到 910B
- [ ] `source .../cann-8.5.0/set_env.sh` 与 nnal `set_env.sh` 成功
- [ ] `import torch; import torch_npu` 无报错
- [ ] `LIB_FBGEMM_NPU_API_SO_PATH` 指向的 `.so` 存在且可 `load_library`
- [ ] `hstu_cann.patch` 已应用，无冲突残留
- [ ] `build_install_ops.sh A2` 编译通过
- [ ] `preprocessor.py` 数据准备完成
- [ ] `inference_gr_ranking.py --mode eval` 跑通并打出指标/日志

---

## 7. 常见卡点

| 现象 | 排查方向 |
|---|---|
| 找不到 NPU / davinci 设备 | 驱动未挂载进容器；检查 `--device` 与 driver 卷 |
| 算子缺失 / aclnn 报错 | 是否装了 **910b-ops**；CANN 版本是否 8.5.0 |
| KV Cache / reshape_and_cache 失败 | nnal 未装或未 `source set_env.sh` |
| `load_library` FBGEMM 失败 | `LIB_FBGEMM_NPU_API_SO_PATH` 路径不对 |
| jagged concat 维度/数量报错 | 未按 README 改 `concat_nd_jagged_npu` |
| `build_install_ops.sh` 失败 | `CATLASS_HOME`、json tarball 命名/路径 |
| 显存不足（大 embedding） | 改用多卡 + `--embed_tp_size` |

---

## 8. 这个样例实际做了什么（便于排障）

相对 NVIDIA `recsys-examples`，Ascend 侧主要改动：

1. TensorRT-LLM 风格 KV Cache 管理改为 NPU 实现  
2. Triton/CUDA 算子改为 torch 小算子或 NPU 融合算子  
3. 使能：`hstu_paged`、`in_linear_silu`、`concat_nd_jagged`、ACLGraph、`_npu_reshape_and_cache`  
4. 支持 embedding 切分的分布式推理与 KV offload / 通算掩盖  

`modules/hstu_processor.py` 只是其中的输入处理模块（jagged 拼接、位置编码、MLP 等），部署时必须和 inference 入口、融合算子、配置文件一起用。

---

## 9. 推荐执行顺序（最短路径）

```text
主机确认 910B + 驱动
  → 拉起 rec_sdk-torch 容器
  → 装 CANN 8.5.0 toolkit + 910b-ops + nnal
  → clone cann-recipes-infer + recsys-examples，cp -an，apply patch
  → clone RecSDK，放好 json/catlass，改 concat，build_install_ops.sh A2
  → pip 依赖 + FBGEMM 环境变量
  → preprocessor 准备数据
  → inference_gr_ranking.py --mode eval
```

---

## 参考链接

- 样例 README：https://github.com/hicann/cann-recipes-infer/blob/master/models/hstu/README.md  
- 官方仓（GitCode）：https://gitcode.com/cann/cann-recipes-infer  
- RecSDK：https://gitcode.com/Ascend/RecSDK  
- NVIDIA 基线：https://github.com/NVIDIA/recsys-examples（tag `v25.11`）  
- CANN 8.5.0 下载：https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0  
