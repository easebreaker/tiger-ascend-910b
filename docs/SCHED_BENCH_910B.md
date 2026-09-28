# 多芯调度实验台（推理优先 / Ascend 910B）

面向组内「多芯调度」：**以推理服务为主**（延迟分位、QPS、占卡浪费、多副本扩展），训练仅可选短画像。

> 旧版以训练 step 为主；现已改为推理优先。全量 Beauty Hit@K 仍用 `run_xlt_beauty_910b.sh`，与本 bench 分离。

## 为什么偏推理

调度日常面对的是 **持续占卡的 serving**（含与 vLLM-TP 共机），不是偶发训练。小模型 TIGER 的正确扩展是 **多副本完整模型**，不是 TP/PP。

## 策略（`--strategy`）

| 策略 | 含义 | 调度结论 |
|---|---|---|
| `single` | 单卡推理基线 | P50/P90/P99、QPS、峰值显存 |
| `replicas` | N 卡各一份完整模型，请求分片 | 小模型水平扩展加速比/效率 |
| `overalloc` | 占 N 卡仅 1 卡服务 | **配额浪费** |
| `tp`/`pp`/`ep`/`sp` | 边界 JSON（不实现） | 本负载不适用模型并行 |
| `train_profile` | 可选短训练画像 | 次要；默认矩阵不跑 |

## 采集指标

- `latency_ms`: **p50 / p90 / p99**（请求 batch 耗时）
- `qps` / **`aggregate_qps`**（集群有效吞吐）
- `peak_mem_gb`、`phase_s`（data / init_model / infer）
- `card_seconds`、`speedup_vs_baseline`、`parallel_efficiency`
- `extra.waste_factor`（overalloc）

## 一键矩阵

```bash
git pull
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7   # 空闲卡
# 可选：用已训权重更贴近业务
# export CKPT=artifacts/ckpt_beauty_xlt/pytorch_model.bin
bash scripts/run_sched_bench.sh
# 或
REQUESTS=32 bash scripts/run_sched_bench.sh --quick

# 若也要附带训练短画像：
TRAIN_PROFILE=1 bash scripts/run_sched_bench.sh --quick
```

输出目录：`artifacts/sched_bench/<timestamp>/summary.md`

## 与业务怎么对齐

1. 训练出 ckpt（可另开全量 Beauty）→ 业务指标  
2. 用同一 ckpt 跑本矩阵 → **调度策略数据**（单卡 vs 多副本 vs 多占卡浪费）  
3. 汇报：推荐「小模型默认 1 卡；要吞吐用 replicas；禁止对本负载开 TP/PP；避免 overalloc」

## 手动单条

```bash
source scripts/ascend_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
python -u scripts/sched_bench.py --strategy single --device npu \
  --requests 64 --batch_size 8 --infer_op generate --beam_size 10

ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  scripts/sched_bench.py --strategy replicas --device npu --requests 128 \
  --baseline_json artifacts/sched_bench/<run>/single_n1_baseline.json
```
