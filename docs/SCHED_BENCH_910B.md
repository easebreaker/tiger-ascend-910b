# 多芯调度实验台（TIGER / Ascend 910B）

面向组内「多芯调度」方向：用 **可参数指定的分配策略** 跑完一组可汇报实验，自动采集 **P50/P90/P99、加速比、峰值显存、分段耗时、卡时浪费**。

入口：

- `scripts/sched_bench.py` — 单次策略跑测
- `scripts/run_sched_bench.sh` — 一键矩阵（single / overalloc / dp + tp|pp|ep|sp 边界）
- `scripts/summarize_sched_bench.py` — 汇总 `summary.csv` / `summary.md`

## 策略（`--strategy`）

| 策略 | 含义 | 调度实验目的 |
|---|---|---|
| `single` | 单卡基线 | 吞吐 / 显存 / step 延迟画像 |
| `dp` | DDP 数据并行 | 加速比与并行效率（强/弱缩放） |
| `overalloc` | torchrun N 进程，仅 rank0 训练 | **多占卡浪费**（负向调度案例） |
| `tp`/`pp`/`ep`/`sp` | **不实现**，写 boundary JSON | 小模型不适用模型并行的边界结论 |

TP/PP/EP/SP 不会空跑假实现；输出结构化「不适用」报告，避免虚假工作量，同时保留可引用的边界结论。

## 指标

每个成功 run 写一份 JSON（`artifacts/sched_bench/...`），含：

- `step_ms`: mean / **p50 / p90 / p99** / min / max（去掉 `--warmup_steps`）
- `samples_per_sec` / `steps_per_sec`
- `peak_mem_bytes` / `peak_mem_gb`（`torch.npu.max_memory_allocated`）
- `phase_s`: `data` / `init_model` / `train`
- `wall_s` / `card_seconds`（`wall × nproc`）
- `speedup_vs_baseline`（相对 `--baseline_json` 的 single）
- `parallel_efficiency`（DP：speedup/nproc；overalloc：同）
- `extra.waste_factor`（overalloc：nproc/speedup）

## 一键跑矩阵

```bash
# 选空闲卡（例：4 张）
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
bash scripts/run_sched_bench.sh          # 默认 50 step
# 或
STEPS=30 bash scripts/run_sched_bench.sh --quick
```

脚本会：

1. 写出 `tp/pp/ep/sp` boundary JSON  
2. 跑 `single` 基线  
3. 若可见卡 ≥2/4：跑 `overalloc` 与 `dp`（fixed + weak）  
4. 生成 `summary.md` / `summary.csv`

手动单条：

```bash
source scripts/ascend_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
python -u scripts/sched_bench.py --strategy single --device npu --steps 30 --tag baseline

ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  scripts/sched_bench.py --strategy dp --device npu --steps 30 \
  --global_batch_mode fixed --target_global_batch 32 \
  --baseline_json artifacts/sched_bench/<run>/single_n1_baseline.json
```

## 汇报可用结论模板

1. **单卡画像**：峰值显存 ≪ 单卡容量 → 无「必须切模型」动机。  
2. **overalloc**：speedup≈1 而 card_seconds×N → 小任务多卡独占浪费。  
3. **DP**：给出 2/4 卡加速比与效率；效率随卡数下降则说明通信占比高。  
4. **边界**：TP/PP/EP/SP 对 ~5M T5 + 短 SID **不适用**（已用 footprint + 策略枚举固化）。

## 注意

- 必须经 `ascend_env.sh`（driver lib64 置顶）；不要对 NPU 调 `set_device`（本 bench 已避开）。  
- 不要与 `vllmworker-tp` 抢同一物理卡。  
- NPU 上 `num_workers` 保持 0。  
- 本 bench 测的是 **训练 step 吞吐与调度分配**，不是 Beauty 全量 Hit@K（全量测算仍用 `run_xlt_beauty_910b.sh`）。
