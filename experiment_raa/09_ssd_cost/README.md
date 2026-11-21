# SSD KV 回迁与重算成本

沿用 07 已证实命中的 SSD-LRU 配置：Qwen2.5-1.5B-Instruct FP16、单序列、64 token/块、GPU 32 MiB、CPU 56 MiB。`restore` 回迁连续前缀；`recompute` 仅在探测请求的原生三级查找中跳过 SSD，仍加载连续 CPU 前缀并保留 GPU 命中。准备阶段和写入始终沿用原生 LRU。`gate` 在未见首块上运行冻结的成本表；表外几何保留原生回迁。

## 运行

从 Linux / WSL 项目根目录执行。需有 07 两轮的 `complete.json`、`review.json`、计划、服务配置及请求审计。若 07 产物在其他 `.agent/` 路径，用 `--source` 指定。每次独立回放使用新的输出目录与独立缓存；默认服务端口 `18089`，可用 `--port` 修改。

```bash
source .env.webui
OUT=.agent/kv-ssd-cost
RUN=experiment_raa.09_ssd_cost.run_cost
REVIEW=experiment_raa.09_ssd_cost.review_cost

flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$RUN" smoke --output "$OUT/smoke"
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$RUN" calibration --output "$OUT/calibration"
"$PYTHON" -m "$REVIEW" "$OUT/calibration"
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$RUN" heldout \
  --output "$OUT/heldout" --profile "$OUT/calibration/profile.json"
"$PYTHON" -m "$REVIEW" "$OUT/heldout"
```

校准保留 07 两轮均已证实命中的原始复访（1 和 8 个 SSD 块），另测 2 和 4 块与混合 GPU/CPU 前缀。配对复核要求模型收到相同 token、匹配相同层级与候选块、生成相同 token ID。拒绝分支查验 SSD 提升、提交和完成均为零，且已有 CPU 前缀完成 DMA。门槛以每种精确几何至少 3 对中的服务端 TTFT 配对差中位数为依据；回迁需快过重算 0.5 ms。保留单独的客户端 TTFT，不参与拟合。

## 证据与计时

- `plan.json` 固定所有输入；各臂的 `audit.jsonl`、`requests.jsonl`、`*-before.prom` 和 `*-after.prom` 可复核。`complete.json` 只在各臂实测和配对验收后写入；`review.json` 为离线复核结果，`profile.json` 为冻结门槛。
- SSD→CPU：当前请求原生 `O_DIRECT` 读作业的运行时间之和；另报各作业从提交到调度器观察完成的时间之和，包含线程排队与轮询。均不计文件写入；有重叠时累计值可大于墙钟时间。
- CPU→GPU：vLLM 原生 CUDA event 计时，累计本请求的 DMA；混合前缀中包含已在 CPU 的块和 SSD 提升块，无法拆分为独立 DMA 时延。
- 实际 I/O：请求前至写入和读取都排空时的引擎 Linux `read_bytes` / `write_bytes` 差，以及来宾块设备的扇区差。进程可能有其他文件 I/O；来宾块设备计数包含共享活动。两者均不代表物理 NVMe 总线计数。WSL 的 O_DIRECT 证据限于 ext4 来宾层，需另外核对 VHDX 的宿主介质。
