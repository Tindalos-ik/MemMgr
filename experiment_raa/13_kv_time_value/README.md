# CPU KV 前缀按时间价值保留

`kv_cache.policy = "time_value"` 使用原有冻结 GBDT 和阈值作连续前缀准入。`kv_cache.retention_profile` 须为项目 `.agent/` 内的本地校准文件；`model_dir` 指向冻结预测器，回迁保持 `restore = "always"`。盘层只支持 `ssd_write = "eager"`；旧策略默认值保持 `lru`。

候选与受害块的复用概率取该块及所有祖先的冻结分数最小值。加权收益以「缺失后重算」或「已有连续可恢复 SSD 副本」的路径为基线，减去保留在 CPU 并回迁的时间。已驻留 CPU 的块不重复计写入；新块计实测 GPU→CPU DMA，且原生盘层还需要首次写盘时才计盘写。按当前槽和引用状态从叶块开始选择低价值受害块，再选净收益最大的正收益连续前缀。SSD 提升仍走原生路径。

## 首先离线回放

从项目根目录执行，使用 vLLM 0.29.0 与项目同一 WSL Python。历史校准来自实验 06 的独立校准组，历史评价来自实验 08 的 `pressure-14` 两层请求审计。回放执行实际 CPU 管理器，假定每条请求 GPU 冷、写入立即完成；其中的计数只用于反事实比较。

```bash
PYTHON=/path/to/vllm/.venv/bin/python
ROOT=.agent/kv-time-value
MOD=experiment_raa.13_kv_time_value
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MOD.historical" "$ROOT/historical"
```

## 独立校准并测量

每个 `flock` 只运行一个 vLLM 服务。采集器核对继承的 Linux 独占锁，服务进程也持有同一个描述符，停止后释放。输出目录须为新路径。所有组用同一预热、同一冻结评价输入、28 MiB CPU 和 32 MiB GPU KV 配额。默认端口为 18113；可通过 `--port` 选择其他空闲端口。

```bash
D="$ROOT/run-13"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MOD.run" prepare "$D"
for arm in recompute cpu; do
    flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$MOD.run" \
        calibration "$D" --tier cpu --arm "$arm" || exit
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MOD.run" fit "$D" --tier cpu
for arm in admission learned time_value; do
    flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$MOD.run" \
        eval "$D" --tier cpu --arm "$arm" || exit
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MOD.review" "$D" --tier cpu
```

要测 eager 盘层，校准循环依次用 `recompute cpu ssd`，其余命令使用 `--tier ssd`。校准和评价目录按层级隔离，两个层级可以使用同一个冻结计划。每个对照组独立启动，使用空盘目录。若有组中断，将未生成 `complete.json` 的组目录移出正式组路径，保留原始证据，并用原命令重采该组。

已有 CPU 结果时保留 `D` 中的冻结 `plan.json` 和 `experiment.json`，直接运行 SSD 阶段。SSD 组把子进程继承的独占锁写入 `gpu-lock.json`。拟合前核对三臂逐 token、实机原始指标和从盘写完成、CPU 槽淘汰、盘读入槽到 GPU DMA 的同块链；复核证据为 `review-ssd-calibration.json`。`time_value` 评价额外从按时间排序的盘写完成事件复算每个已发布副本的成本分支，`review-ssd.json` 中保留能定位到原始作业和决策时点的样本。

## 口径和适用范围

- 校准测 1、4、8、16 块、GPU 冷起点的连续前缀；同一个探测输入跨路径逐 token 校验。固定深度的累计收益使用服务端 TTFT 的配对差中位数，锚点之间由相邻累计差的斜率给出边际成本。写入使用已完成传输计时，每块取各深度的中位数。CPU、SSD 两种 profile 不能混用。
- 校准的首块与 GBDT 历史计划及评价首块互斥。评价为 12 个新前缀、确定性填充和两次复访；它是合成压力回放。固定输出 8 token，串行贪心解码，无抢占。开始加载时校验模型 revision、GPU UUID 与驱动、CPU、WSL 内核、软件和源码摘要、KV 布局与容量；盘层额外校验挂载身份、容量和写入模式。适用深度以文件锚点为限，不向外推。
- 成本模型以 GPU 冷起点、不同前缀深度的实测作为近似；不预测未来 GPU 驻留，不校准混合 CPU/SSD 段的每种几何。复用分数、DMA/盘写时间与 TTFT 的估计量不能当成实际端到端收益。
- `requests.jsonl` 和 Prometheus 快照保存实际 token ID、两侧 TTFT 和块传输。`audit.jsonl` 证明淘汰、计价和实际存储；`resources.jsonl` 的 Linux 进程字节含其他文件 I/O。`review-{cpu,ssd}.json` 核对对应输出 ID、策略准入与实际写入，按同一请求分别计算全体及复访的处理组减对照组中位数和最近秩 P95。
