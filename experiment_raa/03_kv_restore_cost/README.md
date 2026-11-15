# KV 回迁成本门槛

在 vLLM 原生 CPU offload 路径上，根据独立校准的成本决定读取 CPU KV 或重算。`always`、`never`、`gate` 共用 CPU 容量、prompt 写入、LRU、GPU 容量和 DMA 条件。

## 环境与运行

使用独占的 WSL / Linux NVIDIA GPU，沿用 `experiment_raa/01_start_with_prefix_reuse/requirements.txt`，另需 `psutil==7.2.2`。固定 vLLM 0.29.0、Qwen2.5-1.5B-Instruct FP16，模型 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

从项目根目录执行。`PYTHON` 指向已有实验解释器，`MODEL` 指向固定 revision 的本地快照，`RUNTIME` 指向计算缓存目录。临时审阅期间，将下面的 `EXPERIMENT` 替换为候选目录路径。

```bash
EXPERIMENT=experiment_raa/03_kv_restore_cost
"$PYTHON" "$EXPERIMENT/run_cost.py" \
  --model "$MODEL" --runtime "$RUNTIME" \
  --output /var/tmp/kv-cost-results \
  --calibration-repeats 2 --repeats 3 --margin-ms 0.5

"$PYTHON" "$EXPERIMENT/review_cost.py" /var/tmp/kv-cost-results
```

默认地址为 `127.0.0.1:18082`，可用 `--port` 修改。输出目录必须尚不存在。两种重复次数均至少为 2。运行期间保持源码和 GPU 工作负载不变；入口顺序启动服务，结束或发生异常时停止本次服务进程。

测量器复用 `experiment_raa/02_kv_split_among_gpu_cpu/tier_baseline.py` 的原始指标采集和资源采样。目录分开存放时，使用 `--baseline-dir` 指定该目录。四个 Python 文件需要放在同一目录，入口通过 `PYTHONPATH` 将 connector 提供给服务进程。

## 决策

原生 connector 查找 CPU 命中后，尚未创建 load job 时执行门槛。输入包括总输入 token、GPU 已命中 token、CPU 可恢复 token。拒绝时将外部命中改为零，保留 GPU 命中，后续由原生 prefill 计算缺失部分。CPU 写入和淘汰继续走原生实现。

成本表以 `(总输入, GPU 命中, CPU 候选)` 为键，分别保存 `always` 与 `never` 的服务端 TTFT 中位数。这两条路径覆盖传输、调度、首 token 计算及未恢复尾部的计算；传输计数器耗时单独保存。相同键固定上下文长度、剩余未命中长度和可恢复量。

当 `重算路径成本 − 回迁路径成本 > margin_ms` 时回迁；相等时重算。`margin_ms` 为评价前指定的最小预计节省，默认 0.5 ms。

校准阶段按正反顺序执行 `always / never`；冻结成本表后，用全新的首块执行两轮 `always / never / gate`，第二轮反转顺序。评价期间不更新成本。未校准的候选几何直接报错；当前表用于受控工作负载。模型、GPU、依赖、原生调度源码摘要和缓存参数随表保存，入口检查校准与评价环境相同。

## 工作负载

| 已存前缀 token | 复访输入 token | 中间干扰 | 检查目标 |
|---:|---:|---|---|
| 64、128、256、1024、3072 | 与已存前缀相同 | 两条独立的 3072-token 请求 | 完全淘汰后的不同回迁量 |
| 256、3072 | 与已存前缀相同 | 无 | 已有 GPU 命中，仅余少量缺失 |
| 1024 | 1536 | 两条独立的 3072-token 请求 | CPU 恢复前缀后继续计算 512-token 新尾部 |

GPU KV 使用 64 个 64-token 块，名义容量 112 MiB；CPU 为 512 MiB，布局为 `LBHNC`。单 worker、单序列、eager，上下文上限 3584，默认输出 32 token。三种策略均采集逐 token logprobs。

## 输出与复核

- `environment.json`：实际参数、依赖、GPU、模型及执行源码摘要。
- `calibrate-trace-*.json`、`evaluate-trace-*.json`：两阶段固定输入与请求顺序。
- `calibration.json`：冻结成本表及校准观测来源。
- 各服务目录的 `server.json`、`server.log`：启动参数和原始服务日志。
- `requests.jsonl`、编号 `*.prom`：逐请求输出、客户端计时、用量及原始指标快照。
- `decisions.jsonl`：引擎请求标识、GPU 命中、CPU 候选、准入量、估计成本、决定原因和成本表摘要；包含预热请求。
- `resources.jsonl`：GPU 状态与进程树 RSS 采样。
- `summary.json`、`validated-requests.jsonl`：复核后统计及逐请求派生字段。

复核器从原始快照重新计算差值，检查实际回迁字节数、外部命中、计算量、拒绝后的零传输、GPU 命中保留、相同输入的候选量、冻结成本与实际决定，并确认校准和评价首块不相交。正式评价必须覆盖准入和拒绝两条路径。

`remaining_tokens` 按 `输入 − GPU 命中 − 准入 CPU token` 记录，与 vLLM 的 prefill KV 未命中统计核对。完全恢复后，引擎仍需执行生成首 token 所需的计算；路径成本通过服务端 TTFT 包含这部分开销。

`paired_saved_ms` 按同轮同请求计算 `always TTFT − 当前策略 TTFT`，随后取中位数；正值表示当前策略更快。`epoch_paired_saved_ms` 保留两轮分别的结果。输出 token 或文本存在差异时保存差异位置和统计，并以非零状态退出；逐请求 logprobs 用于进一步检查。

请求之间等待指标发布，延迟统计适用于本次串行回放。成本表与当前模型、设备及缓存条件关联，迁移到其他环境时重新校准。

## 接口参考

- [vLLM 0.29.0 OffloadingConnector](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)
- [vLLM 0.29.0 offloading scheduler](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)
- [vLLM 0.29.0 外部 connector 加载](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/distributed/kv_transfer/kv_connector/factory.py)
