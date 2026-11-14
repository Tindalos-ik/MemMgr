# KV 分层缓存压力实验

在独占的 WSL / Linux GPU 服务上比较 GPU-only、GPU + 512 MiB CPU、GPU + 64 MiB CPU 三种配置。固定 Qwen2.5-1.5B-Instruct，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，使用现有 `experiment_raa/requirements.txt` 的环境，并安装 `psutil==7.2.2`。

## 运行

将 `tier_baseline.py`、`review_tier.py` 与已有 `prefix_baseline.py` 放在同一目录。`MODEL` 指向上述模型 revision 的本地快照，`RUNTIME` 指向可写的模型与计算缓存目录。

```bash
python tier_baseline.py --model "$MODEL" --runtime "$RUNTIME" \
  --output /var/tmp/kv-tier-results --repeats 5
python review_tier.py /var/tmp/kv-tier-results
```

脚本与已有客户端分开存放时，通过 `--client-dir experiment_raa` 指定客户端目录。默认端口为 `127.0.0.1:18081`；端口占用或输出目录已存在时退出。`--pilot` 仅运行一次 1024-token、间隔 6 条请求的 GPU-only / CPU512 检查。

短前缀补测使用相同的测量器，分别测试 64、128 token；复访前插入两条 3072-token 请求：

```bash
python short_prefix.py --model "$MODEL" --runtime "$RUNTIME" \
  --client-dir experiment_raa --output /var/tmp/kv-short-results
python review_tier.py /var/tmp/kv-short-results --short
```

两种实验分别运行并独占 GPU。主实验和补测的干扰请求长度不同，报告分别保存。

复算会核对逐请求输入、用量、GPU / CPU 命中、实际计算量、迁移字节数和生成文本。文本一致性检查失败时，保存报告及差异位置并以非零状态退出。

需要调查输出差异时，可重新回放某轮原始序列并记录逐 token logprobs。诊断数据单独保存，不参与性能汇总：

```bash
python diagnose_output.py --model "$MODEL" --runtime "$RUNTIME" \
  --trace /var/tmp/kv-short-results/trace-0.json --output /var/tmp/kv-logprobs
```

## 参数与工作负载

- FP16、eager、单序列，上下文上限 3584 token，输出固定 32 token，temperature 0。
- GPU KV 为 64 个 64-token 块，名义容量 112 MiB；扣除空块可用 4032 token。各组统一使用 `LBHNC` 布局。
- CPU 配置使用 vLLM 0.29.0 原生 `OffloadingConnector`，LRU，prompt-only 卸载。64-token GPU 块使该模型使用 DMA 传输路径。
- 该模型 FP16 KV 每 token 为 28672 字节，每块 1.75 MiB；64 MiB CPU 预算容纳 36 块（2304 token），512 MiB 预算容纳 292 块（18688 token）。完成的 prompt KV 块即时复制到 CPU，GPU 腾出空间后可按需回迁。
- 热请求输入为 256、1024、3072 token；首次请求与复访之间分别插入 0、2、6 条独立的 1024-token 请求。每个用例使用不同的首块，相同用例的 cold/revisit 输入相同。
- 每轮各条件重复 5 次，条件顺序按种子打乱；同轮各模式读取相同请求序列。第二轮使用新序列并反转模式执行顺序。
- 正式采集前预热三种 prefill 长度、decode 与回迁路径。每条请求结束后等待统计发布，保留异步传输完成时间。

## 输出

- `trace-*.json`：固定请求顺序和原始输入 token。
- `environment.json`：软件版本、GPU 信息、模型配置、参数和执行源码摘要。
- `e*-*/server.json`、`server.log`：实际服务参数和引擎日志。
- `e*-*/requests.jsonl`：逐请求流式响应、用量、计时和计数器差值。
- `e*-*/initial.prom`、编号 `*.prom`、`final.prom`：完整原始 Prometheus 快照。
- `e*-*/resources.jsonl`：每秒整卡 GPU 状态、进程树 RSS 和系统可用内存。
- `summary.json`：按长度、复访间隔和配置汇总的中位数与配对差值。
- `validated-requests.jsonl`：通过完整性核对的请求及派生字段。

TTFT 从发起 HTTP 请求计至第一个非空正文块。迁移时间使用引擎传输计数器，单独报告；它与计算可能重叠，不能直接与 TTFT 相加。客户端在请求间等待指标发布，本实验不测连续负载吞吐。

CPU 分配预算与 RSS 分别记录。`kv_offload_cpu_cache_usage_perc` 表示在途传输占用比例，不能用来解释 CPU 缓存驻留量。首次 load 之前，对应带 label 的计数样本尚未创建；原始快照保留该状态。

资源采样峰值包含服务启动；RSS 为进程树求和，共享页可能重复统计。

## 参考

[vLLM 0.29.0 KV Offloading](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/features/kv_offloading_usage.md)
