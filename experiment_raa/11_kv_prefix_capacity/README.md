# 实验 11：14 MiB 连续 KV 前缀容量

复用[实验 08 冻结 78 条](../08_kv_capacity/results-14.md)的正式请求和 1,025-token 预热。CPU 配额 14 MiB（8 个 64-token 块），GPU 配额 32 MiB；SSD-LRU 使用单独的空 SSD 目录、4 GiB 配额及 `O_DIRECT`。两组统一使用正式 `ManagedConnector` 与 `always` 回迁。

CPU 写入保留已驻留的连续命中，按空闲槽及可淘汰槽截断待写前缀。GPU 和 SSD 在途引用仍由原生计数保护；实际分配或传输错误会直接报错。

[14 MiB 实测结果与原始工件摘要](RESULTS.md)

## 采集

在 WSL Linux 的项目根目录运行。输出目录须不存在；中断后另选目录保留原始工件。

```bash
OUT=.agent/kv-prefix-capacity-11/replay-14
PYTHON=.venv/bin/python
RUN=experiment_raa.11_kv_prefix_capacity.run_prefix_capacity
"$PYTHON" -m "$RUN" --output "$OUT"
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$RUN" --output "$OUT" --group cpu-lru
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m "$RUN" --output "$OUT" --group ssd-lru
```

每条 `flock` 命令包住一个服务的启动、请求采集和停止。若默认端口 `18111` 被占用，为该组指定空闲的 `--port`。`--historical` 在准备阶段指定实验 08 原始目录，默认 `.agent/kv-capacity/frozen-14`。

## 复核

```bash
"$PYTHON" -m experiment_raa.11_kv_prefix_capacity.review_prefix_capacity "$OUT" > "$OUT/review.json"
sha256sum "$OUT"/{environment,review}.json
```

报告包含逐请求输出 token、连续 CPU 前缀块数、GPU 命中、外层回迁、SSD→CPU→GPU 完整链路、重算量、TTFT 配对差和原始证据清单。正式指标不计预热。实验 08 的 LRU 失败组保留为历史故障证据，无延迟、命中或重算对照值；08 的 admission／learned 指标单列作历史参照。
