# SSD KV 恢复实验

使用项目真实的 SciFact、CPU BGE、Chroma 检索与聊天模板，比较 GPU-only、GPU＋CPU、GPU＋CPU＋SSD。两轮使用相同配置，第二轮反转服务执行顺序；每组回放同一轮已保存的完整 token 输入。

## 准备

按项目 `requirements.in` 与 `constraints.txt` 安装依赖，下载固定模型，启动 Chroma，准备数据并建立 small 索引。需要独占的 Linux / WSL NVIDIA GPU，以及支持 O_DIRECT 的本地 SSD 文件系统。

`PYTHON` 指向项目解释器，`MODEL` 指向 Qwen2.5-1.5B-Instruct 的固定快照 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，`RUNTIME` 指向计算缓存与 `hf/` 所在目录。自定义 Chroma 地址时，通过 `--config` 指定对应的项目配置。

## 执行

在项目根目录运行：

```bash
"$PYTHON" experiment_raa/05_kv_ssd_restore/run_ssd.py \
  --model "$MODEL" --runtime "$RUNTIME" \
  --output .agent/ssd-results \
  --ssd-root /本地SSD路径/kv-ssd-results

"$PYTHON" experiment_raa/05_kv_ssd_restore/review_ssd.py .agent/ssd-results
```

结果目录与 SSD 目录须尚不存在。默认服务端口为 `18084`，使用 `--port` 修改。默认每轮重复两次，输出 32 token。入口仅管理自己启动的服务，异常时停止本次服务进程。

## 参数与序列

固定 vLLM 0.29.0、FP16、eager、单 worker、单序列、64-token 块和 `LBHNC`。GPU 名义容量为 64 块 / 112 MiB；CPU 预算为 64 MiB，可存 36 块 / 2304 token。文件层通过 CPU 中转，沿用原生异步写入、提升、DMA 与引用保护。

每个问题检索五个候选块，按项目模板分别使用 512、1024、1536-token 预算组装完整 prompt。请求长度保留组装结果，包含共享系统提示和未填满的尾块。序列覆盖热点复访、扫描干扰、热点切换与长间隔复访。复核器要求连续前缀块工作集超过 GPU 与 CPU 的合计名义容量。

SSD 使用原生 O_DIRECT 读写；文件系统不支持时直接报错。O_DIRECT 绕过 Linux 文件页缓存，结果描述当前文件系统与底层存储路径，不代表断电持久性。

## 文件与复核

- `prompts.json`：实际检索结果对应的消息、证据 ID 与完整输入 token。
- `trace-*.json`：固定请求顺序和输入。
- `environment.json`：依赖、模型、索引身份、GPU、文件系统与执行源码摘要。
- `e*-*/requests.jsonl`、`*.prom`：真实生成、逐 token logprobs、计时与原始指标。
- `e*-*/audit.jsonl`：CPU 写入、淘汰、SSD 任务提交与完成、GPU 回迁完成。
- `e*-*/resources.jsonl`、`disk.json`：进程树实际 I/O、RSS 与缓存文件占用。
- `summary.json`、`validated-requests.jsonl`：从原始记录复算的统计与逐请求计算量。
- `review-metadata.json`：本次复核器与原始环境记录的摘要。

复核逐请求用量、指标边界、输入工作集和输出；SSD 组还要求同一块依次具有 SSD 写入完成、CPU 淘汰、SSD 读取完成、GPU 回迁完成的证据。生成 token 或文本存在差异时保留差异并以非零状态退出。logprobs 的最大差值单列报告。

TTFT 使用客户端首个非空正文块。指标发布等待不计入单次请求计时，结果描述串行回放。两轮分别保留 TTFT 中位数、P95、相对 GPU-only 的配对节省、重算与恢复 token、实际 SSD 读写量。

`fs_*_payload_bytes` 为成功文件层任务的 KV 载荷累计量，包含单独保存的预热请求；`process_read_bytes`、`process_write_bytes` 来自健康进程的内核 I/O 计数，反映实际读写，包含该进程的其他文件 I/O。复核要求实际读取量覆盖已审计的 O_DIRECT 读回载荷。正式请求窗口的逻辑传输量可从 `initial.prom`、`final.prom` 复算。

`requests.jsonl` 的 `client_request_id` 标识采集器请求，审计事件保留引擎实际请求标识。
