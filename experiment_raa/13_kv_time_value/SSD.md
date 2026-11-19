# 实验 13：eager SSD 已发布副本的 CPU 保留价值

复用实验 13 的冻结评价计划、冻结 GBDT、RTX 4050 Laptop GPU、Qwen2.5-1.5B-Instruct FP16、32 MiB GPU 和 28 MiB CPU KV 配额。eager SSD 上限 4 GiB，每组从独立空盘目录开始。三条校准路径均启用 eager SSD，各组按独立的 WSL `flock -x .agent/kv-experiment-gpu.lock` 启停；`gpu-lock.json` 记录服务子进程实际继承的独占锁和 inode。

## 校准证据

冻结计划 `plan.json` 的 SHA-256：`6e0c644c99517b65dcbb817a574b7e9913af3e1c7fa21bd9bef7c210770eb134`。

- 冷重算臂：16 条探测均无 SSD 读回、CPU 回迁和 GPU 命中。复核 17 条请求的原始 Prometheus 指标（含预热）、144 个完成发布的 O_DIRECT 文件和正常关闭记录。原始目录 `.agent/kv-time-value/run-13/calibration-ssd/recompute/`，`complete.json` SHA-256 `9094cea2f534cbbeba0791cd6b7a373cf400d92b27ab1be398c2e34a36fd6235`，`gpu-lock.json` SHA-256 `5ad6e66898157603e36fe28b39c1fc0222baacb5bcb35205a7ede475d2366a5a`。
- CPU 回迁臂：16 条探测共 116 块 CPU→GPU DMA，SSD 读回为零；128 个探测输出 token ID 与冷重算臂逐一一致。复核 33 条请求的原始指标和 144 个发布文件。原始目录 `.agent/kv-time-value/run-13/calibration-ssd/cpu/`，`complete.json` SHA-256 `69c306184222bfd8aa04ade25798230a47a87524a7c7292d6e2f7da055f12b48`，`gpu-lock.json` SHA-256 `a8a183209e6c179abf25eab4c234c5f92a725fe7654505d9747683b0a5406bcf`。
