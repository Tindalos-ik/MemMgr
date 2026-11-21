# 实验 12：SSD 写入时机

| 参数 | 固定值 |
| --- | --- |
| 模型与 CPU 策略 | 实验 07 的 Qwen2.5-1.5B-Instruct、冻结 GBDT `learned` |
| KV 容量 | GPU 32 MiB、CPU 14 MiB、SSD 4 GiB；完整块 64 token |
| SSD 回迁 | `restore: "always"` |
| 请求 | 同一条预热；07 的冻结 78 条（在 08 的 14 MiB learned 组无读回），08 的 50 条热点压力 |
| 对照顺序 | 每份计划各两轮：`eager → on_evict`、`on_evict → eager`；各组独立进程与空盘目录 |

实测数据见 [RESULTS.md](RESULTS.md)。

## 配置

在服务的 `kv_cache` 中设置 `tier: "ssd"`、`batch_invariant: true`、`restore: "always"`，可用 `ssd_write` 选择时机：

- `"eager"`（默认）：GPU→CPU 完成即提交异步盘写；SSD I/O 完成前持有原生 CPU 读引用。
- `"on_evict"`：CPU 达到容量且选择牺牲块时才写，包括 SSD 提升占用的槽。对尚未入盘的块直接从原槽完成写入，再允许 CPU 重用槽；服务关闭只排空已发起的作业，仍驻留的块不补写。

已完成盘写的块无需重复写。`ssd_root` 必须支持 `O_DIRECT`；同一缓存目录只允许一个写入者。CPU 和 GPU 单层模式不接受 `on_evict`。

可单独运行原槽字节、回迁和故障传播检查，检查无需启动 GPU 服务：

```bash
.venv/bin/python -m unittest tests.integration.test_ssd_writeback
```

## 采集与复核

在项目根目录的 WSL shell 中运行，默认读取 `.agent/kv-joint-ssd/output` 和 `.agent/kv-capacity/pressure-14` 的冻结工件。`OUT` 初次使用时须不存在。每次获得共享锁只运行一个服务组，服务停止后释放锁。已有完成标记的组会被跳过；中断留下的不完整组应保留排查，重新采集用新目录。默认端口 `18092`，可用 `--port` 指定空闲端口。

```bash
set -e
OUT=.agent/kv-ssd-write/paired-12
while [ ! -f "$OUT/complete.json" ]; do
  flock -x .agent/kv-experiment-gpu.lock .venv/bin/python \
    experiment_raa/12_ssd_write/run_ssd_write.py --output "$OUT"
done
.venv/bin/python -m experiment_raa.12_ssd_write.review_ssd_write "$OUT" > "$OUT/review.json"
sha256sum "$OUT"/{environment,complete,review}.json
```

- **写入量**：逻辑写按成功作业唯一块数计，逐组核对 `.bin` 文件数及字节。实际 I/O 用排空前后 Linux 服务进程树的 `write_bytes`/`read_bytes` 差，包含审计等文件 I/O；不代表设备 NAND 写入量。
- **回迁与错失**：只统计写盘成功、CPU 淘汰、SSD→CPU 读完、同一请求 CPU→GPU 完成的块。盘查询未命中单列；`eager` 已回迁且 `on_evict` 未从任何层命中的同请求块按块在前缀中的位置配对。额外重算既给出净差，也累加各请求的正差。
- **容量与等待**：CPU 峰值包含已预约的槽，SSD 写源槽峰值包含异步在途和同步交接。等待为调度线程实际阻塞秒数，与原始 Prometheus 计数核对。
- **时延与输出**：逐请求核对生成 token ID。TTFT 分客户端和服务端，报告中位数和最近秩 P95；配对差先逐请求计算 `on_evict − eager`。
