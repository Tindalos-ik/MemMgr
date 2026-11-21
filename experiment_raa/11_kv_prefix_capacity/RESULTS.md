# 实验 11：14 MiB CPU-LRU 与 SSD-LRU 实测

使用提交 `8b6d0319a4f57f2ac54f51ad07a6d2ee7f22ca69`、Qwen2.5-1.5B-Instruct、vLLM 0.29.0。在 WSL 使用同一个 GPU 锁分别启动和停止两组独立服务，组间释放。沿用实验 08 的原始 1,025-token 预热与固定 78 条；两组各 79 条输出逐 token 与实验 08 的成功组一致。

| 指标 | CPU-LRU | SSD-LRU |
| --- | ---: | ---: |
| 预热完整块 → 连续保留块 | 16 → 8 | 16 → 8 |
| CPU 峰值 / 正式淘汰块 | 8 / 258 | 8 / 258 |
| 正式连续前缀（块数：请求数） | 1：24；2：6；8：48 | 1：24；2：6；8：48 |
| GPU 前缀命中 token | 10,368 | 10,368 |
| 外层回迁 token | 0 | 3,456 |
| SSD→CPU→GPU 完整回迁（块 / 请求） | — | 54 / 12 |
| 提示词重算 token（实验 08 口径） | 26,190 | 22,734 |
| SSD 逻辑读 / 写 MiB | — | 94.50 / 357.00 |
| 进程树实际读 / 写 MiB | 0 / 0.36 | 94.50 / 357.52 |
| 全体客户端 / 服务端 TTFT 中位数（ms） | 124.08 / 119.44 | 110.09 / 107.06 |

SSD 回迁的 54 块均有写盘、CPU 淘汰、盘读及同请求 GPU 回迁记录；未出现只读到 CPU 的块。复核器核对了盘层提交及完成记录中的 532 条块槽对应关系。

| 配对子集 | 请求数 | SSD-LRU − CPU-LRU 客户端 / 服务端 TTFT 中位差（ms） | 重算 token 差 |
| --- | ---: | ---: | ---: |
| 全体 | 78 | −2.06 / −1.38 | −3,456 |
| 两组均有前缀命中 | 36 | +0.54 / +0.63 | 0 |
| SSD 完整回迁 | 12 | −67.07 / −66.96 | −3,456 |

[实验 08 的 14 MiB 基线](../08_kv_capacity/results-14.md)中，CPU-LRU 与 SSD-LRU 两轮均在同一预热失败，四组都完成 0 条预热和 0 条正式请求。历史 admission／learned 八组的时延、回迁与重算另列于 `review.json`，未并入本次配对。本次每组只运行一次，时延差仅描述这批配对。

原始工件目录：`.agent/kv-prefix-capacity-11/paired-14/`。分组目录下保留服务启动参数与日志、审计 JSONL、逐请求输出、Prometheus 快照、进程 I/O 样本和盘层文件。相邻的 `paired-14.log` 记录两次 `flock` 调用，`paired-14-lock.txt` 记录锁文件的设备号与 inode。工作树源码变更时完成的首次 CPU 采集保留于 `.agent/kv-prefix-capacity-11/frozen-14/`，未纳入正式配对。

| 工件 | SHA-256 |
| --- | --- |
| `environment.json` | `45c8da2aa5f0b200389aaaed00ffffcdfc0980d2c12cbee400f3af771c7ca2c8` |
| `review.json` | `a8e1feb78e85b3f38a988aef97fede6279bd78d2f843c4b4de80d1dd74e631ce` |
| CPU-LRU 原始证据清单 | `4f0043431505aa658e73790ee5e397cd4f2aa06d3174abf2853c5013b6248950` |
| SSD-LRU 原始证据清单 | `b4ba9d543c875300457c47c1c6c19d554ae537dd1e50ae924c67bdf11bf13704` |
| 79 条输出 token 序列 | `6652cad00ca2a66889540daa772124ec8da918d6147a941026de3cb33e563aed` |
