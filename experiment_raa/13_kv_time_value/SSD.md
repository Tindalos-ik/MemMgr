# 实验 13：eager SSD 已发布副本的 CPU 保留价值

复用实验 13 的冻结评价计划、冻结 GBDT、RTX 4050 Laptop GPU、Qwen2.5-1.5B-Instruct FP16、32 MiB GPU 和 28 MiB CPU KV 配额。eager SSD 上限 4 GiB，每组从独立空盘目录开始。三条校准路径均启用 eager SSD，各组按独立的 WSL `flock -x .agent/kv-experiment-gpu.lock` 启停；`gpu-lock.json` 记录服务子进程实际继承的独占锁和 inode。

## 校准证据

冻结计划 `plan.json` 的 SHA-256：`6e0c644c99517b65dcbb817a574b7e9913af3e1c7fa21bd9bef7c210770eb134`。

- 冷重算臂：16 条探测均无 SSD 读回、CPU 回迁和 GPU 命中。复核 17 条请求的原始 Prometheus 指标（含预热）、144 个完成发布的 O_DIRECT 文件和正常关闭记录。原始目录 `.agent/kv-time-value/run-13/calibration-ssd/recompute/`，`complete.json` SHA-256 `9094cea2f534cbbeba0791cd6b7a373cf400d92b27ab1be398c2e34a36fd6235`，`gpu-lock.json` SHA-256 `5ad6e66898157603e36fe28b39c1fc0222baacb5bcb35205a7ede475d2366a5a`。
- CPU 回迁臂：16 条探测共 116 块 CPU→GPU DMA，SSD 读回为零；128 个探测输出 token ID 与冷重算臂逐一一致。复核 33 条请求的原始指标和 144 个发布文件。原始目录 `.agent/kv-time-value/run-13/calibration-ssd/cpu/`，`complete.json` SHA-256 `69c306184222bfd8aa04ade25798230a47a87524a7c7292d6e2f7da055f12b48`，`gpu-lock.json` SHA-256 `a8a183209e6c179abf25eab4c234c5f92a725fe7654505d9747683b0a5406bcf`。
- SSD 回迁臂：16 条探测共 116 块完成 `eager` 写入发布 → CPU 淘汰 → O_DIRECT 读入 CPU 槽 → 同请求 GPU DMA，逻辑读 212,860,928 字节，进程树实读 212,865,024 字节。128 个输出 token ID 与另两臂逐一一致。原始目录 `.agent/kv-time-value/run-13/calibration-ssd/ssd/`，`complete.json` SHA-256 `924ad092efe53667c39279f8b5d6608f27523a42d73992ef429c0f27fea367de`，`gpu-lock.json` SHA-256 `7650f6da9a1373e49b60ecadf542546af7ac361979aad78f0e572a3291c59910`。

各深度 4 个独立配对样本，GPU 命中为零。单位 ms。配对差先对相同 prompt 相减，再取中位数。

| 连续块数 | 重算 / CPU / SSD TTFT 中位数 | 重算 − CPU 配对差 | SSD − CPU 配对差 | GPU→CPU / 初次盘写中位用时 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 175.59 / 117.47 / 109.93 | +62.16 | −8.42 | 1.62 / 7.47 |
| 4 | 181.15 / 106.67 / 137.24 | +73.74 | +30.58 | 5.76 / 23.63 |
| 8 | 312.26 / 113.13 / 128.42 | +198.02 | +20.55 | 11.62 / 46.69 |
| 16 | 506.62 / 85.03 / 159.18 | +416.03 | +64.58 | 20.17 / 97.02 |

实测写入估计为 1.446 ms/块（GPU→CPU）和 5.986 ms/块（首次 SSD 写入）；保留负的 SSD 差值，按锚点差值计算边际量。校准只使用实验 13 独立校准输入；SSD 评价不读取 CPU-only profile。设备范围为当前 WSL `/dev/sdd` ext4 O_DIRECT、eager、单序列及 1–16 块，未校准混合层几何。

`profile-ssd.json` SHA-256 `b27ced277d8d0e25ec00d52980745b167e5a4abbc4d6a84af33597ecedffc188`；`review-ssd-calibration.json` SHA-256 `7ac67bcb16cf29a56b56f668cdbeb99a4f4e43a1c0953980295bbd6a235f5143`。两者在 `.agent/kv-time-value/run-13/`。

## 未见计划的逐组证据

沿用同一冻结的 88 条正式请求（24 条复访）和 1 条预热。每组保留原始输出 ID、请求审计、原始 Prometheus 快照、进程 I/O 和锁证据。

- `admission`：24 条完成 184 块 SSD→CPU→GPU 读回链，盘写 112 块。Linux 进程树实读 337,641,472 字节，与盘读逻辑字节一致。原始目录 `eval-ssd/admission/`，`complete.json` SHA-256 `0bcce08e9343fd6ccb65c5b4c78924fb8e85f9ad47dcf69200094e72b9975c7e`，`review-ssd-admission.json` SHA-256 `a8d4175538eb0bf09cad19537b0e00f93e1a9e7abf6afd819f390f1989f4f1cc`。
- `learned`：88 条输出与 `admission` 逐 token 一致。24 条回访读盘 190 块，其中 188 个同请求唯一块完成整链。请求 `cmpl-tv13-0066-0-9a3a5d9e` 的两个块在同请求中读盘两次；原生盘读计 190，整链不重复计两次。进程树实读 348,651,520 字节。原始目录 `eval-ssd/learned/`，`complete.json` SHA-256 `3b4a7eb281fde110acd06a2692efb05b7450d0fbbc6b191bafa78e09a2dde20e`，`review-ssd-learned.json` SHA-256 `be2b9efb1cc5db08bea496abc1930bc87b9ca100adb353082627ca8541c345b7`。
- `time_value`：24 条完成 163 块 SSD→CPU→GPU 读回链，无重复盘读。进程树实读 299,106,304 字节。原始目录 `eval-ssd/time_value/`，`complete.json` SHA-256 `54f3078e9a3f47a63d32ce2813acd1b2b0b93d4a1638f982a52a8bd473fa1d6d`，`audit.jsonl` SHA-256 `d2cf5c3a3df228ad052fb50e8ddb6f58e0300a7f24956d4323c2dee1882edb16`，`gpu-lock.json` SHA-256 `830c4dd1b7da9390d8705e5b463c925783dfa5b33ad44ee666f0e1e1188f16ec`。

三组的输入、冻结准入、704 个正式输出 token ID 和 8 个预热输出 token ID 逐一一致。六个正式 GPU 服务时间窗互不重叠，子进程证据指向同一把 Linux 锁。

## 已发布副本的保留计价

正式窗口内 48 次计价，12 次缩短或拒绝写入。47 条请求触发已发布副本分支：111 次受害候选计价走 `ssd`，其中实际淘汰 90 块。126 个待写入候选走 `recompute`；本 workload 未观察到候选写入本身已有盘副本的分支。提升级的原生淘汰另计，不混入保留决策的 90 块。

可定位样本：预热 `cmpl-tv13-0000-0-901df68f` 的 `fs_complete` 作业 0 在 `30452910617813 ns` 发布首块 `7b934c95…`；约 2.886 秒后，`cmpl-tv13-0002-0-a1d3562e` 将同一块作为 SSD 受害项计价。该块概率约 0.002850，应用 1 块 SSD 锚点后机会成本约 −0.024 ms；下一条 `store` 事件确认淘汰同一块。复核按事件时间重建当时可恢复的祖先链，并逐项重算路径、写入费用、受害机会成本和净收益；不使用稍后写盘的记录。

正式请求的累计估值：复用收益 2,218.47 ms，减预计写入 780.34 ms 和受害机会成本 177.40 ms，净估值 1,260.73 ms。此值用于策略选择，实际 TTFT 未显示净改善。

## 未见计划的配对结果

| 策略 | CPU 写 / 读回块 | SSD 写 / 物理读块 | 重算 token | GPU→CPU / SSD 写入累计（ms） | SSD 读取累计（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `admission` | 112 / 192 | 112 / 184 | 22,616 | 197.04 / 673.63 | 604.01 |
| `learned` | 96 / 192 | 96 / 190 | 22,616 | 136.06 / 607.86 | 543.39 |
| `time_value` | 105 / 178 | 105 / 163 | 23,512 | 242.18 / 737.60 | 514.07 |

配对差为处理组减对照组，按相同请求位置逐条相减。P95 为差值的最近秩。

| 比较 | 窗口 | 服务端中位数 / P95（ms） | 客户端中位数 / P95（ms） |
| --- | --- | ---: | ---: |
| `learned − admission` | 全部 88 条 | −7.48 / +34.26 | −7.79 / +33.68 |
| `time_value − admission` | 全部 88 条 | +2.68 / +58.09 | +3.71 / +61.59 |
| `time_value − learned` | 全部 88 条 | +7.47 / +73.06 | +9.90 / +76.16 |
| `learned − admission` | 24 条复访 | −3.78 / +28.79 | −3.75 / +25.15 |
| `time_value − admission` | 24 条复访 | +2.68 / +55.43 | +2.30 / +73.10 |
| `time_value − learned` | 24 条复访 | +7.15 / +71.31 | +6.20 / +75.56 |

相对 `learned`，`time_value` 少读 27 次物理块（47.25 MiB），多写 9 块（15.75 MiB），多重算 896 token；TTFT 中位数和配对 P95 均变差。相对 `admission` 少写 7 块，少读 21 块，但也多重算 896 token。各策略只采集一轮，运行时段不同。证据限于本机串行 eager SSD 压力回放，支持保留原默认策略。

原始数据 `.agent/kv-time-value/run-13/{calibration-ssd,eval-ssd}/`；三组复核 `review-ssd.json` SHA-256 `b6ca97b475f430bacd69d4962d9a89dfb0e97dbe9aba56fa9e0e666cbec55d49`。复核中的 `first_replica_pricing` 保留块的完整身份及发布、计价两处审计时点。
