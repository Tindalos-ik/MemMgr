# CPU 与 CPU+SSD KV 同期对照

| 正式组 | 缓存层 | CPU 策略 |
| --- | --- | --- |
| `cpu-lru` | GPU + CPU | LRU |
| `cpu-learned` | GPU + CPU | 冻结 GBDT |
| `ssd-lru` | GPU + CPU + SSD | LRU |
| `ssd-learned` | GPU + CPU + SSD | 冻结 GBDT |

四组均为 `always`：回迁可用前缀，learned 仍控制写入准入和保留。每轮使用相同的 1 条预热和 78 条正式输入，各组启动独立服务；第二轮按第一轮的逆序执行。实验沿用 [06](../06_kv_joint/README.md) 的计划与模型，不使用其 CPU 成本门槛推算 SSD 代价。

## 采集

在 Linux / WSL 上按[部署说明](../../docs/deploy.md)准备依赖、固定模型和独占 NVIDIA GPU。`PYTHON` 指向实验 06 使用的解释器，运行器核对软件版本与模型配置摘要。SSD 组的独立缓存目录建在输出目录内；将输出放在支持 `O_DIRECT`、由待测 SSD 承载的文件系统，预留四个 SSD 组的缓存空间。`findmnt -T <输出目录的现有父目录>` 可核对文件系统；WSL 虚拟磁盘的宿主介质另行核对。

从项目根目录运行：

```bash
PYTHON=${PYTHON:-.venv/bin/python}
SOURCE=.agent/kv-joint/output
OUT=.agent/kv-joint-ssd/output
"$PYTHON" experiment_raa/07_kv_joint_ssd/run_ssd_joint.py --source "$SOURCE" --output "$OUT"
```

输出必须在项目 `.agent/` 内且尚未创建。默认端口 `18087`，可用 `--port` 修改。进程退出后检查根 `complete.json`，失败时保留原始文件，查看对应组的 `server.log`，重新采集须另选输出目录。

## 离线复核

采集结束后，从项目根目录执行。复核只读取文件，不启动服务、不需要 GPU：

```bash
PYTHON=${PYTHON:-.venv/bin/python}
OUT=.agent/kv-joint-ssd/output
"$PYTHON" -m experiment_raa.07_kv_joint_ssd.review_ssd_joint "$OUT" > "$OUT/review.json"
```

退出码 `0` 和根 `status: complete` 表示两轮通过；每轮结果在 `epochs.epoch-0`、`epochs.epoch-1`。缺少组完成标记的轮次显示 `not_measured` 和缺失组，未完成的组不参与配对。若仅有完整的单轮，根状态为 `partial`，该轮仍可单独读取。采集未完成时退出码为 `2`；文件缺损或证据矛盾时复核报错。`ssd_recovery: zero` 只用于已通过复核但没有完整 SSD 回迁链的组；CPU 组为 `not_applicable`。

| 输入文件 | 内容 |
| --- | --- |
| `environment.json`、`plans/`、`model/` | 可比环境、冻结请求和模型、文件摘要 |
| 根与各组 `complete.json` | 根记录两轮实际执行顺序；各组记录正式与预热请求数 |
| `epoch-{0,1}/<组>/` | `server.json`、`requests.jsonl`、`audit.jsonl`、`initial.prom`、逐请求 Prom、`final.prom` 和 `warmup/` |
| 各组 `resources.jsonl`、`disk.json` | 同一服务进程树的正式窗口 I/O 样本与期末 `.bin` 文件数和字节 |

## 指标口径

- **输入与输出**：复算输入 token 序列摘要并与客户端、引擎审计比对。按计划索引逐条比较实际输出 token ID；从相邻 Prom 快照复算 GPU 命中、外层回迁、重算 token、双向迁移字节和服务端 TTFT。
- **SSD 回迁链**：已成功写入的同一块后来从 CPU 淘汰，随后在同一请求内从 SSD 读入 CPU，并成功回 GPU，计一条。块重新提升可产生新的一条。预热状态参与验证，正式窗口才计回迁；只读到 CPU 的块单列。
- **逻辑读写与实际 I/O**：逻辑字节按成功传输作业的块数乘 SSD 文件块字节，同一文件的重复逻辑写可能由原生文件层跳过。实际进程 I/O 用排空后的 `formal_start` / `formal_end` 取 Linux `read_bytes` / `write_bytes` 差，要求其覆盖直接读取及新增缓存文件。该计数包含进程的其他文件 I/O，不等同 SSD 设备计数。WSL 子进程的共同创建时钟偏移单列；PID 或累计计数不连续则复核失败。
- **缓存占用**：CPU 报告正式窗口初始、峰值、期末块数及淘汰量，峰值包含已预约的 CPU 槽；SSD 报告初末文件数、字节与新增文件字节，期末以 `.bin` 文件 `st_size` 汇总核对。
- **配对 TTFT 与重算**：每轮按同一输入分别比较两种策略的 SSD 增量、两种层级的 learned 增量。TTFT 差按每对请求先算处理组减对照组，再取中位数；客户端和服务端分别报，负值更快。重算 token 报总量与配对差。同策略的 SSD 对照还单列有证实回迁的请求；空子集为 `n: 0`、中位数 `null`。结果描述此次回放，不作为 SSD 回迁门槛。
