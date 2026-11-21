# KV 容量与策略归因

正式计划沿用 [07](../07_kv_joint_ssd/README.md) 的 78 条请求、1 条预热、Qwen 和冻结 GBDT。GPU 配额 32 MiB，盘层配额 4 GiB，每组各自启动一个服务与空缓存，重复两轮且第二轮逆序。CPU 28 / 14 MiB 各测 CPU / CPU+SSD × LRU / admission / learned；56 MiB 复用 07 的 LRU 和 learned 两轮，只重放同条件的 admission。

| 策略 | GPU→CPU 写入准入 | CPU 保留否决 | 淘汰 |
| --- | --- | --- | --- |
| LRU | 全部完整前缀 | 无 | LRU |
| admission | 冻结 GBDT 的连续前缀 | 无 | LRU |
| learned | 同一冻结准入 | 入驻分数与候选牺牲块分数比较 | 按分数淘汰叶块 |

三组统一 `always` 回迁。`admission − LRU` 固定淘汰规则，可比较写入准入的端到端效果；其中 TTFT 含预测开销。`learned − admission` 同时引入保留否决与替换规则，不能分离两者。56 MiB 的 admission 与 07 是分批测量，跨批 TTFT 还可能受时间和设备状态变化影响。

## 采集

在项目根目录运行，使用项目 WSL Python。`CAP` 每次取 `56`、`28` 或 `14`，输出目录须不存在。端口默认 `18088`；选择空闲端口可加 `--port`。

```bash
CAP=28
PYTHON=.venv/bin/python
OUT=.agent/kv-capacity/frozen-$CAP
mkdir -p .agent/kv-capacity
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" \
  experiment_raa/08_kv_capacity/run_capacity.py --cpu-mib "$CAP" --output "$OUT"
```

盘层使用项目 `.agent/` 文件系统，须支持 `O_DIRECT`。`--source` 默认读取 `.agent/kv-joint-ssd/output`。组失败会留下日志和 `failure.json`，阶段内其他组按原容量运行；采集退出码 `2` 表示存在失败组。重跑时另选输出目录。

GPU 需轮转时，在同一条 `flock` 命令末尾加 `--one-group`。每次只运行下一组，进程退出即释放锁。以相同参数和输出目录继续，直到生成根 `complete.json` 或 `failures.json`；已结束的组不会重跑，中断且未生成标记的目录须保留另查。

### 零回迁的压力计划

单独运行出现零回迁的策略，比较同策略 CPU 与 CPU+SSD。压力计划取十个独立的 513-token 热前缀，各连续访问四次，再按同序复访；同一前缀的第四次访问到复访之间相隔 9–36 条请求。该计划共 50 条，不并入正式 78 条结果。

```bash
CAP=56
PYTHON=.venv/bin/python
OUT=.agent/kv-capacity/pressure-$CAP
flock -x .agent/kv-experiment-gpu.lock "$PYTHON" \
  experiment_raa/08_kv_capacity/run_capacity.py --cpu-mib "$CAP" \
  --pressure-policies admission learned --output "$OUT"
```

## 离线复核

```bash
"$PYTHON" -m experiment_raa.08_kv_capacity.review_capacity "$OUT" > "$OUT/review.json"
sha256sum "$OUT"/{environment,review}.json
```

- 复核以逐请求 token ID、Prometheus 增量和块事件为依据。SSD 回迁只计完成写入、CPU 淘汰、SSD→CPU 读取、同请求 CPU→GPU 成功的块。逻辑字节按完成作业统计；实际 I/O 是服务进程树正式窗口的 Linux 字节计数，包含其他进程文件 I/O。
- `groups` 为已复核的成功组，含全体、任意前缀命中、外层命中和 SSD 链命中 TTFT，客户端及服务端分报。`paired` 按计划索引求处理组减对照组；空命中集的样本数为 0，时延为 `null`。
- `excluded` 为失败或缺失组；不参与成功组汇总，也不解释为零回迁。组的 `evidence` 列出审计、请求、I/O、快照、服务日志及完成或失败标记的 SHA-256，不读取 SSD KV 内容文件。复核退出 `2` 表示存在失败或未测组；证据矛盾直接报错。
