# GPU/CPU KV 联合实验

四组均使用 GPU 前缀缓存与 CPU KV 层。`lru` 使用 CPU LRU 淘汰，`learned` 使用冻结 GBDT 选择连续前缀并决定保留及淘汰；`always` 回迁可用的 CPU 前缀，`gate` 按匹配的成本表决定回迁或重算。`never` 只用于校准。

## 执行

使用 Linux / WSL、独占的 NVIDIA GPU、vLLM 0.29.0 和 CUDA 兼容环境。按[项目部署说明](../../docs/deploy.md)安装 Python 依赖并下载固定模型；运行器从 [config/server.json](../../config/server.json) 取得模型 ID 和 revision，在本地 Hugging Face 缓存解析快照。无需启动 WebUI 或 Chroma。

在项目根目录运行，`PYTHON` 指向安装上述依赖的解释器：

```bash
PYTHON=${PYTHON:-.venv/bin/python}
OUT=.agent/kv-joint/output
"$PYTHON" experiment_raa/06_kv_joint/run_joint.py --output "$OUT" --through eval
```

`--output` 必须在项目 `.agent/` 内且此前不存在；默认端口 `18086`，可用 `--port` 修改。完整运行顺序为：LRU 采集、训练 GBDT、按 CPU 策略分别做 `always`/`never` 校准、四组依次回放同一评价计划。每组使用独立服务和单序列，评价前缀与预热、训练采集和校准分开。`--through smoke`、`collection`、`calibration` 可执行到对应阶段；每次须使用新输出目录，部分运行不可作为四组评价。

## 离线复核

在采集结束后，从项目根目录运行。复核只读取采集文件和可信本地训练产物，不启动服务，也不需要 GPU：

```bash
PYTHON=${PYTHON:-.venv/bin/python}
OUT=.agent/kv-joint/output
"$PYTHON" -m experiment_raa.06_kv_joint.review_joint "$OUT" > "$OUT/review.json"
```

确认进程退出码为 `0` 后再使用 `review.json`。缺少记录、计数或策略证据不一致、生成 token ID 不同都会以非零状态退出。若失败，按错误中的评价组和请求位置检查原始文件，必要时在新目录重新采集。

| 文件 | 用途 |
| --- | --- |
| `environment.json`、`plans/*.json` | 环境、KV 块字节数、输入 token 计划与摘要 |
| `collect/lru-always/`、`model/` | 真实训练采集、冻结 GBDT 与阈值 |
| `calibration/` | 各 CPU 策略的两组校准原始记录及冻结成本表 |
| `eval/{lru-always,lru-gate,learned-always,learned-gate}/` | `server.json`、`requests.jsonl`、`audit.jsonl`、原始 Prometheus 快照和 `warmup/` |

## 指标口径

- **一致性**：按冻结计划索引配对。复算输入 token 摘要，与请求和引擎审计比对；输出按实际 token ID 序列逐个比较。
- **命中、重算、迁移**：从请求前后原始 Prometheus 快照取差，要求 GPU 命中、CPU 回迁和重算 token 之和等于输入量；双向迁移字节还须与审计中完成的块数相符。总量排除预热请求。
- **无效淘汰**：`evicted_without_read_bytes` 累计正式窗口内被淘汰、且自该次 CPU 写入完成后从未成功回迁的驻留周期。`unread_at_end_bytes` 单列期末仍驻留且未读的字节；不计入无效淘汰。`*_from_warmup` 单列其中由预热写入的部分。
- **策略决策**：对 learned 组重放块访问特征，用冻结模型和阈值复算连续前缀准入及保留决定。对 gate 组按 `(输入 token, GPU 命中 token, CPU 候选 token)` 查对应策略的成本表；仅当预计重算耗时减回迁耗时严格大于配置余量才准入。
- **配对 TTFT**：客户端从发请求到收到首个非空正文片段，服务端取原始 TTFT 指标。配对差值为处理组减对照组，负值表示处理组更快。分别报告全部请求、各 `phase`，以及同一 CPU 策略下候选几何完全相同的 gate 准入和拒绝子集。空子集以 `n: 0` 和 `null` 分位数表示。差值描述此次串行回放。
