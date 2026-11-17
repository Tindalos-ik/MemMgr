# 分层 KV 缓存

CLI、Web 问答与实验任务共用正式 vLLM 服务。在 WebUI 配置页或 `config/server.json` 的 `kv_cache` 中设置，保存后重启 vLLM。

## 配置

| 字段 | 用法 |
|---|---|
| `tier` | `gpu`、`cpu`、`ssd`，默认 GPU |
| `policy` | `lru`、`admission`、`learned`、`placement` |
| `batch_invariant` | CPU、SSD 策略须设为 `true`，固定数值计算路径 |
| `gpu_bytes` | GPU KV 预算；`0` 使用引擎自动分配 |
| `cpu_bytes` | CPU KV 预算，包含 SSD 中转空间 |
| `ssd_bytes` | SSD KV 字节上限，包含在途新增文件的预留量 |
| `ssd_root` | `.agent` 内的独占 SSD 目录，须支持 O_DIRECT |
| `model_dir` | 自己训练并确认可信的 GBDT 目录 |
| `placement_profile` | 联合放置所用真实路径成本表 |
| `restore` | CPU 支持 `always`、`never`、`gate`；SSD 使用 `always` |
| `cost_profile` | `restore=gate` 所用 CPU 回迁成本表 |
| `margin_ms` | 决策要求的最小预计节省，默认 0.5 ms |
| `audit_path` | `.agent` 内的真实块事件 JSONL |

当前支持 vLLM 0.29.0、单 worker、单序列、FP16、64-token 块和 LBHNC。SSD 经 CPU 中转；容量、I/O、布局与配置错误直接报错。总览页的 KV 卡片显示真实引擎指标。

`lru` 接纳完整前缀并使用 LRU；`admission` 使用 GBDT 筛选写入；`learned` 同时比较预测替换收益。CPU 策略可按真实请求历史训练模型：

```bash
"$PYTHON" -m memmgr.short_term.train \
  --audit .agent/collect/audit.jsonl \
  --output .agent/kv-model --horizon 8
```

模型文件使用 joblib，只加载自己生成且来源可信的文件。预测模型的统计定义、模型 revision 或块布局改变后重新采集与训练。

训练同时生成窗口内的复用分类器与条件下次复用间隔回归器。`--requests N` 可限定采集前段，将后续成本专用序列排除在训练和验证之外。

## 联合放置

先按[三级放置实验](../experiment_raa/06_kv_placement/README.md)的采集步骤获得同环境的 `model/` 与 `costs.json`，再设置：

```json
{
  "tier": "ssd",
  "policy": "placement",
  "batch_invariant": true,
  "gpu_bytes": 60817408,
  "cpu_bytes": 134217728,
  "ssd_bytes": 234881024,
  "ssd_root": ".agent/placement-session-01",
  "model_dir": ".agent/placement-results/model",
  "placement_profile": ".agent/placement-results/costs.json",
  "restore": "always",
  "audit_path": ".agent/placement-session-01/events.jsonl"
}
```

其余字段保留默认值。设备、服务数值模式、实际容量或实现改变后重新校准；最大前缀长度须处于成本表范围内。校准环境中的 GPU 预算、`enforce_eager`、`gpu_memory_utilization` 与上下文上限也须保持一致。

联合策略使用复用概率与条件下次复用间隔，比较 CPU、SSD 与丢弃的预计净收益，并为 SSD 预留一个最大前缀所需的 CPU 中转空间。间隔按串行请求数计量。只在真实写入完成且引用归零后释放中转块。每个请求的 `placement` 事件记录概率、间隔、各目标收益、替换损失与实际目标；`residency` 事件记录传输完成后的真实驻留。模型目录须包含新训练产生的 `delay.joblib`。

当前联合策略使用串行请求，每次启动指定空 SSD 会话目录。一般 LRU、预测策略的 SSD 缓存可在相同数值模式与命名空间下重启复用。文件层保障正常读写与原子提交，断电持久性不属于接口保证。

## CPU 回迁门槛

按完全相同输入分别运行 `restore=always` 与 `restore=never`，保存原始 `audit.jsonl`、`requests.jsonl`，每种输入几何至少取得两次配对观测：

```bash
"$PYTHON" -m memmgr.short_term.cost \
  --pair .agent/cpu-always .agent/cpu-never \
  --output .agent/cpu-costs.json
```

将 `cost_profile` 指向生成文件，`restore` 设为 `gate`。未知输入几何或运行身份不匹配会报错。该门槛与联合放置分别配置。

## 真实入口检查

完成第 06 组实验后，可用生成的联合策略服务配置检查真实推理、Web 配置往返、实际运行参数、KV 指标和非法路径拒绝：

```bash
"$PYTHON" tests/integration/placement_live_acceptance.py \
  --server .agent/placement-results/e0-placement/server.json \
  --runtime "$RUNTIME" --prompts .agent/ssd-results/prompts.json \
  --output .agent/placement-acceptance --chroma-port 8001
```

通过 `--chroma-port` 指定已建立 small 索引的 Chroma 服务，同时检查 CLI 与 Web 检索问答。检查时独占 GPU 与项目配置。脚本启动自有 vLLM、WebAPI，并在退出时停止服务和恢复原配置。
