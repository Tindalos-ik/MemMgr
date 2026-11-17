# 多轮 RAG 与三级 KV 对照

使用正式服务和 WebAPI，对照 `lru`、`learned`、`placement`。每组包含两个交替会话，每个会话四轮，覆盖检索证据相同、部分变化、主题变化三种条件。两轮反转策略执行顺序，共 144 条评价请求。

使用已有可信预测模型，在当前设备重新采集 90 条真实路径成本。GPU／CPU／SSD 物理预算分别为 58／128／224 MiB；数值模式、模型、输入与生成参数保持一致。评价问题取第 05 组输入问题的后半部分，预测模型应仅由前半部分训练。运行前检查预测模型来源。

## 运行

要求 Linux／WSL、独占 NVIDIA GPU、支持 O_DIRECT 的 SSD，以及已启动于 `127.0.0.1:8001` 的已有 SciFact small 索引。沿用项目固定依赖，`MODEL` 为固定 Qwen 模型快照，`RUNTIME/hf` 为已有 Hugging Face 缓存。

```bash
"$PYTHON" experiment_raa/07_session_rag/run_session.py \
  --model "$MODEL" --runtime "$RUNTIME" \
  --prompts .agent/ssd-results/prompts.json \
  --predictor .agent/placement-results/model \
  --output .agent/session-results
```

输出目录须尚不存在且位于项目 `.agent` 内。默认引擎端口 18086，WebAPI 端口 18765，可通过 `--port`、`--web-port` 修改。运行时独占项目配置，退出恢复原配置，只停止自己启动的服务。

联合策略评价后集中检查双会话、历史追加、自动检索上下文、未知及已结束会话、索引边界、预算拒绝、生成前取消、正文期间取消、连接断开和后端重启续接。另取三条实际 prompt，使用独立 `cache_salt` 强制全量重算，核对真实重算 token、回答和服务端用量。

## 输出

- `environment.json`、`trace.json`：源码摘要与固定评价输入。
- `collect/`、`costs.json`、`model/`：原始成本测量、当前环境成本表与冻结预测模型。
- `e*-*/`：真实配置、逐轮响应与指标、块事件和进程 I/O。
- `e0-placement/acceptance/`：集中验收和强制重算原始响应。
- `summary.json`：模型及端到端首次内容等待、P95、配对时延节省、重算 token 与传输载荷。

复核使用第 06 组块审计器，重建 CPU 引用、SSD 作业、容量与驻留，并检查每个请求的 GPU 命中、外部恢复和重算 token 守恒。策略间实际消息、回答与用量须一致。

模型首次内容等待从上游生成请求开始；端到端等待包含会话准备、检索与 prompt 组装。指标稳定等待在请求计时之外。评价不额外插入扫描压力，SSD 使用情况取决于实际多轮输入，按原始结果报告。
