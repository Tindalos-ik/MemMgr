# SciFact 检索问答的 KV 回放

## 准备真实输入

从项目根目录执行。`PYTHON` 指向已安装项目依赖的 WSL 解释器。使用 `chromadb==1.5.3` 的独立 Chroma CLI，在另一个 WSL 终端启动服务：

```bash
mkdir -p .agent/kv-scifact
chroma run --host 127.0.0.1 --port 18110 --path .agent/kv-scifact/chroma-data
```

```bash
PYTHON=${PYTHON:-.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m experiment_raa.10_scifact_kv.prepare_scifact \
  .agent/kv-scifact/prepared --chroma-port 18110
```

准备程序要求新的输出目录。它将 SciFact 语料、问题和官方测试标签复制到该目录，用项目固定版本的 BGE 在 CPU 上编码，调用项目索引器导入全部 5,183 篇文献，再通过 Chroma HTTP 查询取得每个问题的前三个文本块。按文档去重后使用项目 RAG 聊天模板和 1,024 token 输入预算。`inputs.json` 保留证据、检索距离、原始问题、聊天消息、输入 token 和序列。`--verify-only` 可用保存的证据离线逐 token 核对渲染结果。

以种子 1010 从官方 300 条测试问题中抽取 48 条，分成六个互不重复的序列。每条序列先按固定顺序提交八个不同问题，再按固定计划对其中四个问题逐字节复访十次。不同问题之间可能共享模板或文本前缀；复访是人为安排的，整次回放没有使用真实用户流量。BGE 和 Chroma 只参与准备，生成阶段复用冻结的检索结果。

## 回放与离线复核

每个 GPU 阶段单独通过 Linux `flock` 持有项目约定的锁。使用实验 06 正式产物 `.agent/kv-joint/output/model/`；程序核对模型、元数据的 SHA-256 和冻结阈值 `0.35`。`--model-dir` 可指定同一冻结产物的其他 `.agent/` 路径。正式运行使用新的回放目录：

```bash
PYTHON=${PYTHON:-.venv/bin/python}
D=.agent/kv-scifact
for epoch in forward reverse; do
  flock -x .agent/kv-experiment-gpu.lock "$PYTHON" -m experiment_raa.10_scifact_kv.run_scifact \
    "$D/prepared" "$D/replay" --epoch "$epoch" --port 18111 || exit
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m experiment_raa.10_scifact_kv.verify_scifact "$D/replay"
```

每个顺序阶段中，三个策略各启动一个真实 vLLM 进程。策略执行顺序为 `lru → admission → learned` 及其反序；序列执行顺序也反转，序列内的请求位置不变。每条序列开始前通过 vLLM 清空 GPU 和 CPU 缓存，同时清空策略访问历史，首个请求必须零命中。三组均使用 32 MiB GPU 和 56 MiB CPU KV 预算、64-token 块、串行贪心生成（最多 64 个输出 token），并允许回迁所有实际可用的连续 CPU 前缀。`admission` 和 `learned` 使用同一冻结模型做写入准入，只有 `learned` 使用复用评分决定保留和淘汰；本实验不启用实验 06 的回迁成本表。

`audit.jsonl` 记录实际策略事件及纳秒级分段计时；每条序列保留流响应的实际输出 token ID 和逐请求 Prometheus 原始快照。特征计时涵盖历史更新及模型特征读取，评分计时涵盖真实非空模型批次，保留和淘汰单列并扣除内层特征、评分与审计写入。客户端 TTFT 从发起生成请求到首个带输出 token ID 的流事件；服务端使用 vLLM 的逐请求 TTFT 指标。两者都不含检索时间。
