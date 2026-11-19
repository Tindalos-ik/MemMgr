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
