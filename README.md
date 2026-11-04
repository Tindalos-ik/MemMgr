# MemMgr：面向大模型推理的长短期记忆协同管理系统

## 项目目标

MemMgr 面向大模型推理，围绕短期 KV 缓存与长期记忆检索、写入之间的协同管理开展研究，探索记忆访问、资源分配和推理效率之间的关系。

## 本地部署与使用

先完成本地模型部署、基础检索生成链路验证和初测数据采集。本轮实验采用 Qwen 1.5B / vLLM、CPU BGE 与 Chroma，为后续协同管理研究建立可复现的功能和性能基线。后续计划开展固定版本的 7B 模型实验。

本轮实验的数据流为：**SciFact 文献 → CPU 向量编码 → Chroma 检索 → 组织证据 → Qwen 回答**。链路图、计时图和实测结果见[最终实验报告](reports/baseline-report.md)；实验分组、每个参数与指标的含义见[实验设计](docs/experiment-design.md)。

以下命令在 **WSL 项目根目录**执行。当前部署使用已有共享 Python 环境，核心版本为 vLLM 0.29.0、PyTorch 2.13.0+cu132、Transformers 5.17.0。

### 1. 准备环境和模型

这些地址是笔者本地的，根据自己的情况进行修改

```bash
export PYTHON="${PYTHON:-$HOME/mycode/vLLM/.venv/bin/python}"
export UV="${UV:-$HOME/.local/bin/uv}"
export HF_HOME="$HOME/.cache/huggingface"
"$PYTHON" -m memmgr --help
"$PYTHON" scripts/download_models.py
```

依赖安装、显存要求和固定模型版本见[部署说明](docs/deployment.md)。现有下载及启动脚本采用当前部署的固定缓存路径；迁移时需先调整脚本中的路径，具体位置见部署说明。

### 2. 启动服务

```bash
bash scripts/chroma.sh up
bash scripts/start-vllm.sh
```

vLLM 在当前终端运行。保留该终端，再开一个 WSL 终端继续第三步，并设置同样的 `PYTHON` 变量。

服务地址：Qwen 为 `127.0.0.1:8000`，Chroma 为 `127.0.0.1:8001`。已有服务运行时，可用以下命令查看状态：

```bash
"$PYTHON" scripts/service_status.py
```

### 3. 准备数据并建立索引

```bash
"$PYTHON" -m memmgr --config config/baseline.json prepare
"$PYTHON" -m memmgr --config config/baseline.json index --tier small
```

`baseline.json` 固定实验数据与模型参数。数据来自 BEIR SciFact，文档和相关性标签保留原始 ID；索引结束时检查文本块数量并执行一次检索。

## 单条问答

下面的问题使用刚建立的 small 索引：

```bash
"$PYTHON" -m memmgr --config config/baseline.json ask --tier small \
  --query "The DdrB protein from Deinococcus radiodurans is an alternative SSB."
```

单条问答命令适合检查检索证据与模型回答；本轮实际问答示例见下文和[实测报告](reports/baseline-report.md#5-真实问答示例)。

[本轮实验的完整例子](reports/baseline-report.md#5-真实问答示例)探讨 DdrB 蛋白的单链 DNA 结合能力，采用文档 `15305881`、`14275671`、`4406819`，实际回答为：

> The DdrB protein from Deinococcus radiodurans is an alternative SSB because it forms a pentameric ring capable of binding single-stranded but not double-stranded DNA, and its crystal structure reveals a novel fold distinct from other single-stranded binding proteins.

该例的完整证据和引用表现见[实测报告](reports/baseline-report.md#5-真实问答示例)。

## 初测结果与后续实验

本轮初测采用固定配置，包含三级语料检索和 7 组问答条件：

```bash
"$PYTHON" -m memmgr --config config/baseline.json prepare
"$PYTHON" scripts/run_baseline.py
```

后续实验可按新配置再次运行统一基线脚本。

本次完成 180 条检索和 112 条问答请求。基线条件下，问答平均耗时约 1.06 秒。实验条件、结果表、证据摘录和统计口径统一保存在[最终实验报告](reports/baseline-report.md)。

## 项目导航

| 入口 | 内容 |
|---|---|
| [部署说明](docs/deployment.md) | 依赖、模型、服务、运维和实验室迁移 |
| [实验设计](docs/experiment-design.md) | 实验矩阵、变量与指标定义、计时边界、统计方法和复现步骤 |
| [实测报告](reports/baseline-report.md) | 数据配置、性能结果、功能验收和问答示例 |
| [配置](config/) | 模型、检索和正式实验参数 |
| [应用](memmgr/) | 数据处理、检索、问答与记忆操作 |
| [脚本](scripts/) | 服务管理、实验执行和结果复算 |
| [测试](tests/) | 文本分块、隔离与提示词预算测试 |

模型回答的引用遵从率、事实核对要求和样本范围统一列在报告的[适用范围](reports/baseline-report.md#6-适用范围与结果解释)中。
