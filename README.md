# MemMgr：面向大模型推理的长短期记忆协同管理系统

## 项目目标

MemMgr 面向大模型推理，围绕短期 KV 缓存与长期记忆检索、写入之间的协同管理开展研究，探索记忆访问、资源分配和推理效率之间的关系。

## Web 实验工作台

WebUI 包含总览、流式问答、实验任务、数据库、数据与索引、日志、实验报告和配置八页。准备好 PATH 中的 `uv`、Node.js 22 LTS / npm 10，以及兼容 CUDA 13.2 的 NVIDIA 驱动。克隆仓库并进入项目根目录后执行：

```bash
uv venv --python 3.12 .venv
export PYTHON="$PWD/.venv/bin/python"
uv pip install --python "$PYTHON" -c constraints.txt -r requirements.in --extra-index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$PYTHON" scripts/download_models.py
cd web
npm ci
npm run build
cd ..
```

```bash
bash start-webui.sh
```

页面地址为 **http://127.0.0.1:8765**。可在总览页分别启动 vLLM 与 Chroma；Chroma 需要可用的 Docker Engine。完整环境要求见[部署说明](docs/deploy.md)。也可从任意目录执行 `bash /项目路径/start-webui.sh`。解释器按 `PYTHON` 环境变量、项目 `.env.webui` 中的 `PYTHON`、项目 `.venv/bin/python` 的顺序选择，无需激活环境。使用已有环境时，在 `.env.webui` 中保存一次解释器路径即可，文件不提交到 Git，详见[安装依赖](docs/deploy.md#2-首次安装)。构建产物缺失时，脚本会提示前端构建命令并退出。按 Ctrl-C 停止 WebUI。手动服务管理命令见[CLI 服务管理](docs/deploy.md#6-可选cli-启动服务)。

## 本地部署与使用

先完成本地模型部署、基础检索生成链路验证和初测数据采集。本轮实验采用 Qwen 1.5B / vLLM、CPU BGE 与 Chroma，为后续协同管理研究建立可复现的功能和性能基线。后续计划开展固定版本的 7B 模型实验。

本轮实验的数据流为：**SciFact 文献 → CPU 向量编码 → Chroma 检索 → 组织证据 → Qwen 回答**。链路图、计时图和实测结果见[最终实验报告](reports/Tindalos-ik_test_report.md)；实验分组、每个参数与指标的含义见[实验设计](docs/design.md)。

以下命令在 **Linux / WSL 项目根目录**执行。核心版本为 vLLM 0.29.0、PyTorch 2.13.0+cu132、Transformers 5.17.0。

### 1. 准备环境和模型

先按上面的步骤创建项目 `.venv`、安装依赖并下载模型。每个新终端从项目根目录设置 `export PYTHON="$PWD/.venv/bin/python"`，也可直接使用 `.venv/bin/python` 执行 Python 命令；使用已有环境时将 `PYTHON` 设置为对应解释器。

模型配置只保存模型 ID 与固定 revision。下载与本地加载遵循 Hugging Face 标准缓存：优先使用 `HF_HUB_CACHE`，否则使用 `HF_HOME` 下的 `hub`；未设置 `HF_HOME` 时使用 `XDG_CACHE_HOME` 下的 `huggingface/hub`，默认位置为 `~/.cache/huggingface/hub`。自定义缓存时，下载和所有服务使用相同环境即可，无需修改 JSON；下载脚本不写回配置。固定模型版本及服务参数见[部署说明](docs/deploy.md)。

### 2. 启动服务（CLI 方式）

```bash
bash scripts/chroma.sh up
bash scripts/start-vllm.sh
```

vLLM 在当前终端运行。保留该终端，再开一个终端，从项目根目录设置 `PYTHON` 并继续第三步；自定义缓存环境也需保持一致。

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

单条问答命令适合检查检索证据与模型回答；本轮实际问答示例见下文和[实测报告](reports/Tindalos-ik_test_report.md#5-真实问答示例)。

[本轮实验的完整例子](reports/Tindalos-ik_test_report.md#5-真实问答示例)探讨 DdrB 蛋白的单链 DNA 结合能力，采用文档 `15305881`、`14275671`、`4406819`，实际回答为：

> The DdrB protein from Deinococcus radiodurans is an alternative SSB because it forms a pentameric ring capable of binding single-stranded but not double-stranded DNA, and its crystal structure reveals a novel fold distinct from other single-stranded binding proteins.

该例的完整证据和引用表现见[实测报告](reports/Tindalos-ik_test_report.md#5-真实问答示例)。

## 初测结果与后续实验

本轮初测采用固定配置，包含三级语料检索和 7 组问答条件：

```bash
"$PYTHON" -m memmgr --config config/baseline.json prepare
"$PYTHON" scripts/run_baseline.py
```

后续实验可按新配置再次运行统一基线脚本。

本次完成 180 条检索和 112 条问答请求。基线条件下，问答平均耗时约 1.06 秒。实验条件、结果表、证据摘录和统计口径统一保存在[最终实验报告](reports/Tindalos-ik_test_report.md)。

## 项目导航

| 入口 | 内容 |
|---|---|
| [部署说明](docs/deploy.md) | 依赖、模型、服务、运维和实验室迁移 |
| [目录与协作](docs/目录与协作.md) | 包边界、模块职责、共享入口负责人和开发测试命令 |
| [实验设计](docs/design.md) | 实验矩阵、变量与指标定义、计时边界、统计方法和复现步骤 |
| [实测报告](reports/Tindalos-ik_test_report.md) | 数据配置、性能结果、功能验收和问答示例 |
| [配置](config/) | 模型、检索和正式实验参数 |
| [应用](memmgr/) | `long_term/` 长期记忆、`short_term/` 短期策略命名空间、`inference/` 推理、`benchmarks/` 实验与 `webui/` 后端 |
| [脚本](scripts/) | 服务管理、实验执行和结果复算 |
| [测试](tests/) | 文本分块、隔离与提示词预算测试 |

模型回答的引用遵从率、事实核对要求和样本范围统一列在报告的[适用范围](reports/Tindalos-ik_test_report.md#6-适用范围与结果解释)中。
