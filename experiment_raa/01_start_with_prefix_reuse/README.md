# KV Cache 前缀复用基线实验

在独占的本机 vLLM 服务上比较冷前缀、相同输入重复请求和新前缀请求，观察缓存命中与首个正文等待时间。实验覆盖 GPU 自动前缀缓存，为后续分层管理实验提供参照。

## 材料

- `prefix_baseline.py`：实验客户端，连接独占服务并记录请求、计时与缓存指标。
- `run_local.py`：下载固定模型、启动本机服务、运行实验并停止自有服务进程。
- `requirements.txt`：独立实验环境的直接依赖版本，与项目根目录的依赖分别管理。
- `.gitignore`：排除实验环境、模型缓存、日志及运行结果。

## 准备环境

Python 3.12、uv 及可用的 NVIDIA GPU。

在项目根目录执行：

```bash
cd experiment_raa
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

运行入口会将实际软件版本、GPU 和服务参数写入 `environment.json`。

## 运行

在 `experiment_raa/` 中执行：

```bash
.venv/bin/python run_local.py
```

入口使用 `127.0.0.1:18080`，端口占用时退出。模型固定为 Qwen2.5-1.5B-Instruct，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；FP16、eager、上下文上限 2048、服务端并发上限 1、显存预算 65%，显式启用 prefix caching。

服务日志、环境信息和结果分别写入本目录的 `server.log`、`environment.json` 和 `results/`。`results/` 已存在时客户端退出；再次运行前自行处理已有结果。入口结束时停止本次启动的服务。

连接已启动的独占服务时，可单独运行客户端，指定本地 tokenizer 快照和新的输出目录：

```bash
.venv/bin/python prefix_baseline.py \
  --url http://127.0.0.1:18080 \
  --tokenizer /固定模型快照的绝对路径 \
  --output results-next \
  --input-tokens 1024 --output-tokens 64 --repeats 5 --seed 7
```

## 实验过程

1. 用相同输入与输出长度的无关前缀预热，单独保存预热记录，再等待统计发布。
2. 每轮生成新的前缀标识，构造长度相同、首个缓存块内容不同的 A、B 两组输入。
3. 顺序发送 A（cold）、A（reused）、B（fresh），共重复 5 轮。
4. 每次请求前后读取 `/metrics`，保存原始快照及前缀缓存计数差值。

默认每条输入 1024 token、输出 64 token，temperature 为 0、seed 为 7。`ignore_eos=True` 保持输出长度一致。客户端核对服务端用量、流结束标记、完成原因和同轮 cold/reused 输出的一致性。

计数器缺失、指标发布超时、冷前缀命中或重复前缀未命中均报错。运行期间应独占服务，其他请求会污染累计指标差值。客户端不调用清空缓存接口。

## 输出字段

| 字段 | 含义 |
|---|---|
| `ttft_seconds` | 发出 HTTP 请求至首个非空正文块的等待，包含服务端等待与本机通信 |
| `total_seconds` | 发出请求至流结束的总耗时 |
| `usage` | 服务端记录的输入和输出 token 数 |
| `prefix_queries` / `prefix_hits` | 请求前后前缀缓存查询和命中 token 累计值之差 |
| `hit_ratio` | 命中 token 数除以查询 token 数 |
| `prompt_token_ids` | 实际提交的输入 token |
| `answer` | 实际生成文本 |

`requests.jsonl` 保存逐请求数据，`*.prom` 保存指标快照，`warmup.json` 保存预热记录，`summary.json` 保存各条件的样本数、耗时中位数及缓存计数总和。模型信息和运行参数分别保存在 `models.json` 与 `arguments.json`。

流式正文块可能包含多个 token，TTFT 的采集粒度为正文块。总耗时包含 prefill，输出 token 数除以总耗时不能代表纯 decode 速度。固定请求顺序、小样本和 GPU 频率变化都会影响计时；分层迁移、淘汰策略与多轮会话需要另行实验。

## 参考

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.29.0/features/automatic_prefix_caching/)
- [vLLM Metrics](https://docs.vllm.ai/en/v0.29.0/design/metrics/)
