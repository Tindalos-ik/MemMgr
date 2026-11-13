# KV 复用预测与 CPU 保留

在 vLLM 0.29.0 的真实 CPU offload 路径上采集访问历史，训练小型 GBDT，比较 LRU、GBDT 写入准入 + LRU、GBDT 写入准入 + 预测淘汰。GPU worker、异步传输和引用保护沿用原生实现。

## 环境与执行

使用独占的 WSL / Linux NVIDIA GPU。沿用 `experiment_raa/01_start_with_prefix_reuse/requirements.txt` 的实验环境，再安装本目录的 `requirements.txt`。固定 Qwen2.5-1.5B-Instruct FP16，revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

从项目根目录执行。`PYTHON` 指向实验解释器，`MODEL` 指向该 revision 的本地快照，`RUNTIME` 指向已有计算缓存目录。临时审阅期间，将 `EXPERIMENT` 替换为候选源码目录。

```bash
EXPERIMENT=experiment_raa/04_kv_reuse_prediction
uv pip install --python "$PYTHON" -r "$EXPERIMENT/requirements.txt"

"$PYTHON" "$EXPERIMENT/run_reuse.py" \
  --model "$MODEL" --runtime "$RUNTIME" \
  --output /var/tmp/kv-reuse-results \
  --collect-rounds 6 --rounds 2 --horizon 12

"$PYTHON" "$EXPERIMENT/review_reuse.py" /var/tmp/kv-reuse-results
```

默认端口为 `127.0.0.1:18083`，可用 `--port` 修改。输出目录必须尚不存在。采集至少 4 轮，评价每个序列至少 2 轮，观察窗口为 1～24 条请求。入口按顺序启动服务，结束或发生异常时停止本次服务进程。执行期间保持源码与 GPU 工作负载不变。

脚本复用第 02 组的 `tier_baseline.py` 采样器和第 03 组的 `run_cost.py` 流式客户端。分开存放时用 `--baseline-dir`、`--cost-dir` 指定这两个目录。当前目录的六个 Python 文件必须放在一起；入口将该目录加入服务进程的 `PYTHONPATH`。

## 训练与策略

每次真实请求到达时，在 CPU 查找前观察完整 prompt。前缀以首个 64-token 块的内容标识，同一标识的长度固定；出现不同长度立即报错。阶段、热点角色和请求编号仅用于复核。

特征为访问次数、距最近访问的请求数、上次访问间隔、历史平均间隔、prompt 长度。模型不接收内容标识或未来访问信息。历史最多保留 4096 个前缀，按最近访问顺序更新；本实验的工作集小于该上限。淘汰评分所需的驻留前缀若已失去历史，直接报错。

标签为当前请求之后 H 条请求内是否再次访问该前缀。采集序列的前半段用于训练，后半段用于验证；训练末尾移除 H 条请求的样本，验证末尾也移除 H 条请求，保证标签观察完整且不跨入验证区间。样本包含每个时刻全部已跟踪前缀，覆盖未接纳、未驻留和已老化的前缀。

GBDT 使用 32 棵深度最多为 3 的树，学习率 0.1，叶子至少 8 个样本。只在验证区间按 F1 选择准入阈值，同分选择较高阈值。模型、阈值和样本在独立测试前冻结；测试使用两个全新前缀序列，策略执行顺序互相反转。

三组均使用 request-level、prompt-only 存储：完整 prompt 的 GPU 命中块也有写入机会，使首次拒绝后再次访问的前缀能够进入 CPU。回迁统一允许，不叠加第 03 组成本门槛。

- `lru`：全部接纳，原生 LRU 淘汰。
- `admission`：当前前缀预测分数达到阈值时接纳，原生 LRU 淘汰。
- `learned`：相同准入规则；容量不足时批量重算空闲驻留前缀分数，先淘汰低分前缀，同分按最久未使用排序，前缀内部从尾块开始淘汰。正在写入的前缀及在途传输块受保护。

写入准入按完整前缀决定。淘汰可截短前缀，保留连续的前部，供原生查找继续恢复。模型错误、未知状态、容量或传输异常直接终止。

## 工作负载与参数

每轮依次包含稳定热点、一次性扫描干扰、热点切换和长间隔复访。热点输入为 1024 token；一次性请求长度从 512、1024、1536 token 中按种子选择。每阶段使用预先保存的真实 token 输入发起推理；各策略回放完全相同的序列。输入用于受控策略实验。

GPU KV 为 64 个 64-token 块，名义容量 112 MiB；CPU 预算 64 MiB，容纳 36 块、2304 token。每轮四个热点加扫描请求超过两级容量。布局为 `LBHNC`，单 worker、单序列、eager，上下文上限 3584，默认输出 32 token。逐 token logprobs 随响应保存。

## 输出与复核

- `environment.json`：模型、设备、实际依赖、参数及执行源码摘要。
- `collect-trace.json`、`evaluate-trace-*.json`：固定 token 输入和请求顺序。
- `collect/`、`e*-*/`：原始响应 JSONL、Prometheus 快照、资源采样、启动参数、服务日志和缓存审计。
- `samples.jsonl`：带请求时钟、特征、标签和时间分区的训练与验证样本。
- `model.joblib`、`model.json`：冻结模型、阈值、特征顺序、训练参数、验证指标及摘要。
- `summary.json`、`validated-requests.jsonl`：复核后分组统计与逐请求派生字段。

只加载本次实验生成且摘要匹配的可信本地 `model.joblib`；joblib 文件能够执行代码，不加载外部提供的模型文件。

复核器重建访问历史和未来标签，重新训练并比较预测结果；逐事件复核准入、淘汰排序、容量、在途保护、写入完成与回迁完成，再从原始指标核对实际传输字节和计算 token。生成 token 或文本不一致时保存差异，并以非零状态退出。

指标口径：

- `hot_resident_ratio`：已访问热点复访到达时，CPU 中就绪 token 数占这些请求完整输入 token 总数的比例。
- `hot_cpu_recovery_ratio`：上述请求实际 CPU 恢复 token 总数，除以其 GPU 未命中 token 总数。GPU 命中单列保留。
- `unused_evicted_bytes`：写入完成后直到被淘汰均未发生实际读回的字节数。
- `pending_observation_bytes`：结束时仍驻留且尚未读回的字节数，后续用途尚待观察。
- `used_evicted_bytes`、`used_resident_bytes`：至少发生一次实际读回的写入，按是否已淘汰区分。四类字节之和等于正式请求的实际写入字节数。
- `paired_saved_median_ms`：同轮同请求的 `LRU TTFT − 当前策略 TTFT` 的中位数，正值表示节省。`epochs` 保留两轮结果。
- `policy_p95_ms`：特征更新、模型评分与淘汰选择耗时之和的最近秩 P95；审计文件写入开销包含在真实请求耗时中。

延迟同时记录客户端 TTFT、服务端 TTFT、P95 与各请求总耗时之和。请求间等待指标发布，统计描述串行回放；请求总耗时之和不包含请求间等待。训练验证指标单列，不代替独立序列上的缓存与延迟结果。

## 接口参考

- [vLLM 0.29.0 CPU manager](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/kv_offload/cpu/manager.py)
- [vLLM 0.29.0 CPU spec](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/kv_offload/cpu/spec.py)
- [scikit-learn GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
