# 实验 13：CPU KV 保留时间价值

## 真实历史离线回放

实验 08 `pressure-14` 的首轮 CPU `learned` 审计，包含 1 条预热和 50 条正式请求。所有策略接收相同的键链，使用实验 06 冻结 GBDT。反事实计算假定 GPU 冷、CPU 写入立即完成。时间系数单独取实验 06 的真实配对校准：1 块、8 块各 4 对，重算减 CPU 回迁的服务端 TTFT 中位差为 15.896、40.196 ms；实测 GPU→CPU 写入估计为 1.303 ms/块。历史校准 CPU 容量 32 块，历史回放只有 8 块；该估计仅用于离线回放。

| 策略 | CPU 写入块 | CPU 淘汰块 | CPU 命中 token | 前缀重算 token |
| --- | ---: | ---: | ---: | ---: |
| `admission` | 81 | 73 | 11,392 | 15,283 |
| `learned` | 81 | 73 | 10,816 | 15,859 |
| `time_value` | 18 | 10 | 3,200 | 23,475 |

`time_value` 的 39 次计价中有 29 次缩短或拒绝写入；相对 `learned` 少写 63 块，多重算 7,616 token。此处不报告端到端 TTFT：CPU 命中为基于真实历史输入的反事实。

原始结果：`.agent/kv-time-value/historical/{review.json,pilot.json,time_value.jsonl}`。
