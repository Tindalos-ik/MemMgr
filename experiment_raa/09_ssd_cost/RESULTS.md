# SSD 回迁成本实验结果

## 07 对照来源

使用 07 的 `ssd-lru`，GPU 32 MiB、CPU 56 MiB、每块 1,835,008 字节。`epoch-0` 与 `epoch-1` 各有 6 条同请求 SSD→CPU→GPU 回迁链请求、27 个完成回迁的块。两轮的确定性复访索引相同：12、32、70 各 8 块；19、39、77 各 1 块。09 校准保留这些案例的原始输入。

数据：`.agent/kv-joint-ssd/output/`；`review.json` 的 SHA-256 为 `75cba28c052bd0080522e5759bcac6f8ed19bf9f938e48c6f4936f6036ce24a8`。
