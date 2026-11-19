# SSD 回迁成本实验结果

## 07 对照来源

使用 07 的 `ssd-lru`，GPU 32 MiB、CPU 56 MiB、每块 1,835,008 字节。`epoch-0` 与 `epoch-1` 各有 6 条同请求 SSD→CPU→GPU 回迁链请求、27 个完成回迁的块。两轮的确定性复访索引相同：12、32、70 各 8 块；19、39、77 各 1 块。09 校准保留这些案例的原始输入。

数据：`.agent/kv-joint-ssd/output/`；`review.json` 的 SHA-256 为 `75cba28c052bd0080522e5759bcac6f8ed19bf9f938e48c6f4936f6036ce24a8`。

## 文件层验收

Linux 来宾使用 `/dev/sdd` 上的 ext4。Windows 侧核对 WSL 的 `ext4.vhdx` 位于 `C:`，其磁盘编号 0 对应 `HFS512GEJ9X125N` NVMe SSD。宿主介质证据见 `.agent/kv-ssd-cost/host-storage.txt`。来宾 `O_DIRECT` 和 Linux I/O 计数不覆盖宿主缓存及 NVMe 控制器。

独立文件层验收使用 vLLM 原生共享内存与 `O_DIRECT`：相同 1 个 CPU 候选和 2 个 SSD 候选，允许分支完成 2 次提升并读取 3,670,016 字节，加载内容逐字节一致；拒绝分支无提升、无读任务，两侧 CPU 块均命中。该验收未运行 GPU 或模型，原始审计见 `.agent/kv-ssd-cost/local-verification.log`。
