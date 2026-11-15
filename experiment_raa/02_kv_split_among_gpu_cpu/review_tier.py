"""核对真实分层实验的完整性，汇总配对延迟、迁移量和容量失效条件。"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read_rows(path):
    """从 path 读取完整 JSONL 记录，拒绝缺少换行的末条记录。"""
    content = path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in content.splitlines()]


def median(rows, name):
    """返回 rows 中 name 字段的中位数。"""
    return statistics.median(row[name] for row in rows)


def report(root, result, rows):
    """依据 root 的 result 汇总及 rows 原始记录生成实验报告。"""
    n = result["groups"][0]["n"]
    lines = ["# KV 分层缓存压力实验", "", "日期：2026-11-11。", "",
             "## 实验条件", "",
             "RTX 5070 Laptop GPU 8 GiB，WSL Ubuntu，Qwen2.5-1.5B-Instruct FP16；"
             "vLLM 0.29.0、PyTorch 2.13.0。模型 revision 与完整软件环境见 `environment.json`。", "",
             "GPU KV 名义容量 112 MiB，64 个 64-token 块，扣除空块可用 4032 token。"
             "CPU 分别为 0、64、512 MiB，原生 LRU，即时复制 prompt KV；各组统一 LBHNC 布局与 DMA 传输条件。", "",
             "每 token KV 为 28 层 × K/V 两份 × 2 个 KV head × 128 维 × FP16 2 字节，合计 28672 字节。"
             "每块 1.75 MiB；CPU64 可存 36 块 / 2304 token，CPU512 可存 292 块 / 18688 token。", "",
             "每个用例先发送热请求，再发送 0、2、6 条独立的 1024-token 干扰请求，最后复访热请求。"
             f"每轮每种条件重复 {n // 2} 次，共两轮；第二轮反转三种配置的执行顺序。"
             f"同轮配置回放相同输入，第二轮使用新的前缀。每条输出固定 {result['output_tokens']} token。", "",
             f"正式请求共 {result['requests']} 条，其中复访 {result['revisits']} 条；每个配置、长度、间隔组合 n={n}。"
             "预热覆盖三种 prefill 长度、decode 和回迁，不纳入正式统计。", "",
             "## 完全淘汰后的复访", "",
             "间隔为 6 条干扰请求。单位均为 ms，TTFT 为客户端收到首个非空正文块的时间。", "",
             "| 输入 token | GPU-only TTFT | CPU512 TTFT | CPU64 TTFT | CPU512 配对节省 | CPU512 回迁 MiB | CPU512 传输时间 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    lookup = {(g["length"], g["gap"], g["mode"]): g for g in result["groups"]}
    for length in (256, 1024, 3072):
        gpu, large, small = [lookup[(length, 6, mode)] for mode in ("gpu", "cpu512", "cpu64")]
        lines.append(f"| {length} | {gpu['ttft_ms']:.2f} | {large['ttft_ms']:.2f} | {small['ttft_ms']:.2f} | "
                     f"{large['paired_ttft_saved_ms']:.2f} | {large['load_MiB']:.2f} | {large['load_ms']:.2f} |")
    small_evicted = [r for r in rows if r["mode"] == "cpu64" and r["phase"] == "revisit" and r["gap"] == 6]
    small_misses = sum(r["external_hits"] == 0 for r in small_evicted)
    lines += ["", "配对节省先对同轮、同用例求 GPU-only TTFT 减 CPU512 TTFT，再取中位数；它与两组中位数之差可有区别。", "",
              "## 策略设计依据", "",
              f"CPU64 在长间隔条件下有 {small_misses}/{len(small_evicted)} 次复访未命中 CPU。"
              "在当前即时复制、LRU 和顺序访问模式下，一次性干扰请求消耗写入带宽并挤占 CPU 保留空间。", "",
              "零干扰时，GPU 留有绝大部分前缀，CPU512 仍为剩余的 64-token 块执行回迁。"
              "是否回迁应结合剩余未命中 token 的重算成本与传输、调度成本判断；完整前缀长度和总命中率不能单独决定收益。", "",
              "后续策略可分别控制写入准入与读回准入：用重用频率限制一次性 KV 写入，"
              "再依据剩余未命中长度和实测成本决定读回或重算。当前提交提供真实存取路径与成本记录。", "",
              "## 全部条件", "",
              f"| 输入 token | 干扰请求数 | 配置 | TTFT ms | GPU 命中 token | CPU 命中 token | 计算 token | 回迁 MiB | 配对节省 ms | 配对获益次数 / {n} |",
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in result["groups"]:
        lines.append(f"| {group['length']} | {group['gap']} | {group['mode']} | {group['ttft_ms']:.2f} | "
                     f"{group['gpu_hit_tokens']:.0f} | {group['external_hit_tokens']:.0f} | {group['computed_tokens']:.0f} | "
                     f"{group['load_MiB']:.2f} | {group['paired_ttft_saved_ms']:.2f} | {group['paired_wins']} |")
    lines += ["", "## 存储与传输代价", "",
              "下表累加两轮全部正式请求，包含首次请求、干扰请求和复访；资源采样峰值包含服务启动。", "",
              "| 配置 | GPU→CPU GiB | CPU→GPU GiB | GPU→CPU 累计传输 s | CPU→GPU 累计传输 s | 进程树 RSS 采样峰值 GiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode in ("gpu", "cpu512", "cpu64"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        resources = [run for run in result["runs"] if run["mode"] == mode]
        lines.append(f"| {mode} | {sum(r['store_bytes'] for r in mode_rows)/2**30:.3f} | "
                     f"{sum(r['load_bytes'] for r in mode_rows)/2**30:.3f} | "
                     f"{sum(r['store_seconds'] for r in mode_rows):.3f} | {sum(r['load_seconds'] for r in mode_rows):.3f} | "
                     f"{max(r['peak_process_tree_rss'] for r in resources)/2**30:.3f} |")
    lines += ["", "传输时间使用引擎 CUDA event 计数。传输可与计算重叠，这些累计值不能直接与请求延迟相加。"
              "CPU 分配预算与进程 RSS 含义不同；进程树 RSS 可能重复统计共享页。", "",
              "## 完整性", "",
              f"输入身份、逐请求 token 用量、结束原因、请求数、原始快照及无抢占条件均通过核对。"
              f"输出差异共 {len(result['output_mismatches'])} 处，详细列表见 `summary.json`。", "",
              "## 解释范围", "",
              "输入使用受控文本，各用例首块不同，测量涵盖真实 vLLM 推理和 GPU / CPU KV 存取。"
              "该实验用于研究缓存容量与传输成本；问答正确性及语义长期记忆检索需单独评价。", "",
              f"请求之间等待统计发布，数据不用于推断连续负载吞吐。每个组合有 {n} 次复访，"
              "GPU 频率、Windows 显示负载和固定模式顺序均可能影响延迟；正反顺序用于检查方向是否稳定。", "",
              "## 复算", "", "```bash", "python review_tier.py /结果目录", "```", "",
              "`trace-*.json` 保存全部输入；`e*-*/requests.jsonl` 保存逐请求结果；编号 `*.prom` 保存完整计数器快照。"
              "运行指南见上级 `README.md`。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_short(root, result):
    """将 root 中的短前缀补测 result 写为独立报告，保留干扰请求长度差异。"""
    affected = {(item["epoch"], item["mode"], item["index"]) for item in result["output_mismatches"]}
    lines = ["# 短前缀回迁成本补测", "",
             "沿用正式实验的模型、GPU 容量、块大小、布局、32-token 输出与计时方法。"
             "输入长度为 64、128 token，复访前插入两条独立的 3072-token 请求，合计 6144 token。"
             "干扰请求的条数和长度与主实验不同，结果单独报告。", "",
             f"正式请求 {result['requests']} 条，复访 {result['revisits']} 条；每种长度、配置 n=10。"
             "同轮输入相同，两轮执行顺序为 GPU→CPU512、CPU512→GPU。", "",
             "| 输入 token | 配置 | TTFT ms | GPU 命中 token | CPU 命中 token | 计算 token | 回迁 MiB | 传输 ms | 配对节省 ms | 配对获益次数 / 10 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in result["groups"]:
        lines.append(f"| {g['length']} | {g['mode']} | {g['ttft_ms']:.2f} | {g['gpu_hit_tokens']:.0f} | "
                     f"{g['external_hit_tokens']:.0f} | {g['computed_tokens']:.0f} | {g['load_MiB']:.2f} | "
                     f"{g['load_ms']:.2f} | {g['paired_ttft_saved_ms']:.2f} | {g['paired_wins']} |")
    lines += ["", f"输入与计数器完整性通过核对；有 {len(affected)} 条响应出现文本差异，"
              f"触发 {len(result['output_mismatches'])} 次一致性检查。详细位置见 `summary.json`。", "",
              "配对节省为同轮 GPU TTFT 减 CPU512 TTFT，正值表示回迁更快。详细配对范围及两轮分别统计见 `summary.json`。", "",
              "该补测定位本机、当前 DMA 与串行条件下的成本转折；它不构成通用 token 阈值。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def review(root, short=False):
    """校验 root 的两轮数据；short 指定独立的短前缀补测，随后写出汇总。"""
    config_path = root / ("arguments.json" if short else "environment.json")
    output_tokens = json.loads(config_path.read_text())["arguments"]["output_tokens"]
    all_rows = []
    identities = {}
    mismatches = []
    runs = []
    for epoch in (0, 1):
        plan = json.loads((root / f"trace-{epoch}.json").read_text())
        for mode in (("gpu", "cpu512") if short else ("gpu", "cpu512", "cpu64")):
            directory = root / f"e{epoch}-{mode}"
            rows = read_rows(directory / "requests.jsonl")
            if len(rows) != len(plan):
                raise ValueError(f"Missing requests: {directory}")
            if not (directory / "final.prom").is_file():
                raise ValueError(f"Missing final metrics: {directory}")
            cold = {}
            for index, (row, expected) in enumerate(zip(rows, plan)):
                tokens = expected["prompt_token_ids"]
                digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                if row["index"] != index or row["prompt_sha256"] != digest:
                    raise ValueError("Replay identity mismatch")
                if row["usage"]["prompt_tokens"] != len(tokens) or row["finish_reason"] != "length":
                    raise ValueError("Invalid token usage or termination")
                key = (epoch, index)
                identity = (row["answer"], row["usage"])
                if mode == "gpu":
                    identities[key] = identity
                elif identities[key] != identity:
                    mismatches.append(dict(epoch=epoch, mode=mode, index=index, type="cross-mode"))
                if row["phase"] == "cold":
                    cold[row["case"]] = row
                if row["phase"] == "revisit" and row["answer"] != cold[row["case"]]["answer"]:
                    mismatches.append(dict(epoch=epoch, mode=mode, index=index, type="cold-revisit"))
                counters = row["metrics"]
                if row["usage"]["completion_tokens"] != output_tokens or counters["vllm:num_preemptions_total"] != 0:
                    raise ValueError("Unexpected output length or active-request preemption")
                if row["phase"] in ("cold", "filler") and counters["vllm:prefix_cache_hits_total"] != 0:
                    raise ValueError("Unique prompt unexpectedly hit GPU cache")
                row.update(epoch=epoch, mode=mode,
                           gpu_hits=counters["vllm:prefix_cache_hits_total"],
                           external_hits=counters.get("vllm:external_prefix_cache_hits_total", 0),
                           load_bytes=counters.get("vllm:kv_offload_load_bytes_total", 0),
                           load_seconds=counters.get("vllm:kv_offload_load_time_total", 0),
                           store_bytes=counters.get("vllm:kv_offload_store_bytes_total", 0),
                           store_seconds=counters.get("vllm:kv_offload_store_time_total", 0),
                           computed_tokens=counters["vllm:request_prefill_kv_computed_tokens_sum"],
                           prefill_seconds=counters["vllm:request_prefill_time_seconds_sum"],
                           server_ttft_seconds=counters["vllm:time_to_first_token_seconds_sum"],
                           queue_seconds=counters["vllm:request_queue_time_seconds_sum"])
                if row["computed_tokens"] + row["gpu_hits"] + row["external_hits"] != len(tokens):
                    raise ValueError("Computed and cached token counts do not cover the input")
                if row["load_bytes"] != row["external_hits"] * 28672:
                    raise ValueError("CPU transfer bytes do not match the model KV size")
                if row["phase"] in ("cold", "filler") and row["external_hits"] != 0:
                    raise ValueError("Unique prompt unexpectedly hit CPU cache")
                if mode == "gpu" and row["phase"] == "revisit" and row["gap"] == 6 and row["gpu_hits"] != 0:
                    raise ValueError("GPU pressure failed to evict the old prefix")
                if mode == "cpu512" and row["phase"] == "revisit" and row["gap"] == 6:
                    if row["external_hits"] != len(tokens):
                        raise ValueError("Large CPU cache failed to restore the complete prefix")
                if short and row["phase"] == "revisit":
                    if row["gpu_hits"] != 0 or row["external_hits"] != (len(tokens) if mode == "cpu512" else 0):
                        raise ValueError("Short-prefix eviction or promotion failed")
                all_rows.append(row)
            resources = read_rows(directory / "resources.jsonl")
            runs.append(dict(epoch=epoch, mode=mode, requests=len(rows),
                             peak_process_tree_rss=max(row["process_tree_rss"] for row in resources),
                             minimum_host_available=min(row["host_available"] for row in resources),
                             load_bytes=sum(row["load_bytes"] for row in rows),
                             store_bytes=sum(row["store_bytes"] for row in rows)))
    groups = []
    for length in ((64, 128) if short else (256, 1024, 3072)):
        for gap in ((2,) if short else (0, 2, 6)):
            for mode in (("gpu", "cpu512") if short else ("gpu", "cpu512", "cpu64")):
                group = [row for row in all_rows if row["length"] == length and row["gap"] == gap
                         and row["phase"] == "revisit" and row["mode"] == mode]
                baseline = {(row["epoch"], row["case"]): row for row in all_rows
                            if row["mode"] == "gpu" and row["phase"] == "revisit"}
                paired_ms = [(baseline[(row["epoch"], row["case"])]["ttft_seconds"] - row["ttft_seconds"]) * 1000
                             for row in group]
                epoch_saved = {str(epoch): statistics.median(
                    (baseline[(row["epoch"], row["case"])]["ttft_seconds"] - row["ttft_seconds"]) * 1000
                    for row in group if row["epoch"] == epoch) for epoch in (0, 1)}
                groups.append(dict(length=length, gap=gap, mode=mode, n=len(group),
                                   ttft_ms=median(group, "ttft_seconds") * 1000,
                                   total_ms=median(group, "total_seconds") * 1000,
                                   gpu_hit_tokens=median(group, "gpu_hits"),
                                   external_hit_tokens=median(group, "external_hits"),
                                   load_MiB=median(group, "load_bytes") / 2**20,
                                   load_ms=median(group, "load_seconds") * 1000,
                                   computed_tokens=median(group, "computed_tokens"),
                                   prefill_ms=median(group, "prefill_seconds") * 1000,
                                   server_ttft_ms=median(group, "server_ttft_seconds") * 1000,
                                   queue_ms=median(group, "queue_seconds") * 1000,
                                   paired_ttft_saved_ms=statistics.median(paired_ms),
                                   paired_saved_range_ms=[min(paired_ms), max(paired_ms)],
                                   epoch_paired_saved_ms=epoch_saved,
                                   paired_wins=sum(value > 0 for value in paired_ms)))
    result = dict(runs=runs, groups=groups, output_mismatches=mismatches,
                  output_tokens=output_tokens, requests=len(all_rows),
                  revisits=sum(row["phase"] == "revisit" for row in all_rows))
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "validated-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in all_rows), encoding="utf-8")
    if short:
        report_short(root, result)
    else:
        report(root, result, all_rows)
    print(json.dumps(result, indent=2))
    if mismatches:
        raise RuntimeError("Output identity check failed; inspect summary.json and REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    review(args.root, args.short)
