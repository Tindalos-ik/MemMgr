"""拟合独立 KV 路径实测，按连续前缀深度估计保留的边际时间。"""

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess


SCHEMA = 1
CONTEXT_KEYS = ("identity", "gpu", "packages", "native", "cpu_blocks", "gpu_blocks", "layout",
                "enforce_eager", "batch_invariant", "gpu_memory_utilization", "gpu_bytes",
                "max_model_len", "source")


def retention_context(context, tier, device, env=None):
    """提取 context、tier/device 和实际服务 env 所确定的路径成本身份。"""
    # GBDT 的策略与模型摘要单独审计；校准用 LRU 保证探针命中所需的前缀。
    cpu_info = Path("/proc/cpuinfo").read_text()
    cpu = sorted({line.strip() for line in cpu_info.splitlines()
                  if line.partition(":")[0].strip() in {"model name", "microcode"}})
    if env is None:
        env = os.environ
    return {key: context[key] for key in CONTEXT_KEYS} | dict(
        tier=tier, device=device, cpu=cpu, kernel=os.uname().release,
        omp_threads=env.get("OMP_NUM_THREADS"),
        visible_cuda=env.get("CUDA_VISIBLE_DEVICES"))


def filesystem_device(root):
    """返回 root 所在 Linux 文件系统的实际设备身份和挂载选项。"""
    return subprocess.check_output(
        ["findmnt", "-n", "-o", "MAJ:MIN,SOURCE,FSTYPE,OPTIONS", "-T", str(root)], text=True,
    ).strip()


def depth_rates(anchors):
    """把 anchors 中实测累计差值转换成逐深度边际量，锚点之间使用线性插值。"""
    rates = [None]
    last_depth, last_value = 0, 0.0
    for depth, value in anchors:
        rates.extend([(value - last_value) / (depth - last_depth)] * (depth - last_depth))
        last_depth, last_value = depth, value
    return rates


def best_prefix(costs, victims, free):
    """在每个新写入处比较 costs 的累计时间收益、写入和牺牲 victims，free 是剩余空位。"""
    best = (0, 0.0, 0.0, 0.0)
    benefit = written = displaced = 0.0
    best_net = 0.0
    for index, (gain, write) in enumerate(costs, 1):
        needed = index - free
        if needed > len(victims):
            break
        if needed > 0:
            displaced += victims[needed - 1]
        benefit += gain
        written += write
        net = benefit - written - displaced
        if net > best_net:
            best_net = net
            best = (index, benefit, written, displaced)
    return best


class RetentionProfile:
    """只接受同一设备、路径、布局和运行配置下完整校准的前缀深度。"""

    def __init__(self, path, context):
        """读取 path，验证实测身份与运行 context，构造唯一支持的深度范围。"""
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        profile = json.loads(raw)
        if profile["schema"] != SCHEMA or profile["context"] != context:
            raise ValueError("Retention profile does not match the runtime device and configuration")
        self.has_ssd = context["tier"] == "ssd"
        self.calibration_prefixes = frozenset(profile["calibration_prefix_sha256"])
        if not self.calibration_prefixes or not profile["sources"]:
            raise ValueError("Retention profile has no calibration provenance")
        rows = profile["anchors"]
        if not rows or rows[0]["blocks"] != 1:
            raise ValueError("Retention calibration must start at one block")
        fields = ("recompute_gain", "cpu_write") + (("ssd_gain", "ssd_write") if self.has_ssd else ())
        previous = 0
        for row in rows:
            depth, samples, prompt = row["blocks"], row["samples"], row["prompt_tokens"]
            if (type(depth) is not int or type(samples) is not int or type(prompt) is not int
                    or not previous < depth <= context["cpu_blocks"] or samples < 3
                    or not depth * context["identity"]["block_tokens"] < prompt < context["max_model_len"]):
                raise ValueError("Invalid retention depth, prompt geometry or sample count")
            for name in fields:
                value = row[name]
                if type(value) not in (int, float) or not math.isfinite(value) or (name.endswith("write") and value <= 0):
                    raise ValueError(f"Invalid measured retention cost: {name}")
            previous = depth
        self.max_depth = rows[-1]["blocks"]
        # DMA 写入使用按组实测的每块均值；TTFT 用累计差的相邻斜率。
        self.recompute = depth_rates([(row["blocks"], row["recompute_gain"]) for row in rows])
        self.ssd = depth_rates([(row["blocks"], row["ssd_gain"]) for row in rows]) if self.has_ssd else None
        self.cpu_write = median(row["cpu_write"] / row["blocks"] for row in rows)
        self.ssd_write = median(row["ssd_write"] / row["blocks"] for row in rows) if self.has_ssd else 0.0

    def gain(self, depth, ssd):
        """返回 depth 处 CPU 避免 ssd 回读或重算的边际秒数，允许负收益。"""
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"Retention depth {depth} outside calibrated range 1..{self.max_depth}")
        return (self.ssd if ssd else self.recompute)[depth]


def fit(samples, context, prefixes, sources):
    """以 samples 中独立同 token 配对拟合 profile；prefixes/sources 保留校准来源。"""
    tiers = {"recompute", "cpu", "ssd"} if context["tier"] == "ssd" else {"recompute", "cpu"}
    cases = defaultdict(dict)
    for row in samples:
        if row["phase"] != "retention-calibration" or row["path"] in cases[row["case"]]:
            raise ValueError("Unexpected or duplicate retention calibration observation")
        cases[row["case"]][row["path"]] = row
    groups = defaultdict(list)
    for paths in cases.values():
        if paths.keys() != tiers:
            raise ValueError("Missing paired retention calibration path")
        cpu, cold = paths["cpu"], paths["recompute"]
        for row in paths.values():
            for field in ("blocks", "prompt_tokens", "prompt_sha256", "generated_tokens"):
                if row[field] != cpu[field]:
                    raise ValueError(f"Unmatched calibration request: {cpu['case']}, {field}")
        sample = dict(recompute_gain=cold["ttft_seconds"] - cpu["ttft_seconds"], cpu_write=cpu["write_seconds"])
        if "ssd" in paths:
            sample.update(ssd_gain=paths["ssd"]["ttft_seconds"] - cpu["ttft_seconds"],
                          ssd_write=paths["ssd"]["ssd_write_seconds"])
        groups[cpu["blocks"], cpu["prompt_tokens"]].append(sample)
    anchors = []
    for (blocks, prompt), values in sorted(groups.items()):
        if len(values) < 3 or (anchors and blocks == anchors[-1]["blocks"]):
            raise ValueError("Each retention depth needs one geometry and at least three complete pairs")
        anchors.append(dict(blocks=blocks, prompt_tokens=prompt, samples=len(values),
                            **{key: median(row[key] for row in values) for key in values[0]}))
    if not anchors:
        raise ValueError("No paired retention calibration data")
    return dict(schema=SCHEMA, context=context, anchors=anchors,
                calibration_prefix_sha256=sorted(prefixes), sources=sources)
