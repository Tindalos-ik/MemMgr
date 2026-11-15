"""从真实路径计时冻结成本模型，比较连续前缀在 CPU、SSD 与丢弃时的收益。"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


PATHS = ("compute", "cpu_read", "cpu_write", "ssd_read", "ssd_write")


def cost_context(context):
    """从 context 提取各策略共用的设备、数值路径、容量与实现身份。"""
    return {key: value for key, value in context.items() if key not in ("policy", "model_sha256")}


def fit_costs(observations, context, max_blocks, sources):
    """用 observations 的真实作业数、块数与秒数拟合非负路径成本，保留 context 和 sources。"""
    models = {}
    for name in PATHS:
        rows = observations[name]
        if len(rows) < 4 or len({row["blocks"] / row["jobs"] for row in rows}) < 2:
            raise ValueError(f"Insufficient distinct real transfer sizes for {name}")
        matrix = np.asarray([[row["jobs"], row["blocks"]] for row in rows], dtype=float)
        seconds = np.asarray([row["seconds"] for row in rows], dtype=float)
        if (not np.isfinite(matrix).all() or not np.isfinite(seconds).all()
                or (matrix <= 0).any() or (seconds <= 0).any()):
            raise ValueError("Invalid measured path timing")
        coefficients, _ = nnls(matrix, seconds)
        models[name] = dict(job_seconds=float(coefficients[0]), block_seconds=float(coefficients[1]),
                            samples=len(rows), median_blocks=float(np.median(matrix[:, 1] / matrix[:, 0])),
                            median_absolute_error_seconds=float(np.median(abs(matrix @ coefficients - seconds))))
    return dict(schema=1, context=cost_context(context), max_blocks=max_blocks,
                paths=models, observations=observations, sources=sources)


class PlacementCosts:
    """读取冻结的非负仿射成本；超出校准长度或环境不匹配直接终止。"""

    def __init__(self, path, context):
        """从可信本地 path 加载成本表，并核对当前 context。"""
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.profile = json.loads(raw)
        if self.profile["schema"] != 1 or self.profile["context"] != cost_context(context):
            raise ValueError("Placement costs and runtime context differ")
        self.max_blocks = self.profile["max_blocks"]
        if type(self.max_blocks) is not int or not 1 <= self.max_blocks <= 4096:
            raise ValueError("Invalid calibrated block range")
        for name in PATHS:
            model = self.profile["paths"][name]
            values = [model["job_seconds"], model["block_seconds"], model["median_blocks"]]
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                    or min(values) < 0 or values[2] <= 0 or sum(values[:2]) <= 0
                    or model["samples"] < 4):
                raise ValueError("Invalid calibrated path coefficients")

    def seconds(self, name, blocks):
        """返回 name 路径一次传输 blocks 个块的秒数；零块不产生作业。"""
        if type(blocks) is not int or not 0 <= blocks <= self.max_blocks:
            raise ValueError("Prefix exceeds the calibrated placement range")
        model = self.profile["paths"][name]
        return model["job_seconds"] + model["block_seconds"] * blocks if blocks else 0.0

    def unit(self, name):
        """以 name 的校准中位作业大小分摊固定开销，返回驻留块的替换价值密度。"""
        model = self.profile["paths"][name]
        return model["block_seconds"] + model["job_seconds"] / model["median_blocks"]

    def gain(self, probabilities, positions, tier, cpu_writes, ssd_writes=0):
        """计算 probabilities/positions 在 tier 的收益；cpu_writes/ssd_writes 为新增传输块数。"""
        if not probabilities:
            return 0.0
        expected = sum(probability * (self.seconds("compute", position) - self.seconds("compute", position - 1))
                       for probability, position in zip(probabilities, positions))
        probability = sum(probabilities) / len(probabilities)
        restore = self.seconds("cpu_read", len(probabilities))
        write = self.seconds("cpu_write", cpu_writes)
        if tier == "ssd":
            restore += self.seconds("ssd_read", len(probabilities))
            write += self.seconds("ssd_write", ssd_writes)
        elif tier != "cpu":
            raise ValueError("Unknown placement tier")
        return expected - probability * restore - write


@dataclass
class PlacementDecision:
    """将一次请求的冻结放置决定附在原生 ReqContext，延续到异步传输完成。"""

    targets: dict = field(default_factory=dict)
    finished: bool = False
