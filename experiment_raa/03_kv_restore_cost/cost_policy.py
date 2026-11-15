"""使用独立校准的首 token 路径成本，决定 CPU KV 回迁准入。"""

import hashlib
import json
import math
from pathlib import Path


def geometry(prompt_tokens, gpu_tokens, cpu_tokens):
    """校验输入总量 prompt_tokens、GPU 命中 gpu_tokens 和 CPU 候选 cpu_tokens，返回查表键。"""
    if any(type(value) is not int for value in (prompt_tokens, gpu_tokens, cpu_tokens)):
        raise ValueError("Token counts must be integers")
    if not 0 <= gpu_tokens < prompt_tokens or not 0 <= cpu_tokens <= prompt_tokens - gpu_tokens:
        raise ValueError("Invalid cache geometry")
    return f"{prompt_tokens}:{gpu_tokens}:{cpu_tokens}"


class CostPolicy:
    """保存校准路径成本；相同查表键包含相同上下文长度和剩余计算量。"""

    def __init__(self, path, margin_ms):
        """从 path 读取已冻结校准表；margin_ms 为回迁所需的最小预计节省。"""
        if not math.isfinite(margin_ms) or margin_ms < 0:
            raise ValueError("margin_ms must be finite and non-negative")
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.margin_seconds = margin_ms / 1000
        profile = json.loads(raw)
        if profile["schema"] != 1 or profile["vllm"] != "0.29.0":
            raise ValueError("Unsupported calibration profile")
        self.context = profile["context"]
        self.entries = profile["entries"]
        if not self.entries:
            raise ValueError("Empty calibration profile")
        for key, entry in self.entries.items():
            if key != geometry(entry["prompt_tokens"], entry["gpu_tokens"], entry["cpu_tokens"]):
                raise ValueError("Calibration geometry mismatch")
            if entry["cpu_tokens"] == 0 or entry["n"] < 2:
                raise ValueError("Insufficient calibration samples")
            for field in ("restore_seconds", "recompute_seconds"):
                value = entry[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid calibrated path cost")

    def decide(self, prompt_tokens, gpu_tokens, cpu_tokens):
        """根据 prompt_tokens、gpu_tokens、cpu_tokens 返回两条路径成本及准入结果；未校准几何直接报错。"""
        key = geometry(prompt_tokens, gpu_tokens, cpu_tokens)
        entry = self.entries[key]
        saved = entry["recompute_seconds"] - entry["restore_seconds"]
        return dict(restore_seconds=entry["restore_seconds"],
                    recompute_seconds=entry["recompute_seconds"],
                    expected_saved_seconds=saved, margin_seconds=self.margin_seconds,
                    admit=saved > self.margin_seconds)
