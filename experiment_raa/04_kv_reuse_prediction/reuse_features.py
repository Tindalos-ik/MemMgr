"""记录有界前缀访问历史，并加载独立训练、冻结的 GBDT 复用预测器。"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


FEATURES = ["count", "recency", "last_gap", "mean_gap", "prompt_tokens"]
HISTORY_LIMIT = 4096


def digest(path):
    """返回 path 文件的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """将 value 以可读 UTF-8 JSON 写入 path。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path):
    """读取 path 的完整 JSONL；空文件或不完整尾行直接报错。"""
    raw = Path(path).read_text(encoding="utf-8")
    if not raw or not raw.endswith("\n"):
        raise ValueError(f"Incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


class History:
    """按请求时钟维护前缀访问统计，容量达到上限后移除最久未访问的记录。"""

    def __init__(self):
        """初始化空历史；时钟在每次真实请求到达时递增。"""
        self.clock = -1
        self.entries = OrderedDict()

    def observe(self, root, length):
        """记录 root 的一次访问；length 为完整 prompt token 数。"""
        self.clock += 1
        old = self.entries.pop(root, None)
        if old is None:
            entry = [1, self.clock, 0, 0.0, length]
        else:
            count, last, _, mean, previous_length = old
            if previous_length != length:
                raise ValueError("This experiment requires a fixed length per first-block identity")
            gap = self.clock - last
            entry = [count + 1, self.clock, gap, mean + (gap - mean) / count, length]
        self.entries[root] = entry
        if len(self.entries) > HISTORY_LIMIT:
            self.entries.popitem(last=False)
        return self.features(root)

    def features(self, root):
        """返回 root 在当前请求时钟的特征；未跟踪的驻留前缀直接报错。"""
        count, last, gap, mean, length = self.entries[root]
        return [count, self.clock - last, gap, mean, length]


class Predictor:
    """只加载本次实验生成且摘要匹配的本地模型。"""

    def __init__(self, directory):
        """从可信 directory 加载元数据及 joblib 模型；禁止使用外部来源的模型文件。"""
        directory = Path(directory)
        self.metadata = json.loads((directory / "model.json").read_text())
        self.sha256 = digest(directory / "model.joblib")
        if self.metadata["features"] != FEATURES or self.sha256 != self.metadata["model_sha256"]:
            raise ValueError("Model schema or digest mismatch")
        self.model = joblib.load(directory / "model.joblib")
        if list(self.model.classes_) != [0, 1] or self.model.n_features_in_ != len(FEATURES):
            raise ValueError("Unexpected GBDT model")
        self.threshold = self.metadata["threshold"]

    def score(self, features):
        """批量预测 features 中各行在观察窗口内再次访问的概率。"""
        result = self.model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]
        if not np.isfinite(result).all():
            raise ValueError("Non-finite reuse probability")
        return result.tolist()
