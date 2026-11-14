"""按连续前缀块维护有界访问历史并读取冻结的 GBDT。"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

FEATURES = ["count", "recency", "last_gap", "mean_gap", "prefix_tokens"]
HISTORY_LIMIT = 4096


class History:
    """同一请求中的每个完整前缀块各计一次访问，共享请求时钟。"""

    def __init__(self):
        """初始化空的连续前缀访问记录。"""
        self.clock = -1
        self.entries = OrderedDict()

    def observe(self, keys, block_tokens, protected=()):
        """观察 keys；block_tokens 为块长，protected 是必须保留统计的驻留块。"""
        self.clock += 1
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate chained prefix key")
        for index, key in enumerate(keys):
            length = (index + 1) * block_tokens
            old = self.entries.pop(key, None)
            if old is None:
                entry = [1, self.clock, 0, 0.0, length]
            else:
                count, last, _, mean, previous_length = old
                if length != previous_length:
                    raise ValueError("Chained prefix key changed position")
                gap = self.clock - last
                entry = [count + 1, self.clock, gap, mean + (gap - mean) / count, length]
            self.entries[key] = entry
        keep = set(protected) | set(keys)
        removable = [key for key in self.entries if key not in keep]
        excess = max(0, len(self.entries) - HISTORY_LIMIT)
        if excess > len(removable):
            raise ValueError("Resident prefix statistics exceed history capacity")
        for key in removable[:excess]:
            del self.entries[key]

    def features(self, key):
        """返回 key 在当前请求时钟下的历史特征，缺失记录直接报错。"""
        count, last, gap, mean, length = self.entries[key]
        return [count, self.clock - last, gap, mean, length]


class Predictor:
    """只读取调用者确认可信的本地目录，不接收远程模型地址。"""

    def __init__(self, directory, identity):
        """从 directory 加载模型并核对 identity、特征定义及文件摘要。"""
        directory = Path(directory)
        self.metadata = json.loads((directory / "model.json").read_text())
        self.sha256 = hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
        if (self.metadata["schema"] != 2 or self.metadata["features"] != FEATURES
                or self.metadata["identity"] != identity
                or self.metadata["model_sha256"] != self.sha256):
            raise ValueError("GBDT identity, schema or digest mismatch")
        self.model = joblib.load(directory / "model.joblib")
        if list(self.model.classes_) != [0, 1] or self.model.n_features_in_ != len(FEATURES):
            raise ValueError("Unsupported GBDT model")
        self.threshold = self.metadata["threshold"]
        if not isinstance(self.threshold, (float, int)) or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid GBDT threshold")

    def score(self, vectors):
        """批量预测 vectors 中各块在观察窗口内复用的概率。"""
        if not vectors:
            return []
        values = self.model.predict_proba(np.asarray(vectors, dtype=np.float32))[:, 1]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid reuse probability")
        return values.tolist()
