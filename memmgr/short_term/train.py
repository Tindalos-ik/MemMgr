"""从真实块访问事件训练时间隔离的 GBDT，并冻结模型与训练来源。"""

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score

from .reuse import FEATURES, History


def train(audit, output, horizon=12, seed=53, request_limit=None):
    """用 audit 训练并写入 output；horizon/seed 固定标签和训练，request_limit 限定采集前段。"""
    raw = Path(audit).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete audit JSONL")
    events = [json.loads(line) for line in raw.splitlines()]
    initial = [row for row in events if row["kind"] == "init"]
    if len(initial) != 1 or initial[0]["policy"] != "lru":
        raise ValueError("Require one complete LRU collection session")
    requests = [row for row in events if row["kind"] == "begin"]
    if request_limit is not None:
        if type(request_limit) is not int or not 1 <= request_limit <= len(requests):
            raise ValueError("Invalid training request limit")
        requests = requests[:request_limit]
    if not 1 <= horizon <= 128 or len(requests) < 4 * horizon + 4:
        raise ValueError("Insufficient complete observations for chronological split")
    future = defaultdict(list)
    for row in requests:
        for key in row["keys"]:
            future[key].append(row["clock"])
    history, samples = History(), []
    boundary = len(requests) // 2
    for row in requests:
        history.observe(row["keys"], 64)
        clock = history.clock
        if clock != row["clock"] or [history.features(key) for key in row["keys"]] != row["features"]:
            raise ValueError("Online features differ from chronological reconstruction")
        split = "train" if clock < boundary - horizon else (
            "validation" if boundary <= clock < len(requests) - horizon else None)
        if split is None:
            continue
        for key in history.entries:
            times = future[key]
            next_index = bisect.bisect_right(times, clock)
            label = int(next_index < len(times) and times[next_index] <= clock + horizon)
            samples.append(dict(clock=clock, key=key, features=history.features(key), label=label,
                                next_gap=times[next_index] - clock if label else None, split=split))
    partitions = [[row for row in samples if row["split"] == name] for name in ("train", "validation")]
    arrays = [(np.asarray([row["features"] for row in part], dtype=np.float32),
               np.asarray([row["label"] for row in part])) for part in partitions]
    if any(set(labels) != {0, 1} for _, labels in arrays):
        raise ValueError("Both partitions require positive and negative samples")
    model = GradientBoostingClassifier(n_estimators=32, max_depth=3, min_samples_leaf=8,
                                       learning_rate=0.1, random_state=seed)
    model.fit(*arrays[0])
    probability = model.predict_proba(arrays[1][0])[:, 1]
    labels = arrays[1][1]
    threshold = max((.2, .35, .5, .65, .8), key=lambda value: (f1_score(labels, probability >= value), value))
    predicted = probability >= threshold
    positives = [[row for row in part if row["label"]] for part in partitions]
    delay = GradientBoostingRegressor(n_estimators=32, max_depth=3, min_samples_leaf=8,
                                     learning_rate=0.1, random_state=seed)
    delay.fit(np.asarray([row["features"] for row in positives[0]], dtype=np.float32),
              np.asarray([row["next_gap"] for row in positives[0]], dtype=float))
    predicted_gap = np.clip(delay.predict(np.asarray([row["features"] for row in positives[1]], dtype=np.float32)), 1, horizon)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sample_bytes = "".join(json.dumps(row) + "\n" for row in samples).encode()
    (output / "samples.jsonl").write_bytes(sample_bytes)
    joblib.dump(model, output / "model.joblib")
    joblib.dump(delay, output / "delay.joblib")
    metadata = dict(schema=2, features=FEATURES, identity=initial[0]["identity"],
                    threshold=threshold, horizon=horizon, seed=seed,
                    split=dict(requests=len(requests), boundary=boundary,
                               train_last_clock=boundary - horizon - 1,
                               validation_first_clock=boundary,
                               last_labeled_clock=len(requests) - horizon - 1),
                    metrics=dict(train_n=len(partitions[0]), validation_n=len(partitions[1]),
                                 f1=f1_score(labels, predicted), precision=precision_score(labels, predicted, zero_division=0),
                                 recall=recall_score(labels, predicted), auc=roc_auc_score(labels, probability),
                                 delay_validation_n=len(positives[1]),
                                 delay_mae_requests=mean_absolute_error([row["next_gap"] for row in positives[1]], predicted_gap)),
                    delay_parameters=delay.get_params(),
                    delay_sha256=hashlib.sha256((output / "delay.joblib").read_bytes()).hexdigest(),
                    parameters=model.get_params(), source_audit_sha256=hashlib.sha256(raw).hexdigest(),
                    samples_sha256=hashlib.sha256(sample_bytes).hexdigest(),
                    model_sha256=hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest())
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--requests", type=int, help="仅使用前 N 条真实请求，排除后续成本校准序列")
    args = parser.parse_args()
    print(json.dumps(train(args.audit, args.output, args.horizon, args.seed, args.requests)))
