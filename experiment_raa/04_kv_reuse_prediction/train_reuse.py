"""从真实访问记录生成时间隔离样本，训练并冻结小型 GBDT 与准入阈值。"""

import bisect
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from reuse_features import FEATURES, History, digest, read_rows, save


def dataset(audit, horizon):
    """由 audit 的真实 begin 事件重建特征；horizon 为未来标签观察的请求数。"""
    events = [row for row in read_rows(audit) if row["kind"] == "begin"]
    future = defaultdict(list)
    for row in events:
        future[row["root"]].append(row["clock"])
    boundary = len(events) // 2
    history, samples = History(), []
    for row in events:
        vector = history.observe(row["root"], row["features"][-1])
        clock = history.clock
        if clock != row["clock"] or vector != row["features"]:
            raise ValueError("Online features differ from chronological reconstruction")
        if clock < boundary - horizon:
            split = "train"
        elif boundary <= clock < len(events) - horizon:
            split = "validation"
        else:
            continue
        for root in history.entries:
            times = future[root]
            following = bisect.bisect_right(times, clock)
            label = int(following < len(times) and times[following] <= clock + horizon)
            samples.append(dict(clock=clock, root=root, features=history.features(root), label=label, split=split))
    return samples, dict(events=len(events), boundary=boundary, horizon=horizon,
                         train_last_clock=boundary - horizon - 1,
                         validation_first_clock=boundary, last_labeled_clock=len(events) - horizon - 1)


def fit(samples, seed):
    """使用 samples 的 train 分区和 seed 训练固定规模 GBDT，仅在 validation 分区选择阈值。"""
    train = [row for row in samples if row["split"] == "train"]
    validation = [row for row in samples if row["split"] == "validation"]
    xtrain = np.asarray([row["features"] for row in train], dtype=np.float32)
    ytrain = np.asarray([row["label"] for row in train])
    xvalid = np.asarray([row["features"] for row in validation], dtype=np.float32)
    yvalid = np.asarray([row["label"] for row in validation])
    if set(ytrain) != {0, 1} or set(yvalid) != {0, 1}:
        raise ValueError("Training and validation both require positive and negative observations")
    model = GradientBoostingClassifier(n_estimators=32, max_depth=3, min_samples_leaf=8,
                                        learning_rate=0.1, random_state=seed)
    model.fit(xtrain, ytrain)
    probability = model.predict_proba(xvalid)[:, 1]
    thresholds = (0.2, 0.35, 0.5, 0.65, 0.8)
    threshold = max(thresholds, key=lambda value: (f1_score(yvalid, probability >= value), value))
    predicted = probability >= threshold
    metrics = dict(train_n=len(train), validation_n=len(validation),
                   train_positive=int(ytrain.sum()), validation_positive=int(yvalid.sum()),
                   validation_f1=f1_score(yvalid, predicted),
                   validation_precision=precision_score(yvalid, predicted, zero_division=0),
                   validation_recall=recall_score(yvalid, predicted),
                   validation_auc=roc_auc_score(yvalid, probability),
                   validation_brier=brier_score_loss(yvalid, probability),
                   threshold_candidates=list(thresholds))
    return model, threshold, metrics


def train(root, horizon, seed):
    """从 root/collect 的真实审计训练模型，按 horizon 和 seed 固定样本与参数。"""
    root = Path(root)
    samples, split = dataset(root / "collect" / "audit.jsonl", horizon)
    model, threshold, metrics = fit(samples, seed)
    path = root / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
    joblib.dump(model, root / "model.joblib")
    save(root / "model.json", dict(schema=1, features=FEATURES, seed=seed, split=split,
         threshold=threshold, metrics=metrics, model_parameters=model.get_params(),
         model_sha256=digest(root / "model.joblib"), samples_sha256=digest(path),
         source_audit_sha256=digest(root / "collect" / "audit.jsonl")))
