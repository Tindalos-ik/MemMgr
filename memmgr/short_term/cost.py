"""读取与当前运行身份匹配的冻结成本表，显式决定 CPU 回迁。"""

import hashlib
import json
import math
from pathlib import Path
import statistics


def geometry(prompt, gpu, external):
    """校验 prompt 总输入、gpu 命中和 external 候选，返回精确查表键。"""
    if any(type(value) is not int for value in (prompt, gpu, external)):
        raise ValueError("Token counts must be integers")
    if not 0 <= gpu < prompt or not 0 <= external <= prompt - gpu:
        raise ValueError("Invalid cache geometry")
    return f"{prompt}:{gpu}:{external}"


class CostPolicy:
    """只对已经校准的几何应用路径成本，未知项直接报错。"""

    def __init__(self, path, context, margin_ms):
        """从 path 读取成本表，核对 context；margin_ms 为要求的最小节省。"""
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        profile = json.loads(raw)
        if profile["schema"] != 2 or profile["context"] != context:
            raise ValueError("Cost profile and runtime context differ")
        if not math.isfinite(margin_ms) or margin_ms < 0:
            raise ValueError("Invalid cost margin")
        self.margin = margin_ms / 1000
        self.entries = profile["entries"]
        if not self.entries:
            raise ValueError("Empty cost profile")
        for key, entry in self.entries.items():
            if key != geometry(entry["prompt"], entry["gpu"], entry["external"]) or entry["n"] < 2:
                raise ValueError("Invalid cost geometry or sample count")
            for field in ("restore_seconds", "recompute_seconds"):
                value = entry[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid calibrated path cost")

    def decide(self, prompt, gpu, external):
        """按 prompt、gpu、external 的精确校准成本决定是否回迁。"""
        entry = self.entries[geometry(prompt, gpu, external)]
        saved = entry["recompute_seconds"] - entry["restore_seconds"]
        return dict(admit=saved > self.margin, expected_saved_seconds=saved,
                    restore_seconds=entry["restore_seconds"], recompute_seconds=entry["recompute_seconds"])


def calibrate(pairs, output):
    """从 pairs 中 always/never 真实运行目录生成成本表，保存至 output。"""
    observations, expected_context, sources = {}, None, {}
    for always, never in pairs:
        runs = []
        for directory in (Path(always), Path(never)):
            payload = {}
            for name in ("audit.jsonl", "requests.jsonl"):
                raw = (directory / name).read_bytes()
                if not raw.endswith(b"\n"):
                    raise ValueError("Incomplete calibration source")
                sources[str(directory / name)] = hashlib.sha256(raw).hexdigest()
                payload[name] = [json.loads(line) for line in raw.splitlines()]
            contexts = [row["context"] for row in payload["audit.jsonl"] if row["kind"] == "context"]
            if len(contexts) != 1:
                raise ValueError("Require one complete calibration session")
            if expected_context is None:
                expected_context = contexts[0]
            if contexts[0] != expected_context:
                raise ValueError("Calibration environments differ")
            lookups = {row["request_id"]: row for row in payload["audit.jsonl"]
                       if row["kind"] == "lookup" and row["external"] > 0}
            runs.append((payload["requests.jsonl"], lookups))
        if len(runs[0][0]) != len(runs[1][0]):
            raise ValueError("Calibration request counts differ")
        for restore, recompute in zip(runs[0][0], runs[1][0]):
            if restore["prompt_sha256"] != recompute["prompt_sha256"]:
                raise ValueError("Calibration inputs differ")
            left = runs[0][1].get(restore["request_id"])
            right = runs[1][1].get(recompute["request_id"])
            if left is None and right is None:
                continue
            if left is None or right is None:
                raise ValueError("Calibration candidate availability differs")
            key = geometry(left["prompt"], left["gpu"], left["external"])
            if key != geometry(right["prompt"], right["gpu"], right["external"]):
                raise ValueError("Calibration candidate geometry differs")
            if left["admitted"] != left["external"] or right["admitted"] != 0:
                raise ValueError("Calibration requires always and never paths")
            observations.setdefault(key, []).append((restore["server_ttft_seconds"], recompute["server_ttft_seconds"]))
    entries = {}
    for key, values in observations.items():
        if len(values) < 2:
            raise ValueError(f"Insufficient observations for geometry {key}")
        prompt, gpu, external = map(int, key.split(":"))
        entries[key] = dict(prompt=prompt, gpu=gpu, external=external, n=len(values),
                            restore_seconds=statistics.median(a for a, _ in values),
                            recompute_seconds=statistics.median(b for _, b in values))
    if not entries:
        raise ValueError("Calibration has no reusable CPU candidates")
    profile = dict(schema=2, context=expected_context, entries=entries, sources=sources)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(profile, indent=2) + "\n")
    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ALWAYS", "NEVER"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.pair, args.output)))
