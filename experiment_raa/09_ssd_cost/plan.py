"""核对实验 07 回迁证据，并生成实验 09 的配对输入。"""

from collections import defaultdict
import hashlib
import json
import random

from transformers import AutoTokenizer

BLOCK = 64
SHAPES = ((0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 8), (2, 2, 2), (2, 2, 4))


def load(path):
    """读取 path 中的 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def events(path):
    """逐行读取 path 的 JSONL 记录。"""
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def token_hash(tokens):
    """按 07 的格式计算 tokens 的摘要。"""
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def anchors(root):
    """在 root 的两轮 07 SSD-LRU 中选取证实完成 SSD→CPU→GPU 的同一批输入。"""
    review = load(root / "review.json")
    if review["status"] != "complete":
        raise ValueError("Complete experiment 07 review is required")
    plan = load(root / "plans/eval.json")
    found = []
    for epoch in ("epoch-0", "epoch-1"):
        group = review["epochs"][epoch]["groups"]["ssd-lru"]
        if (group["ssd_recovery"], group["gpu_capacity_bytes"], group["cpu_capacity_bytes"]) != (
            "confirmed", 32 * 2**20, 56 * 2**20
        ):
            raise ValueError("Experiment 07 SSD-LRU geometry differs")
        directory = root / epoch / "ssd-lru"
        by_request = defaultdict(list)
        written, evicted = set(), set()
        for e in events(directory / "audit.jsonl"):
            req = e.get("request_id")
            if e["kind"] == "fs_complete" and e["direction"] == "store" and e["success"]:
                written.update(e["keys"])
            if e["kind"] in {"store", "promotion"}:
                evicted.update(written.intersection(e["evicted"]))
            if e["kind"] == "fs_complete" and e["direction"] == "load" and e["success"]:
                if not set(e["keys"]) <= evicted:
                    raise ValueError("07 SSD read lacks an earlier persisted and evicted block")
            by_request[req].append(e)
        hits = {}
        for row in events(directory / "requests.jsonl"):
            recorded = plan[row["index"]]
            begin, = (e for e in by_request[row["request_id"]] if e["kind"] == "begin")
            if (row["case"], row["phase"], row["prompt_sha256"]) != (
                recorded["case"], recorded["phase"], recorded["prompt_sha256"]
            ) or begin["prompt_sha256"] != row["prompt_sha256"]:
                raise ValueError("07 client or engine digest differs from the frozen plan")
            if row["phase"] != "revisit":
                continue
            history = by_request[row["request_id"]]
            fs = {k for e in history if e["kind"] == "fs_complete" and e["direction"] == "load"
                  and e["success"] for k in e["keys"]}
            gpu = {k for e in history if e["kind"] == "loaded" for k in e["keys"]}
            blocks = len(fs & gpu)
            if blocks:
                lookup, = (e for e in history if e["kind"] == "lookup")
                if (lookup["gpu"], lookup["external"]) != (0, blocks * BLOCK):
                    raise ValueError("07 anchor unexpectedly has non-SSD hits")
                hits[row["index"]] = blocks
        found.append(hits)
    if found[0] != found[1] or set(found[0].values()) != {1, 8}:
        raise ValueError("07 paired SSD anchors differ")
    result = []
    for index, blocks in found[0].items():
        rows = [row for row in plan if row["case"] == plan[index]["case"]]
        if rows[-1]["index"] != index or any(token_hash(row["prompt_token_ids"]) != row["prompt_sha256"] for row in rows):
            raise ValueError("07 plan differs from measured input")
        result.append(dict(id=f"07-{index:04}", source_index=index, shape=[0, 0, blocks],
                           inputs=[row["prompt_token_ids"] for row in rows]))
    return result


def make_plan(args, frozen):
    """从 frozen 的模型生成 args 阶段的独立首块，并保留校准的 07 原样输入。"""
    original = anchors(args.source)
    warmup, = load(args.source / "plans/warmup.json")
    if token_hash(warmup["prompt_token_ids"]) != warmup["prompt_sha256"]:
        raise ValueError("07 warmup digest differs")
    if args.stage == "smoke":
        shapes, repeats = (SHAPES[-2],), 1
    elif args.stage == "calibration":
        shapes, repeats = tuple(shape for shape in SHAPES if shape not in {(0, 0, 1), (0, 0, 8)}), 3
    else:
        shapes, repeats = SHAPES + ((0, 0, 6),), 3
    tokenizer = AutoTokenizer.from_pretrained(frozen["arguments"]["model"], local_files_only=True)
    rng = random.Random({"smoke": 907, "calibration": 909, "heldout": 919}[args.stage])
    corpus = tokenizer.encode("Attention layers reuse the keys and values of an earlier sequence. " * 125,
                              add_special_tokens=False)

    def prompt(length):
        """生成含当前阶段独立首块、且长度恰为 length 的输入。"""
        nonce = tokenizer.encode(f"09 {args.stage} {rng.randbytes(32).hex()}\n", add_special_tokens=False)
        return (nonce + corpus)[:length]

    cases = list(original) if args.stage == "calibration" else []
    for repeat in range(repeats):
        for gpu, cpu, ssd in shapes:
            blocks = gpu + cpu + ssd
            target = prompt((blocks + 1) * BLOCK + 1)
            prime = target[:blocks * BLOCK + 1]
            inputs = [prime] * 4 + [prompt(1025), prompt(1025)]
            if cpu:
                inputs.extend([target[:(gpu + cpu) * BLOCK + 1], prompt(1025), target[:gpu * BLOCK + 1]])
            inputs.append(target)
            cases.append(dict(id=f"{args.stage}-{repeat}-{gpu}-{cpu}-{ssd}", shape=[gpu, cpu, ssd], inputs=inputs))
    rng.shuffle(cases)
    prefixes = {tuple(case["inputs"][-1][:BLOCK]) for case in original}
    seen = set()
    for case in cases:
        first = tuple(case["inputs"][-1][:BLOCK])
        if first in seen or ("source_index" not in case and first in prefixes):
            raise ValueError("Unexpected shared first block")
        seen.add(first)
    return dict(stage=args.stage, warmup=warmup["prompt_token_ids"], cases=cases)


