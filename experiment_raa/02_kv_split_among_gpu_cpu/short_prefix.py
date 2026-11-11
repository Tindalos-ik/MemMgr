"""补测 64 / 128-token 前缀，在相同 6144-token 干扰量下定位回迁成本转折。"""

import argparse
import hashlib
import os
from pathlib import Path
import random
import sys

from transformers import AutoTokenizer

from tier_baseline import experiment, save


def make_plan(tokenizer, seed):
    """用 tokenizer 和 seed 生成短前缀及两条 3072-token 干扰请求，按用例随机排序。"""
    rng = random.Random(seed)
    corpus = tokenizer.encode("The memory system stores keys and values for attention. " * 4096,
                              add_special_tokens=False)
    cases = [(length, repeat) for length in (64, 128) for repeat in range(5)]
    rng.shuffle(cases)
    plan = []
    for length, repeat in cases:
        prompts = []
        for size in (length, 3072, 3072):
            prefix = tokenizer.encode(rng.randbytes(32).hex() + "\n", add_special_tokens=False)
            tokens = (prefix + corpus)[:size]
            if len(tokens) != size:
                raise ValueError("Invalid short-prefix prompt")
            prompts.append(tokens)
        for phase, tokens in zip(("cold", "filler", "filler", "revisit"), [*prompts, prompts[0]]):
            plan.append(dict(case=f"r{repeat}-l{length}-g2", repeat=repeat, length=length,
                             gap=2, filler_tokens=3072, phase=phase, prompt_token_ids=tokens))
    first_blocks = [tuple(row["prompt_token_ids"][:64]) for row in plan if row["phase"] != "revisit"]
    if len(first_blocks) != len(set(first_blocks)):
        raise ValueError("Short-prefix prompts share a first block")
    return plan


def main():
    """读取本地模型、缓存与输出参数，使用同一测量器执行正反顺序的短前缀补测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--client-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.port, args.output_tokens, args.seed = 18081, 32, 23
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                      VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC",
                      HF_HOME=str(args.runtime / "hf"), VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"),
                      TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"))
    sys.path.insert(0, str(args.client_dir))
    from prefix_baseline import request
    save(args.output / "arguments.json", dict(arguments=vars(args),
         source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(__file__).with_name("tier_baseline.py"),
                                     args.client_dir / "prefix_baseline.py")}))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for epoch, modes in enumerate((("gpu", "cpu512"), ("cpu512", "gpu"))):
        plan = make_plan(tokenizer, args.seed + epoch)
        save(args.output / f"trace-{epoch}.json", plan)
        for mode in modes:
            experiment(args, mode, epoch, plan, request)


if __name__ == "__main__":
    main()
