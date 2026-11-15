"""对独占的本机 vLLM 服务测量冷前缀、重复前缀与新前缀请求。"""

import argparse
import hashlib
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer


def snapshot(client):
    """从 client 连接的服务读取原始指标及前缀查询、命中 token 累计值。"""
    response = client.get("/metrics")
    response.raise_for_status()
    counters = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"):
                counters[sample.name] = counters.get(sample.name, 0) + sample.value
    if len(counters) != 2:
        raise RuntimeError("Required vLLM prefix cache counters are absent")
    return response.text, counters


def request(client, model, tokens, output_tokens, seed):
    """通过 client 流式提交 model 的 tokens，固定 output_tokens 和 seed，返回真实计时与用量。"""
    started = time.perf_counter()
    first = None
    text = ""
    usage = None
    finish = None
    done = False
    payload = dict(model=model, prompt=tokens, max_tokens=output_tokens,
                   temperature=0, seed=seed, ignore_eos=True, stream=True,
                   stream_options={"include_usage": True})
    with client.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("text", "")
                if content:
                    if first is None:
                        first = time.perf_counter() - started
                    text += content
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - started
    if not done or first is None or finish != "length" or usage is None:
        raise RuntimeError("Incomplete streaming response")
    if usage["prompt_tokens"] != len(tokens) or usage["completion_tokens"] != output_tokens:
        raise RuntimeError("Server token usage does not match the fixed workload")
    return dict(ttft_seconds=first, total_seconds=elapsed, usage=usage,
                finish_reason=finish, answer=text)


def run(args):
    """按 args 指定的模型、独占服务和输出目录执行重复实验，保存原始证据与描述统计。"""
    if args.repeats < 1 or args.input_tokens < 128 or args.output_tokens < 1:
        raise ValueError("Invalid experiment size")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        models = client.get("/v1/models")
        models.raise_for_status()
        if args.model not in {item["id"] for item in models.json()["data"]}:
            raise ValueError("Requested model is not served")
        (args.output / "models.json").write_text(models.text, encoding="utf-8")
        corpus = tokenizer.encode("The memory system stores keys and values for attention. " * args.input_tokens)
        warmup_tokens = (tokenizer.encode("Warmup " + uuid.uuid4().hex + "\n") + corpus)[:args.input_tokens]
        warmup = request(client, args.model, warmup_tokens, args.output_tokens, args.seed)
        (args.output / "warmup.json").write_text(json.dumps(warmup, indent=2), encoding="utf-8")
        # 等待引擎将预热统计发布，保持后续快照边界清楚。
        time.sleep(6)
        for repeat in range(args.repeats):
            nonce = uuid.uuid4().hex
            prompts = []
            for label in ("Alpha", "Beta"):
                prefix = tokenizer.encode(label + " " + nonce + "\n")
                tokens = (prefix + corpus)[:args.input_tokens]
                if len(tokens) != args.input_tokens:
                    raise ValueError("Input construction failed")
                prompts.append(tokens)
            if prompts[0][:16] == prompts[1][:16]:
                raise ValueError("Control prefixes must differ in the first cache block")
            for condition, tokens in zip(("cold", "reused", "fresh"), (prompts[0], prompts[0], prompts[1])):
                key = f"{repeat:02d}-{condition}"
                before_raw, before = snapshot(client)
                result = request(client, args.model, tokens, args.output_tokens, args.seed)
                deadline = time.monotonic() + 20
                while True:
                    after_raw, after = snapshot(client)
                    delta = {name: after[name] - before[name] for name in before}
                    if delta["vllm:prefix_cache_queries_total"] > 0:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Prefix metrics were not published within 20 seconds")
                    time.sleep(0.25)
                queries = delta["vllm:prefix_cache_queries_total"]
                hits = delta["vllm:prefix_cache_hits_total"]
                if queries != len(tokens) or not 0 <= hits <= queries:
                    raise RuntimeError(f"Unexpected metric delta; require an idle exclusive server: {delta}")
                if condition == "reused" and hits <= 0:
                    raise RuntimeError("Repeated prefix has no observed cache hits")
                if condition in ("cold", "fresh") and hits != 0:
                    raise RuntimeError("Unique control prefix unexpectedly hit cached blocks")
                (args.output / f"{key}.before.prom").write_text(before_raw, encoding="utf-8")
                (args.output / f"{key}.after.prom").write_text(after_raw, encoding="utf-8")
                row = dict(repeat=repeat, condition=condition, prompt_token_ids=tokens,
                           prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                           prefix_queries=queries, prefix_hits=hits, hit_ratio=hits / queries, **result)
                rows.append(row)
                with (args.output / "requests.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({k: row[k] for k in ("repeat", "condition", "ttft_seconds", "prefix_hits")}), flush=True)
        summary = {}
        for condition in ("cold", "reused", "fresh"):
            group = [row for row in rows if row["condition"] == condition]
            summary[condition] = dict(n=len(group),
                median_ttft_seconds=statistics.median(row["ttft_seconds"] for row in group),
                median_total_seconds=statistics.median(row["total_seconds"] for row in group),
                prefix_hits=sum(row["prefix_hits"] for row in group),
                prefix_queries=sum(row["prefix_queries"] for row in group))
        for repeat in range(args.repeats):
            pair = [row for row in rows if row["repeat"] == repeat and row["condition"] != "fresh"]
            if pair[0]["answer"] != pair[1]["answer"]:
                raise RuntimeError("Cold/reused deterministic responses differ; inspect requests.jsonl")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    run(parser.parse_args())
