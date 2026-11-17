"""通过正式服务入口校准三级路径、训练复用模型，并执行同容量放置对照。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_raa/05_kv_ssd_restore"))

from run_ssd import make_plan, monitor_io, stop_service
from memmgr.short_term.placement import PATHS, fit_costs
from memmgr.short_term.train import train


def save(path, value):
    """将 value 以可读 UTF-8 JSON 保存到 path。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path):
    """返回 path 文件的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counters(raw):
    """解析 raw 原始指标，保留总体计数和 DMA 方向计数。"""
    result = {}
    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            if sample.name.endswith(("_total", "_sum", "_count")):
                result[sample.name] = result.get(sample.name, 0) + sample.value
                if "transfer_type" in sample.labels:
                    name = sample.name + ":" + sample.labels["transfer_type"]
                    result[name] = result.get(name, 0) + sample.value
    return result


def request(client, tokens, identity, output_tokens):
    """经 client 对真实 tokens 生成 output_tokens 个 token，以 identity 标识请求。"""
    payload = dict(model="Qwen/Qwen2.5-1.5B-Instruct", prompt=tokens, request_id=identity,
                   max_tokens=output_tokens, temperature=0, seed=53, ignore_eos=True, logprobs=1,
                   return_tokens_as_token_ids=True, stream=True, stream_options={"include_usage": True})
    started = time.perf_counter()
    answer, generated, probabilities = "", [], []
    first, usage, finish, done = None, None, None, False
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
                if choice["text"] and first is None:
                    first = time.perf_counter() - started
                answer += choice["text"]
                finish = choice.get("finish_reason") or finish
                if choice.get("logprobs"):
                    generated.extend(choice["logprobs"]["tokens"])
                    probabilities.extend(choice["logprobs"]["token_logprobs"])
    if (not done or first is None or finish != "length" or usage is None
            or usage["prompt_tokens"] != len(tokens) or usage["completion_tokens"] != output_tokens
            or len(generated) != output_tokens or len(probabilities) != output_tokens):
        raise RuntimeError("Incomplete real completion")
    return dict(answer=answer, generated_tokens=generated, token_logprobs=probabilities,
                usage=usage, ttft_seconds=first, total_seconds=time.perf_counter() - started)


def run_service(args, name, plan, policy):
    """按 args/name 启动正式 policy 服务，回放 plan 并保留全部真实响应与指标。"""
    output = args.output / name
    output.mkdir()
    cache = args.output / "cache" / name
    server = json.loads((ROOT / "config/server.json").read_text())
    server.update(port=args.port, gpu_memory_utilization=.65)
    server["kv_cache"] = dict(tier="ssd", policy=policy, batch_invariant=True,
                              gpu_bytes=58 * 2**20, cpu_bytes=128 * 2**20, ssd_bytes=224 * 2**20,
                              ssd_root=str(cache), audit_path=str(output / "audit.jsonl"),
                              model_dir=str(args.output / "model") if policy != "lru" else "",
                              placement_profile=str(args.output / "costs.json") if policy == "placement" else "")
    save(output / "server.json", server)
    command = [sys.executable, "-m", "memmgr.inference.server", "--server", str(output / "server.json"),
               "--models", str(ROOT / "config/models.json")]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stopping = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        sampler = None
        try:
            deadline = time.monotonic() + 600
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=180, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited: {output / 'server.log'}")
                    with socket.socket() as probe:
                        listening = probe.connect_ex(("127.0.0.1", args.port)) == 0
                    if listening:
                        client.get("/health").raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("vLLM startup exceeded 600 seconds")
                    time.sleep(.5)
                sampler = pool.submit(monitor_io, process.pid, stopping, output / "resources.jsonl")
                # 使用独立问题预热真实固定数值路径，预热也保留原始响应。
                warm_source = json.loads(args.prompts.read_text())[0]
                warmup = warm_source["prompt_token_ids"][:512]
                save(output / "warmup.json", dict(query_id=warm_source["query_id"],
                     prompt_sha256=hashlib.sha256(json.dumps(warmup).encode()).hexdigest(),
                     **request(client, warmup, "warmup", args.output_tokens)))
                time.sleep(1.5)
                raw = client.get("/metrics").raise_for_status().text
                before = counters(raw)
                (output / "initial.prom").write_text(raw)
                with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
                    for index, row in enumerate(plan):
                        if sampler.done():
                            sampler.result()
                            raise RuntimeError("Resource sampler stopped")
                        result = request(client, row["prompt_token_ids"], f"placement-{index}", args.output_tokens)
                        deadline = time.monotonic() + 30
                        while True:
                            raw = client.get("/metrics").raise_for_status().text
                            after = counters(raw)
                            completed = after["vllm:request_success_total"] - before["vllm:request_success_total"]
                            if completed == 1:
                                time.sleep(.6)
                                raw = client.get("/metrics").raise_for_status().text
                                after = counters(raw)
                                break
                            if completed > 1 or time.monotonic() > deadline:
                                raise RuntimeError("Exclusive metrics boundary failed")
                            time.sleep(.1)
                        delta = {key: after[key] - before.get(key, 0) for key in after}
                        if (delta["vllm:request_success_total"] != 1
                                or delta["vllm:prefix_cache_queries_total"] != len(row["prompt_token_ids"])
                                or delta["vllm:num_preemptions_total"] != 0 or any(value < 0 for value in delta.values())):
                            raise RuntimeError("Invalid exclusive counters")
                        prompt_hash = hashlib.sha256(json.dumps(row["prompt_token_ids"]).encode()).hexdigest()
                        events = [json.loads(line) for line in (output / "audit.jsonl").read_text().splitlines()]
                        begins = [event for event in events if event["kind"] == "begin"]
                        if len(begins) != index + 2 or begins[-1]["prompt_sha256"] != prompt_hash:
                            raise ValueError("Engine input or request identity differs")
                        record = dict(index=index, phase=row["phase"], query_id=row["query_id"],
                                      request_id=begins[-1]["request_id"], prompt_sha256=prompt_hash,
                                      metrics=delta, **result)
                        (output / f"{index:04}.prom").write_text(raw)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        before = after
                        if index % 8 == 0 or index == len(plan) - 1:
                            print(json.dumps(dict(run=name, completed=index + 1, total=len(plan),
                                                  ttft_ms=result["ttft_seconds"] * 1000)), flush=True)
                (output / "final.prom").write_text(raw)
        finally:
            stopping.set()
            try:
                if sampler is not None:
                    sampler.result(timeout=15)
            finally:
                pool.shutdown(wait=True)
                stop_service(process)


def calibration_plan(pool):
    """从 pool 的真实问题生成不同长度、CPU 复访及 SSD 复访的校准请求。"""
    targets = [next(row for row in pool if len(row["prompt_token_ids"]) == length) for length in (512, 1024, 1536)]
    plan = []
    for repeat in range(3):
        for target in targets:
            fillers = sorted((row for row in pool if row["query_id"] != target["query_id"]),
                             key=lambda row: len(row["prompt_token_ids"]), reverse=True)
            by_query = list({row["query_id"]: row for row in reversed(fillers)}.values())
            by_query.sort(key=lambda row: len(row["prompt_token_ids"]), reverse=True)
            sequence = [target, target, *by_query[:2], target, *by_query[2:6], target]
            for row in sequence:
                plan.append(dict(**row, phase="calibrate", cycle=repeat, index=len(plan)))
    return plan


def calibrate(directory, output):
    """从 directory 的真实 prefill、DMA 与 SSD 作业拟合成本，保存到 output。"""
    events = [json.loads(line) for line in (directory / "audit.jsonl").read_text().splitlines()]
    contexts = [row["context"] for row in events if row["kind"] == "placement_context"]
    if len(contexts) != 1:
        raise ValueError("Require one complete calibration context")
    observations = {name: [] for name in PATHS}
    block_bytes = contexts[0]["block_bytes"]
    for line in (directory / "requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        values = row["metrics"]
        computed = values["vllm:request_prefill_kv_computed_tokens_sum"]
        if computed and values.get("vllm:external_prefix_cache_hits_total", 0) == 0:
            observations["compute"].append(dict(jobs=1, blocks=computed / 64,
                                                seconds=values["vllm:request_prefill_time_seconds_sum"]))
        for name, direction in (("cpu_read", "CPU_to_GPU"), ("cpu_write", "GPU_to_CPU")):
            size = values.get("vllm:kv_offload_total_bytes_total:" + direction, 0)
            if size:
                observations[name].append(dict(jobs=values["vllm:kv_offload_size_count:" + direction],
                                               blocks=size / block_bytes,
                                               seconds=values["vllm:kv_offload_total_time_total:" + direction]))
    reserved = {row["job"]: row["keys"] for row in events if row["kind"] == "fs_reserve"}
    for row in events:
        if row["kind"] == "fs_complete":
            if not row["success"]:
                raise ValueError("Failed real SSD transfer")
            name = "ssd_read" if row["direction"] == "load" else "ssd_write"
            size = len(row["keys"]) if row["direction"] == "load" else len(reserved[row["job"]])
            if size:
                observations[name].append(dict(jobs=1, blocks=size, seconds=row["transfer_seconds"]))
    records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    sources = {name: digest(directory / name) for name in ("audit.jsonl", "requests.jsonl")}
    profile = fit_costs(observations, contexts[0], max(row["usage"]["prompt_tokens"] // 64 for row in records), sources)
    save(output, profile)
    return profile


def main():
    """检查真实环境，冻结训练与评价输入，执行同容量三级实验并复核。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--stage", choices=("all", "collect", "evaluate", "smoke"), default="all")
    args = parser.parse_args()
    for name in ("model", "runtime", "prompts", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output.is_relative_to(ROOT / ".agent") or args.model.name != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise ValueError("Require project .agent output and the fixed model revision")
    source = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "memmgr").rglob("*.py")}
    source.update({str(path.relative_to(ROOT)): digest(path) for folder in ("05_kv_ssd_restore", "06_kv_placement")
                   for path in (ROOT / "experiment_raa" / folder).glob("*.py")})
    if args.stage == "evaluate":
        environment = json.loads((args.output / "environment.json").read_text())
        if environment["source"] != source or environment["prompts_sha256"] != digest(args.prompts):
            raise ValueError("Frozen implementation or input changed after collection")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "environment.json", dict(source=source, arguments=vars(args),
                                                     prompts_sha256=digest(args.prompts), model_config_sha256=digest(args.model / "config.json")))
    os.environ.update(HF_HOME=str(args.runtime / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"), TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"), PYTHONHASHSEED="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    from memmgr.inference.models import local_snapshot

    if local_snapshot("Qwen/Qwen2.5-1.5B-Instruct", args.model.name) != args.model:
        raise ValueError("Model cache differs from the fixed provided snapshot")
    pool = json.loads(args.prompts.read_text())
    identities = list(dict.fromkeys(row["query_id"] for row in pool))
    training = set(identities[:len(identities) // 2])
    collect_pool = [row for row in pool if row["query_id"] in training]
    evaluate_pool = [row for row in pool if row["query_id"] not in training]
    collect = make_plan(collect_pool, 6, 15053) + calibration_plan(collect_pool)
    if args.stage == "smoke":
        save(args.output / "collect-trace.json", collect[:20])
        run_service(args, "collect", collect[:20], "lru")
        return
    if args.stage != "evaluate":
        save(args.output / "collect-trace.json", collect)
        run_service(args, "collect", collect, "lru")
        calibrate(args.output / "collect", args.output / "costs.json")
        train(args.output / "collect/audit.jsonl", args.output / "model", horizon=24,
              request_limit=1 + len(collect) - len(calibration_plan(collect_pool)))
    if args.stage == "collect":
        return
    frozen = {str(path.relative_to(args.output)): digest(path) for path in [args.output / "costs.json", * (args.output / "model").glob("*")]}
    for epoch, modes in enumerate((("lru", "learned", "placement"), ("placement", "learned", "lru"))):
        plan = make_plan(evaluate_pool, 2, 65053 + epoch)
        save(args.output / f"evaluate-trace-{epoch}.json", plan)
        for mode in modes:
            run_service(args, f"e{epoch}-{mode}", plan, mode)
    if any(digest(args.output / name) != value for name, value in frozen.items()):
        raise ValueError("Frozen costs or GBDT changed during evaluation")
    if any(digest(ROOT / name) != value for name, value in source.items()):
        raise ValueError("Source changed during evaluation")
    from review_placement import review

    review(args.output)


if __name__ == "__main__":
    main()
