"""在独占 GPU 上独立校准回迁成本，并回放三种准入策略的正式实验。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
from transformers import AutoTokenizer


def save(path, value):
    """将 value 保存到 path，保留可读 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path):
    """计算 path 文件的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(tokenizer, repeats, seed):
    """用 tokenizer、repeats 和 seed 构造独立前缀，覆盖完全淘汰、GPU 部分命中与 CPU 部分覆盖。"""
    rng = random.Random(seed)
    corpus = tokenizer.encode("The memory system stores keys and values for attention. " * 4096,
                              add_special_tokens=False)
    cases = [(length, length, 2) for length in (64, 128, 256, 1024, 3072)]
    cases += [(256, 256, 0), (3072, 3072, 0), (1024, 1536, 2)]
    plan, unique = [], set()
    for repeat in range(repeats):
        rng.shuffle(cases)
        for stored, length, gap in cases:
            prompts = []
            for size in (length, *([3072] * gap)):
                prefix = tokenizer.encode(rng.randbytes(32).hex() + "\n", add_special_tokens=False)
                tokens = (prefix + corpus)[:size]
                first = tuple(tokens[:64])
                if len(tokens) != size or first in unique:
                    raise ValueError("Workload length or first-block identity failed")
                unique.add(first)
                prompts.append(tokens)
            case = f"r{repeat}-s{stored}-l{length}-g{gap}"
            sequence = [("cold", prompts[0][:stored])]
            sequence += [("filler", tokens) for tokens in prompts[1:]]
            sequence += [("revisit", prompts[0])]
            for phase, tokens in sequence:
                plan.append(dict(case=case, repeat=repeat, stored_tokens=stored, length=length,
                                 gap=gap, phase=phase, prompt_token_ids=tokens))
    return plan


def request(client, tokens, output_tokens, seed, request_id):
    """经 client 提交 tokens，使用 output_tokens、seed、request_id，采集真实流响应与逐 token logprobs。"""
    started = time.perf_counter()
    first, usage, finish = None, None, None
    answer, generated, probabilities = "", [], []
    done = False
    payload = dict(model="cost-model", prompt=tokens, max_tokens=output_tokens, temperature=0,
                   seed=seed, ignore_eos=True, stream=True, request_id=request_id,
                   logprobs=1, return_tokens_as_token_ids=True, stream_options={"include_usage": True})
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
                content = choice["text"]
                if content and first is None:
                    first = time.perf_counter() - started
                answer += content
                finish = choice.get("finish_reason") or finish
                if choice.get("logprobs"):
                    generated.extend(choice["logprobs"]["tokens"])
                    probabilities.extend(choice["logprobs"]["token_logprobs"])
    if (not done or first is None or finish != "length" or usage is None
            or usage["prompt_tokens"] != len(tokens) or usage["completion_tokens"] != output_tokens
            or len(generated) != output_tokens or len(probabilities) != output_tokens):
        raise RuntimeError("Incomplete response, token usage or logprobs")
    return dict(ttft_seconds=first, total_seconds=time.perf_counter() - started,
                usage=usage, finish_reason=finish, answer=answer,
                generated_tokens=generated, token_logprobs=probabilities)


def run_service(args, stage, epoch, mode, plan, context):
    """按 args 与 context 启动 stage/epoch/mode 独占服务，回放 plan 并保存原始指标和准入日志。"""
    from tier_baseline import metrics, monitor, settled

    output = args.output / f"{stage}-e{epoch}-{mode}"
    output.mkdir()
    extra = dict(cpu_bytes_to_use=512 * 2**20, block_size=64, eviction_policy="lru",
                 cost_mode=mode, cost_context=context, cost_decisions=str(output / "decisions.jsonl"),
                 cost_profile=str(args.output / "calibration.json"), cost_margin_ms=args.margin_ms)
    transfer = dict(kv_connector="CostOffloadingConnector", kv_role="kv_both",
                    kv_connector_module_path="cost_connector", kv_connector_extra_config=extra)
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model),
            "--served-model-name", "cost-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16", "--max-model-len", "3584", "--max-num-seqs", "1",
            "--block-size", "64", "--num-gpu-blocks-override", "64", "--kv-cache-memory-bytes",
            str(112 * 2**20), "--gpu-memory-utilization", "0.65", "--enforce-eager",
            "--enable-prefix-caching", "--generation-config", "vllm", "--no-enable-log-requests",
            "--kv-transfer-config", json.dumps(transfer)]
    save(output / "server.json", dict(argv=argv, context=context))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        sampler = pool.submit(monitor, process.pid, stop, output / "resources.jsonl")
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120, trust_env=False) as client:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited: {output / 'server.log'}")
                    # TCP 监听就绪后再检查 HTTP，避免把应用错误当作启动等待。
                    with socket.socket() as probe:
                        listening = probe.connect_ex(("127.0.0.1", args.port)) == 0
                    if listening:
                        response = client.get("/health", timeout=10)
                        response.raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("vLLM startup exceeded 600 seconds")
                    time.sleep(0.5)
                # 真实预热 prefill、decode、CPU store 和 load；与正式序列首块分离。
                warmups = [[3400] + [16] * 3071, [3401] + [16] * 3071,
                           [3402] + [16] * 3071, [3400] + [16] * 3071]
                save(output / "warmup.json", [request(client, tokens, args.output_tokens, args.seed,
                     f"warm-{index}") for index, tokens in enumerate(warmups)])
                time.sleep(1.5)
                raw, before = metrics(client)
                required = {"vllm:request_success_total", "vllm:prefix_cache_hits_total",
                            "vllm:request_prefill_kv_computed_tokens_sum", "vllm:time_to_first_token_seconds_sum",
                            "vllm:kv_offload_store_bytes_total", "vllm:num_preemptions_total"}
                if required - before.keys():
                    raise RuntimeError(f"Missing metrics: {required - before.keys()}")
                (output / "initial.prom").write_text(raw, encoding="utf-8")
                with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
                    for index, row in enumerate(plan):
                        if sampler.done():
                            sampler.result()
                            raise RuntimeError("Resource sampler stopped unexpectedly")
                        request_id = f"cost-{index}"
                        tokens = row["prompt_token_ids"]
                        result = request(client, tokens, args.output_tokens, args.seed, request_id)
                        raw, after, delta = settled(client, before, len(tokens))
                        (output / f"{index:04}.prom").write_text(raw, encoding="utf-8")
                        record = dict(index=index, request_id=f"cmpl-{request_id}-0",
                                      **{key: value for key, value in row.items() if key != "prompt_token_ids"},
                                      prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                      metrics=delta, **result)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        before = after
                        if row["phase"] == "revisit":
                            print(json.dumps(dict(stage=stage, epoch=epoch, mode=mode, case=row["case"],
                                  ttft_ms=result["ttft_seconds"] * 1000,
                                  load_bytes=delta.get("vllm:kv_offload_load_bytes_total", 0))), flush=True)
                (output / "final.prom").write_text(raw, encoding="utf-8")
        finally:
            stop.set()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            sampler.result()
            pool.shutdown()


def main():
    """读取实验参数，先独立校准并冻结成本，再用新前缀评价三种准入策略。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiment_raa/02_kv_split_among_gpu_cpu"))
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--margin-ms", type=float, default=0.5)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if (min(args.repeats, args.calibration_repeats) < 2 or not 1 <= args.output_tokens <= 128
            or not 1 <= args.port <= 65535 or not 0 <= args.margin_ms < float("inf")):
        raise ValueError("Invalid experiment arguments")
    args.model, args.runtime, args.output = [path.resolve() for path in (args.model, args.runtime, args.output)]
    if importlib.metadata.version("vllm") != "0.29.0":
        raise ValueError("Require vLLM 0.29.0")
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    here = Path(__file__).resolve().parent
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                      VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC",
                      HF_HOME=str(args.runtime / "hf"), VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"),
                      TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"),
                      PYTHONPATH=str(here) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    args.output.mkdir(parents=True, exist_ok=False)
    import vllm
    native = Path(vllm.__file__).parent
    context = dict(model_revision=args.model.name, model_config_sha256=digest(args.model / "config.json"),
                   gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
                   packages={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "httpx", "prometheus-client", "psutil")},
                   native_sha256={name: digest(native / name) for name in (
                       "distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
                       "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
                       "v1/core/sched/scheduler.py")},
                   block_tokens=64, gpu_blocks=64, cpu_bytes=512 * 2**20, kv_bytes_per_token=28672,
                   dtype="float16", layout="LBHNC", output_tokens=args.output_tokens, seed=args.seed,
                   max_model_len=3584, max_num_seqs=1)
    config = json.loads((args.model / "config.json").read_text())
    if (config["num_hidden_layers"] * 2 * config["num_key_value_heads"]
            * (config["hidden_size"] // config["num_attention_heads"]) * 2 != 28672):
        raise ValueError("Model KV size differs from this experiment")
    source_paths = [*here.glob("*.py"), args.baseline_dir.resolve() / "tier_baseline.py"]
    sources = {str(path): digest(path) for path in source_paths}
    save(args.output / "environment.json", dict(arguments=vars(args), context=context,
         python=sys.version, source_sha256=sources,
         gpu_details=subprocess.check_output(["nvidia-smi", "-q"], text=True)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    from review_cost import calibrate, review

    for epoch, modes in enumerate((("always", "never"), ("never", "always"))):
        plan = make_plan(tokenizer, args.calibration_repeats, args.seed + epoch)
        save(args.output / f"calibrate-trace-{epoch}.json", plan)
        for mode in modes:
            run_service(args, "calibrate", epoch, mode, plan, context)
    calibrate(args.output)
    profile_hash = digest(args.output / "calibration.json")
    for epoch, modes in enumerate((("always", "never", "gate"), ("gate", "never", "always"))):
        plan = make_plan(tokenizer, args.repeats, args.seed + 10000 + epoch)
        save(args.output / f"evaluate-trace-{epoch}.json", plan)
        for mode in modes:
            run_service(args, "evaluate", epoch, mode, plan, context)
            if digest(args.output / "calibration.json") != profile_hash:
                raise RuntimeError("Frozen calibration profile changed during evaluation")
    if any(digest(path) != expected for path, expected in sources.items()):
        raise RuntimeError("Experiment source changed during measurement")
    review(args.output)


if __name__ == "__main__":
    main()
