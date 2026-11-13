"""采集真实访问、训练 GBDT，并在独立序列上比较三种 CPU KV 保留策略。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
from transformers import AutoTokenizer

from reuse_features import digest, save
from reuse_workload import make_plan


def run_service(args, name, mode, plan, context):
    """依据 args/context 启动 name/mode 独占服务，执行 plan 并保存真实响应与原始指标。"""
    from run_cost import request
    from tier_baseline import metrics, monitor, settled

    output = args.output / name
    output.mkdir()
    extra = dict(cpu_bytes_to_use=64 * 2**20, block_size=64,
                 spec_name="ReuseSpec", spec_module_path="reuse_connector",
                 reuse_mode=mode, reuse_model=str(args.output), reuse_audit=str(output / "audit.jsonl"))
    transfer = dict(kv_connector="ReuseConnector", kv_role="kv_both",
                    kv_connector_module_path="reuse_connector", kv_connector_extra_config=extra)
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model),
            "--served-model-name", "cost-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16", "--max-model-len", "3584", "--max-num-seqs", "1",
            "--block-size", "64", "--num-gpu-blocks-override", "64", "--kv-cache-memory-bytes",
            str(112 * 2**20), "--gpu-memory-utilization", "0.65", "--enforce-eager",
            "--enable-prefix-caching", "--generation-config", "vllm", "--no-enable-log-requests",
            "--kv-transfer-config", json.dumps(transfer)]
    save(output / "server.json", dict(argv=argv, context=context, mode=mode))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("x", encoding="utf-8") as log:
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
                    with socket.socket() as probe:
                        listening = probe.connect_ex(("127.0.0.1", args.port)) == 0
                    if listening:
                        client.get("/health", timeout=10).raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("vLLM startup exceeded 600 seconds")
                    time.sleep(0.5)
                warmups = [[3400 + index] + [16] * (length - 1)
                           for index, length in enumerate((512, 1024, 1536, 1024))]
                save(output / "warmup.json", [request(client, tokens, args.output_tokens, args.seed,
                     f"warm-{index}") for index, tokens in enumerate(warmups)])
                time.sleep(1.5)
                raw, before = metrics(client)
                (output / "initial.prom").write_text(raw, encoding="utf-8")
                required = {"vllm:request_success_total", "vllm:prefix_cache_hits_total",
                            "vllm:request_prefill_kv_computed_tokens_sum",
                            "vllm:time_to_first_token_seconds_sum", "vllm:num_preemptions_total"}
                if required - before.keys():
                    raise RuntimeError(f"Missing metrics: {required - before.keys()}")
                with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
                    for index, row in enumerate(plan):
                        if sampler.done():
                            sampler.result()
                            raise RuntimeError("Resource sampler stopped unexpectedly")
                        tokens = row["prompt_token_ids"]
                        result = request(client, tokens, args.output_tokens, args.seed, f"reuse-{index}")
                        raw, after, delta = settled(client, before, len(tokens))
                        (output / f"{index:04}.prom").write_text(raw, encoding="utf-8")
                        record = dict(**{key: value for key, value in row.items() if key != "prompt_token_ids"},
                                      prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                      request_id=f"cmpl-reuse-{index}-0", metrics=delta, **result)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        before = after
                        if index % 10 == 0 or index == len(plan) - 1:
                            print(json.dumps(dict(run=name, completed=index + 1, total=len(plan),
                                  phase=row["phase"], ttft_ms=result["ttft_seconds"] * 1000)), flush=True)
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
    """解析模型、环境和实验大小，先训练冻结模型，再以正反顺序执行独立评价。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiment_raa/02_kv_split_among_gpu_cpu"))
    parser.add_argument("--cost-dir", type=Path, default=Path("experiment_raa/03_kv_restore_cost"))
    parser.add_argument("--collect-rounds", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    if (args.collect_rounds < 4 or args.rounds < 2 or not 1 <= args.horizon <= 24
            or not 1 <= args.output_tokens <= 128 or not 1 <= args.port <= 65535):
        raise ValueError("Invalid experiment parameters")
    for name in ("model", "runtime", "output", "baseline_dir", "cost_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if importlib.metadata.version("vllm") != "0.29.0":
        raise ValueError("Require vLLM 0.29.0")
    for path in (args.baseline_dir, args.cost_dir):
        sys.path.insert(0, str(path))
    here = Path(__file__).resolve().parent
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0",
                      VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC",
                      HF_HOME=str(args.runtime / "hf"), VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"),
                      TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"),
                      PYTHONPATH=str(here) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    import vllm
    native = Path(vllm.__file__).parent
    native_names = ("distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
                    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
                    "v1/kv_offload/base.py", "v1/kv_offload/cpu/spec.py",
                    "v1/kv_offload/cpu/manager.py", "v1/kv_offload/cpu/policies/lru.py")
    context = dict(model_revision=args.model.name, model_config_sha256=digest(args.model / "config.json"),
                   packages={name: importlib.metadata.version(name) for name in
                             ("vllm", "torch", "transformers", "httpx", "psutil", "scikit-learn", "numpy", "joblib")},
                   native_sha256={name: digest(native / name) for name in native_names},
                   gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
                   gpu_blocks=64, cpu_bytes=64 * 2**20, block_tokens=64,
                   kv_bytes_per_token=28672, dtype="float16", layout="LBHNC",
                   store_policy="request_level", restore_policy="always",
                   output_tokens=args.output_tokens, seed=args.seed, max_model_len=3584, max_num_seqs=1)
    config = json.loads((args.model / "config.json").read_text())
    if (args.model.name != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
            or config["num_hidden_layers"] * 2 * config["num_key_value_heads"]
            * (config["hidden_size"] // config["num_attention_heads"]) * 2 != 28672):
        raise ValueError("Require the fixed Qwen2.5-1.5B-Instruct FP16 model")
    sources = {str(path): digest(path) for path in [*here.glob("*.py"),
               args.baseline_dir / "tier_baseline.py", args.cost_dir / "run_cost.py"]}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "environment.json", dict(arguments=vars(args), context=context,
         python=sys.version, source_sha256=sources,
         gpu_details=subprocess.check_output(["nvidia-smi", "-q"], text=True)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    from train_reuse import train
    from review_reuse import review

    plan = make_plan(tokenizer, args.collect_rounds, args.seed)
    save(args.output / "collect-trace.json", plan)
    run_service(args, "collect", "lru", plan, context)
    train(args.output, args.horizon, args.seed)
    frozen = {name: digest(args.output / name) for name in ("model.json", "model.joblib", "samples.jsonl")}
    for epoch, modes in enumerate((("lru", "admission", "learned"), ("learned", "admission", "lru"))):
        plan = make_plan(tokenizer, args.rounds, args.seed + 10000 + epoch)
        save(args.output / f"evaluate-trace-{epoch}.json", plan)
        for mode in modes:
            run_service(args, f"e{epoch}-{mode}", mode, plan, context)
            if any(digest(args.output / name) != expected for name, expected in frozen.items()):
                raise RuntimeError("Frozen model or training data changed during evaluation")
    if any(digest(path) != expected for path, expected in sources.items()):
        raise RuntimeError("Experiment source changed during measurement")
    review(args.output)


if __name__ == "__main__":
    main()
