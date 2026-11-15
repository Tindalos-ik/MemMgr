"""回放真实 SciFact 检索提示词，比较 GPU、CPU 和 SSD 三级 KV 缓存。"""

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
import psutil


def save(path, value):
    """将 value 以可读 UTF-8 JSON 写入 path。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path):
    """返回 path 文件的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompts(config, tier, model):
    """使用 config/tier 的真实索引与 model tokenizer 生成不同长度的完整聊天输入。"""
    from memmgr.inference.prompt import assemble_prompt
    from memmgr.long_term.index import make_embedder
    from memmgr.long_term.retrieval import open_collection
    from memmgr.inference.models import local_snapshot
    from transformers import AutoTokenizer

    collection, manifest = open_collection(config, tier)
    config["embedding"]["snapshot_path"] = str(local_snapshot(config["embedding"]["model"], config["embedding"]["revision"]))
    _, embed = make_embedder(config)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    root = Path(config["dataset"]["root"]) / tier
    queries = [json.loads(line) for line in (root / "queries.jsonl").read_text().splitlines()]
    output = []
    for query in queries:
        result = collection.query(query_embeddings=[embed([query["text"]], query=True)[0].tolist()],
                                  n_results=5, include=["documents", "metadatas", "distances"])
        candidates, seen = [], set()
        for text, meta in zip(result["documents"][0], result["metadatas"][0]):
            if meta["doc_id"] not in seen:
                candidates.append((meta["doc_id"], text))
                seen.add(meta["doc_id"])
        for budget in (512, 1024, 1536):
            messages, ids, _, tokens, truncated = assemble_prompt(tokenizer, query["text"], candidates, budget)
            output.append(dict(identity=f"{query['_id']}-{budget}", query_id=query["_id"],
                               budget=budget, messages=messages, included_doc_ids=ids,
                               truncated=truncated, prompt_token_ids=tokens))
    return output, manifest


def make_plan(pool, repeats, seed):
    """将 pool 中真实输入按 repeats、seed 排列为热点、干扰、切换和长间隔复访。"""
    rng = random.Random(seed)
    identities = list(dict.fromkeys(row["query_id"] for row in pool))
    rng.shuffle(identities)
    if len(identities) < 12:
        raise ValueError("Require at least twelve distinct real questions")
    by_query = {identity: [row for row in pool if row["query_id"] == identity] for identity in identities}
    hot = [max(by_query[identity], key=lambda row: len(row["prompt_token_ids"])) for identity in identities[:4]]
    fillers = [row for identity in identities[4:] for row in by_query[identity]]
    rng.shuffle(fillers)
    plan = []
    for cycle in range(repeats):
        rng.shuffle(fillers)
        groups = [("stable", [hot[0], hot[1], hot[0], hot[1]]),
                  ("scan", fillers[:6] + hot[:2]),
                  ("shift", [hot[2], hot[3], hot[2], hot[3]] + fillers[6:10] + hot[2:]),
                  ("long_gap", fillers[10:18] + hot[:2])]
        for phase, rows in groups:
            for row in rows:
                plan.append(dict(**row, index=len(plan), phase=phase, cycle=cycle))
    return plan


def monitor_io(pid, stop, path):
    """采样 pid 进程树的真实 I/O 与内存，直到 stop 置位，写入 path。"""
    parent = psutil.Process(pid)
    processes = [parent, *[child for child in parent.children(recursive=True)
                          if child.name().startswith("VLLM::")]]
    if len(processes) < 2:
        raise RuntimeError("Cannot identify a healthy vLLM engine process")
    with path.open("x", encoding="utf-8") as stream:
        while not stop.is_set():
            rows = []
            for process in processes:
                rows.append(dict(pid=process.pid, created_at=process.create_time(),
                                 io=process.io_counters()._asdict(), rss=process.memory_info().rss))
            stream.write(json.dumps(dict(time=time.time(), processes=rows)) + "\n")
            stream.flush()
            stop.wait(0.5)


def stop_service(process):
    """停止本次创建的 process 独立进程组并等待退出。"""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_service(args, name, mode, plan):
    """按 args 启动 name/mode 独占服务，回放 plan 并保存实际响应与传输事件。"""
    from run_cost import request
    from tier_baseline import metrics, settled

    output = args.output / name
    output.mkdir()
    extra = dict(cpu_bytes_to_use=64 * 2**20, block_size=64, eviction_policy="lru",
                 spec_name="AuditTieringSpec", spec_module_path="ssd_connector",
                 audit_path=str(output / "audit.jsonl"), secondary_tiers=[])
    cache = args.ssd_root / name
    if mode == "ssd":
        cache.mkdir(parents=True, exist_ok=False)
        extra["secondary_tiers"] = [dict(type="StrictFileSystemTier", module_path="ssd_connector",
                                         root_dir=str(cache), n_read_threads=4, n_write_threads=4)]
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model),
            "--served-model-name", "cost-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16", "--max-model-len", "2048", "--max-num-seqs", "1",
            "--block-size", "64", "--num-gpu-blocks-override", "64", "--kv-cache-memory-bytes",
            str(112 * 2**20), "--gpu-memory-utilization", "0.65", "--enforce-eager",
            "--enable-prefix-caching", "--generation-config", "vllm", "--no-enable-log-requests"]
    if mode != "gpu":
        transfer = dict(kv_connector="AuditConnector", kv_role="kv_both",
                        kv_connector_module_path="ssd_connector", kv_connector_extra_config=extra)
        argv += ["--kv-transfer-config", json.dumps(transfer)]
    save(output / "server.json", dict(argv=argv, mode=mode, ssd_root=cache,
                                      gpu_blocks=64, cpu_bytes=0 if mode == "gpu" else 64 * 2**20))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with (output / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        sampler = None
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=180, trust_env=False) as client:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited: {output / 'server.log'}")
                    with socket.socket() as probe:
                        listening = probe.connect_ex(("127.0.0.1", args.port)) == 0
                    if listening:
                        client.get("/health").raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("vLLM startup exceeded 600 seconds")
                    time.sleep(0.5)
                sampler = pool.submit(monitor_io, process.pid, stop, output / "resources.jsonl")
                save(output / "warmup.json", request(client, [16] * 512, args.output_tokens, args.seed, "warm"))
                time.sleep(1.5)
                raw, before = metrics(client)
                (output / "initial.prom").write_text(raw)
                with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
                    for index, row in enumerate(plan):
                        if sampler.done():
                            sampler.result()
                            raise RuntimeError("I/O sampler stopped")
                        result = request(client, row["prompt_token_ids"], args.output_tokens, args.seed, f"ssd-{index}")
                        raw, after, delta = settled(client, before, len(row["prompt_token_ids"]))
                        (output / f"{index:04}.prom").write_text(raw)
                        record = dict(index=index, identity=row["identity"], phase=row["phase"],
                                      client_request_id=f"ssd-{index}", input_tokens=len(row["prompt_token_ids"]),
                                      prompt_sha256=hashlib.sha256(json.dumps(row["prompt_token_ids"]).encode()).hexdigest(),
                                      metrics=delta, **result)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        before = after
                        if index % 10 == 0 or index == len(plan) - 1:
                            print(json.dumps(dict(run=name, completed=index + 1, total=len(plan))), flush=True)
                time.sleep(1)
                raw, _ = metrics(client)
                (output / "final.prom").write_text(raw)
                save(output / "disk.json", dict(files=sum(1 for _ in cache.rglob("*.bin")),
                     bytes=sum(path.stat().st_size for path in cache.rglob("*.bin"))))
        finally:
            stop.set()
            try:
                if sampler is not None:
                    sampler.result()
            finally:
                try:
                    stop_service(process)
                finally:
                    pool.shutdown()


def main():
    """解析真实模型、索引和 SSD 目录，冻结请求后执行两轮正反顺序比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssd-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/baseline.json"))
    parser.add_argument("--tier", choices=("small", "medium", "full"), default="small")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--seed", type=int, default=14053)
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.output_tokens <= 128 or not 1024 <= args.port <= 65535:
        raise ValueError("Invalid experiment parameters")
    if args.model.name != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise ValueError("Require the fixed Qwen2.5-1.5B-Instruct revision")
    for name in ("model", "runtime", "output", "ssd_root", "config"):
        setattr(args, name, getattr(args, name).resolve())
    root = Path(__file__).resolve().parents[2]
    for directory in (root, root / "experiment_raa/02_kv_split_among_gpu_cpu",
                      root / "experiment_raa/03_kv_restore_cost"):
        sys.path.insert(0, str(directory))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0",
                      VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC",
                      HF_HOME=str(args.runtime / "hf"), VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"),
                      TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"),
                      PYTHONPATH=str(Path(__file__).resolve().parent) + os.pathsep + str(root))
    import vllm
    if vllm.__version__ != "0.29.0":
        raise ValueError("Require vLLM 0.29.0")
    args.output.mkdir(parents=True, exist_ok=False)
    args.ssd_root.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    pool, manifest = prompts(config, args.tier, args.model)
    save(args.output / "prompts.json", pool)
    source = {str(path): digest(path) for path in Path(__file__).parent.glob("*.py")}
    save(args.output / "environment.json", dict(arguments=vars(args), source_sha256=source,
         packages={name: importlib.metadata.version(name) for name in
                   ("vllm", "torch", "transformers", "chromadb-client", "sentence-transformers")},
         model_config_sha256=digest(args.model / "config.json"), index_manifest=manifest,
         gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True),
         filesystem=subprocess.check_output(["findmnt", "-T", str(args.ssd_root), "-J"], text=True)))
    for epoch, modes in enumerate((("gpu", "cpu", "ssd"), ("ssd", "cpu", "gpu"))):
        plan = make_plan(pool, args.repeats, args.seed + epoch)
        save(args.output / f"trace-{epoch}.json", plan)
        for mode in modes:
            run_service(args, f"e{epoch}-{mode}", mode, plan)
    if any(digest(path) != expected for path, expected in source.items()):
        raise RuntimeError("Experiment source changed during measurement")
    from review_ssd import review
    review(args.output)


if __name__ == "__main__":
    main()
