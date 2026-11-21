"""以实验 06 冻结输入串行测量 CPU 和 CPU+SSD 的真实 KV 回迁。"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
joint = importlib.import_module("experiment_raa.06_kv_joint.run_joint")

from memmgr.short_term.config import local_path, transfer_config

GROUPS = ("cpu-lru", "cpu-learned", "ssd-lru", "ssd-learned")
EPOCHS = (GROUPS, GROUPS[::-1])
SSD_BYTES = 4 * 2**30


def load(path):
    """读取 path 的 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(source, output):
    """复制 source 中的冻结计划和模型到 output，并校验输入摘要。"""
    (output / "plans").mkdir()
    (output / "model").mkdir()
    paths = ("plans/eval.json", "plans/warmup.json", "model/model.json", "model/model.joblib")
    hashes = {}
    for path in paths:
        shutil.copyfile(source / path, output / path)
        hashes[path] = joint.digest(output / path)
        if hashes[path] != joint.digest(source / path):
            raise RuntimeError(f"Frozen artifact changed during copy: {path}")
    if hashes["model/model.joblib"] != load(output / "model/model.json")["model_sha256"]:
        raise ValueError("Frozen GBDT digest mismatch")
    plans = [load(output / f"plans/{name}.json") for name in ("eval", "warmup")]
    for plan, expected in zip(plans, (78, 1), strict=True):
        if len(plan) != expected:
            raise ValueError(f"Expected {expected} frozen requests, got {len(plan)}")
        for index, row in enumerate(plan):
            token_digest = hashlib.sha256(json.dumps(row["prompt_token_ids"]).encode()).hexdigest()
            if row["index"] != index or row["prompt_sha256"] != token_digest:
                raise ValueError(f"Frozen request index or digest mismatch: {index}")
    return plans, hashes


def launch_args(args, directory, tier, policy):
    """复用 06 的 args 与 policy 启动参数；只为 SSD tier 增加 directory 下的独立盘层。"""
    argv = joint.command(args, directory, policy, "always")
    cache = None
    if tier == "ssd":
        root = directory / "ssd"
        root.mkdir(mode=0o700)
        options = json.loads(argv[-1])["kv_connector_extra_config"]["policy_config"]
        options.update(tier=tier, ssd_root=str(root), ssd_bytes=SSD_BYTES)
        transfer = transfer_config(dict(args.server, max_model_len=joint.MAX_MODEL_LEN,
                                        max_num_seqs=1, kv_cache=options), ROOT)
        cache = Path(transfer["kv_connector_extra_config"]["secondary_tiers"][0]["root_dir"])
        argv[-1] = json.dumps(transfer)
    return argv, cache


def snapshot(parent, stream, phase):
    """把 parent 完整进程树的真实 Linux I/O 快照按 phase 写到 stream。"""
    children = parent.children(recursive=True)
    if not any(process.name().startswith("VLLM::") for process in children):
        raise RuntimeError("No vLLM engine in the active process tree")
    processes = [dict(pid=process.pid, created_at=process.create_time(),
                      io=process.io_counters()._asdict(), rss=process.memory_info().rss)
                 for process in (parent, *children)]
    stream.write(json.dumps(dict(phase=phase, time_ns=time.monotonic_ns(), processes=processes)) + "\n")
    stream.flush()


def sample_io(parent, stream, formal, stop):
    """从 formal 置位起定期采样 parent 到 stream，直到 stop 置位。"""
    while not stop.wait(0.5):
        if formal.is_set():
            snapshot(parent, stream, "formal_sample")


def drain(client, directory, stage, ssd):
    """核对 directory 中 stage 请求已收齐审计与 SSD 完成事件，并检查 client 健康。"""
    wanted = {row["request_id"] for line in (stage / "requests.jsonl").open(encoding="utf-8")
              if (row := json.loads(line))}
    closed = set()
    stores = set()
    writes = Counter()
    jobs = {}
    deadline = time.monotonic() + 30
    with (directory / "audit.jsonl").open(encoding="utf-8") as audit:
        while True:
            for line in audit:
                event = json.loads(line)
                kind = event["kind"]
                req = event.get("request_id")
                if kind == "ssd_request_closed":
                    closed.add(req)
                elif kind in {"store", "stored"}:
                    if kind == "store":
                        stores.add(req)
                    change = 1 if kind == "store" else -1
                    for key in event["keys"]:
                        entry = (req, key)
                        writes[entry] += change
                        if writes[entry] == 0:
                            del writes[entry]
                        elif writes[entry] < 0:
                            raise RuntimeError(f"Unexpected GPU to CPU completion: {entry}")
                elif kind in {"fs_submit", "fs_complete"}:
                    key = (event["session"], event["job"])
                    spec = (req, event["direction"], event["keys"])
                    if kind == "fs_submit":
                        if key in jobs:
                            raise RuntimeError(f"Repeated pending SSD job: {key}")
                        jobs[key] = spec
                    elif jobs.pop(key) != spec or not event["success"]:
                        raise RuntimeError(f"Invalid SSD completion: {key}")
            client.get("/health").raise_for_status()
            if wanted <= (closed if ssd else stores) and not jobs and not writes:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Transfer drain did not finish for {stage}: "
                                   f"{wanted - (closed if ssd else stores)}, {jobs.keys()}, {len(writes)} writes")
            time.sleep(0.1)
    raw, _ = joint.metrics(client)
    (stage / "final.prom").write_text(raw, encoding="utf-8")


def disk_usage(cache, audit):
    """统计 cache 下 .bin 文件的 st_size 合计，并按 audit 核对独立空盘与 O_DIRECT。"""
    if cache is None:
        return dict(root=None, files=0, bytes=0)
    with audit.open(encoding="utf-8") as stream:
        fs = [event for line in stream if (event := json.loads(line))["kind"] == "fs_init"]
    if len(fs) != 1 or fs[0]["direct_io"] is not True or fs[0]["existing_bytes"] != 0:
        raise RuntimeError(f"Expected an empty O_DIRECT tier: {cache}")
    files = size = 0
    for path in cache.rglob("*.bin"):
        files += 1
        size += path.stat().st_size
    return dict(root=str(cache), files=files, bytes=size)


def mark_complete(directory, **fields):
    """在 directory 中原子发布带 fields 的完整标记。"""
    pending = directory / ".complete.tmp"
    joint.save(pending, dict(status="complete", **fields))
    pending.rename(directory / "complete.json")


def require_idle_gpu():
    """拒绝已被其他计算进程占用的 GPU。"""
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    if processes:
        raise RuntimeError(f"GPU already has compute processes:\n{processes}")


def run(args, epoch, group, plan, warmup):
    """在 args.output 对 epoch/group 运行独立服务，依次使用 warmup 和 plan。"""
    require_idle_gpu()
    tier, policy = group.split("-")
    directory = args.output / f"epoch-{epoch}" / group
    warm = directory / "warmup"
    warm.mkdir(parents=True)
    argv, cache = launch_args(args, directory, tier, policy)
    joint.save(directory / "server.json", dict(argv=argv, env=args.env, policy=policy, restore="always",
                                               tier=tier, ssd_root=cache, gpu_bytes=joint.GPU_BYTES,
                                               cpu_bytes=joint.CPU_BYTES))
    print(f"开始 {epoch}/{group}", flush=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    log = directory / "server.log"
    audit_path = directory / "audit.jsonl"
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=os.environ | args.env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with (httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120, trust_env=False) as client,
                  (directory / "resources.jsonl").open("x", encoding="utf-8") as io_stream,
                  ThreadPoolExecutor(max_workers=1) as pool):
                joint.ready(client, process, args.port, log)
                parent = psutil.Process(process.pid)
                formal, stop = threading.Event(), threading.Event()
                snapshot(parent, io_stream, "warmup_start")
                sampler = pool.submit(sample_io, parent, io_stream, formal, stop)
                try:
                    with audit_path.open(encoding="utf-8") as audit:
                        joint.capture(client, warmup, warm, args.seed, sampler, "warm", audit)
                        drain(client, directory, warm, cache is not None)
                        snapshot(parent, io_stream, "formal_start")
                        formal.set()
                        joint.capture(client, plan, directory, args.seed, sampler, "joint", audit)
                        drain(client, directory, directory, cache is not None)
                finally:
                    stop.set()
                sampler.result()
                snapshot(parent, io_stream, "formal_end")
                joint.save(directory / "disk.json", disk_usage(cache, audit_path))
                client.get("/health").raise_for_status()
                if process.poll() is not None or audit_path.with_suffix(".jsonl.1").exists():
                    raise RuntimeError(f"Service exited early or audit rotated: {directory}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    mark_complete(directory, epoch=epoch, group=group, requests=len(plan), warmup_requests=len(warmup))
    print(f"完成 {epoch}/{group}", flush=True)


def main():
    """读取正式 06 产物，在独占 GPU 上按正反顺序采集四组 07 对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".agent/kv-joint/output")
    parser.add_argument("--output", default=".agent/kv-joint-ssd/output")
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    args.source = local_path(args.source, ROOT)
    args.output = local_path(args.output, ROOT)
    frozen = load(args.source / "environment.json")
    packages = {name: importlib.metadata.version(name) for name in frozen["packages"]}
    if packages != frozen["packages"] or packages["vllm"] != "0.29.0":
        raise ValueError(f"Package versions differ from experiment 06: {packages}")
    args.seed = frozen["arguments"]["seed"]
    args.env = frozen["arguments"]["env"]
    args.server = frozen["arguments"]["server"]
    args.model = joint.local_snapshot(args.server["model"], args.server["revision"])
    model_digest = joint.digest(args.model / "config.json")
    if model_digest != frozen["model_config_sha256"]:
        raise ValueError("Frozen LLM model configuration digest mismatch")
    with (ROOT / ".agent/kv-joint-ssd-gpu.lock").open("a") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpu()
        args.output.mkdir(parents=True)
        (plan, warmup), artifacts = freeze(args.source, args.output)
        sources = (Path(__file__), Path(joint.__file__), *sorted((ROOT / "memmgr/short_term").glob("*.py")))
        joint.save(args.output / "environment.json", dict(
            arguments=vars(args), command=[sys.executable, *sys.argv], packages=packages,
            source_environment_sha256=joint.digest(args.source / "environment.json"), artifacts_sha256=artifacts,
            source_sha256={str(path.relative_to(ROOT)): joint.digest(path) for path in sources},
            model_config_sha256=model_digest, kv_bytes_per_block=frozen["kv_bytes_per_block"], epochs=EPOCHS,
            gpu=subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
            filesystem=subprocess.check_output(["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "-T", args.output], text=True).strip(),
        ))
        for epoch, groups in enumerate(EPOCHS):
            for group in groups:
                run(args, epoch, group, plan, warmup)
        mark_complete(args.output, epochs=EPOCHS, requests_per_group=len(plan), warmup_requests_per_group=len(warmup))


if __name__ == "__main__":
    main()
