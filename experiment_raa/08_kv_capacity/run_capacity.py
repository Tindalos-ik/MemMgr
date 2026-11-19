"""复用实验 07 的真实服务和冻结输入，分目录采集 CPU 容量与压力对照。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module, metadata
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import traceback

import httpx
import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
ssd = import_module("experiment_raa.07_kv_joint_ssd.run_ssd_joint")
joint = ssd.joint
MIB = 2**20
POLICIES = ("lru", "admission", "learned")
GROUPS = tuple(f"{tier}-{policy}" for tier in ("cpu", "ssd") for policy in POLICIES)


def groups_for(cpu_mib, pressure_policies):
    """按 cpu_mib 和压力策略 pressure_policies 选择同轮运行组。"""
    if pressure_policies:
        return tuple(group for group in GROUPS if group.split("-")[1] in pressure_policies)
    return ("cpu-admission", "ssd-admission") if cpu_mib == 56 else GROUPS


def pressure_plan(plan):
    """从冻结 plan 取十个独立 513-token 前缀，集中预热后按同序复访。"""
    targets = {}
    for row in plan:
        tokens = row["prompt_token_ids"]
        if len(tokens) >= 513:
            targets.setdefault(tuple(tokens[:64]), tokens[:513])
    hot = list(targets.values())[:10]
    if len(hot) != 10:
        raise ValueError("压力计划需要十个独立的首块")
    schedule = [(index, "prime", 0) for index in range(len(hot)) for _ in range(4)]
    schedule.extend((index, "revisit", 3 * (len(hot) - 1 - index) + len(hot) - 1)
                    for index in range(len(hot)))
    return [dict(index=i, case=f"pressure-{index}", round=0, phase=phase, gap=gap,
                 prompt_token_ids=hot[index],
                 prompt_sha256=hashlib.sha256(json.dumps(hot[index]).encode()).hexdigest())
            for i, (index, phase, gap) in enumerate(schedule)]


def prepare(args):
    """核对 args.source 的 07 环境、当前软件及模型，冻结本阶段常量和工件。"""
    source = args.source
    reference = ssd.load(source / "environment.json")
    if ssd.load(source / "complete.json")["status"] != "complete":
        raise ValueError(f"07 尚未完成: {source}")
    packages = {name: metadata.version(name) for name in reference["packages"]}
    if packages != reference["packages"] or packages["vllm"] != "0.29.0":
        raise ValueError(f"当前软件版本与 07 不一致: {packages}")
    for name, digest in reference["source_sha256"].items():
        if joint.digest(ROOT / name) != digest:
            raise ValueError(f"当前源码与 07 不一致: {name}")
    frozen = reference["arguments"]
    args.seed, args.env, args.server = (frozen[key] for key in ("seed", "env", "server"))
    args.model = joint.local_snapshot(args.server["model"], args.server["revision"])
    model_digest = joint.digest(args.model / "config.json")
    if model_digest != reference["model_config_sha256"]:
        raise ValueError("Qwen 配置摘要与 07 不一致")
    gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    if gpu != reference["gpu"] or joint.GPU_BYTES != 32 * MIB:
        raise ValueError("GPU 身份或固定 KV 容量不一致")
    args.output.mkdir(parents=True)
    (plan, warmup), artifacts = ssd.freeze(source, args.output)
    if artifacts != reference["artifacts_sha256"]:
        raise ValueError("计划或 GBDT 与 07 的冻结清单不一致")
    filesystem = subprocess.check_output(
        ["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "-T", args.output], text=True).strip()
    if filesystem != reference["filesystem"]:
        raise ValueError(f"盘层文件系统与 07 不一致: {filesystem}")
    workload = "pressure" if args.pressure_policies else "frozen-78"
    if args.pressure_policies:
        plan = pressure_plan(plan)
        path = "plans/pressure.json"
        joint.save(args.output / path, plan)
        artifacts[path] = joint.digest(args.output / path)
    groups = groups_for(args.cpu_mib, args.pressure_policies)
    epochs = (groups, groups[::-1])
    joint.save(args.output / "environment.json", dict(
        arguments=vars(args), command=[sys.executable, *sys.argv], workload=workload, epochs=epochs,
        packages=packages, source_environment_sha256=joint.digest(source / "environment.json"),
        source_sha256={**reference["source_sha256"], str(Path(__file__).relative_to(ROOT)): joint.digest(Path(__file__))},
        artifacts_sha256=artifacts, model_config_sha256=model_digest, gpu=gpu, filesystem=filesystem,
        kv_bytes_per_block=reference["kv_bytes_per_block"], gpu_bytes=joint.GPU_BYTES,
        cpu_bytes=args.cpu_mib * MIB, ssd_bytes=ssd.SSD_BYTES))
    return plan, warmup, epochs


def launch_args(args, directory, tier, policy):
    """从同一 07 启动命令派生 args 的容量、tier、policy 和独立目录 directory。"""
    argv = joint.command(args, directory, "lru", "always")
    options = json.loads(argv[-1])["kv_connector_extra_config"]["policy_config"]
    options.update(tier=tier, policy=policy, cpu_bytes=args.cpu_mib * MIB)
    if policy != "lru":
        options["model_dir"] = str(args.output / "model")
    if tier == "ssd":
        root = directory / "ssd"
        root.mkdir(mode=0o700)
        options.update(ssd_root=str(root), ssd_bytes=ssd.SSD_BYTES)
    server = dict(args.server, max_model_len=joint.MAX_MODEL_LEN, max_num_seqs=1, kv_cache=options)
    transfer = ssd.transfer_config(server, ROOT)
    argv[-1] = json.dumps(transfer)
    extra = transfer["kv_connector_extra_config"]
    cache = Path(extra["secondary_tiers"][0]["root_dir"]) if tier == "ssd" else None
    return argv, cache


def capture_group(args, directory, group, plan, warmup):
    """在独立 directory 运行 group，把 warmup 和 plan 的审计、Prom 及 I/O 原样保存。"""
    tier, policy = group.split("-")
    warm = directory / "warmup"
    warm.mkdir()
    argv, cache = launch_args(args, directory, tier, policy)
    joint.save(directory / "server.json", dict(argv=argv, env=args.env, policy=policy, restore="always",
                                               tier=tier, ssd_root=cache, gpu_bytes=joint.GPU_BYTES,
                                               cpu_bytes=args.cpu_mib * MIB))
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
                ssd.snapshot(parent, io_stream, "warmup_start")
                sampler = pool.submit(ssd.sample_io, parent, io_stream, formal, stop)
                try:
                    with audit_path.open(encoding="utf-8") as audit:
                        joint.capture(client, warmup, warm, args.seed, sampler, "warm", audit)
                        ssd.drain(client, directory, warm, cache is not None)
                        ssd.snapshot(parent, io_stream, "formal_start")
                        formal.set()
                        joint.capture(client, plan, directory, args.seed, sampler, "joint", audit)
                        ssd.drain(client, directory, directory, cache is not None)
                finally:
                    stop.set()
                sampler.result()
                ssd.snapshot(parent, io_stream, "formal_end")
                joint.save(directory / "disk.json", ssd.disk_usage(cache, audit_path))
                client.get("/health").raise_for_status()
                if process.poll() is not None or audit_path.with_suffix(".jsonl.1").exists():
                    raise RuntimeError(f"服务提前退出或审计轮转: {directory}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def collect(args, plan, warmup, epochs):
    """按 epochs 启动 args 的独立分组；失败组保留证据，其他组继续原计划。"""
    failures = []
    for epoch, groups in enumerate(epochs):
        for group in groups:
            ssd.require_idle_gpu()
            directory = args.output / f"epoch-{epoch}" / group
            directory.mkdir(parents=True)
            print(f"开始 {epoch}/{group}", flush=True)
            try:
                capture_group(args, directory, group, plan, warmup)
            except (RuntimeError, OSError, ValueError, httpx.HTTPError, psutil.Error) as error:
                failure = dict(epoch=epoch, group=group, error=type(error).__name__, detail=str(error),
                               traceback=traceback.format_exc())
                joint.save(directory / "failure.json", failure)
                failures.append([epoch, group])
                print(f"失败 {epoch}/{group}: {error}", flush=True)
                continue
            ssd.mark_complete(directory, epoch=epoch, group=group, requests=len(plan), warmup_requests=len(warmup))
            print(f"完成 {epoch}/{group}", flush=True)
    if failures:
        joint.save(args.output / "failures.json", dict(status="failed_groups", failures=failures, epochs=epochs))
        return 2
    ssd.mark_complete(args.output, epochs=epochs, requests_per_group=len(plan), warmup_requests_per_group=len(warmup))
    return 0


def main():
    """验证命令行约束并采集一个容量阶段；调用方用共享 Linux flock 包裹此命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".agent/kv-joint-ssd/output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-mib", type=int, choices=(56, 28, 14), required=True)
    parser.add_argument("--port", type=int, default=18088)
    parser.add_argument("--pressure-policies", choices=POLICIES, nargs="+", default=[])
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or len(set(args.pressure_policies)) != len(args.pressure_policies):
        parser.error("端口越界或压力策略重复")
    args.source = ssd.local_path(args.source, ROOT)
    args.output = ssd.local_path(args.output, ROOT)
    return collect(args, *prepare(args))


if __name__ == "__main__":
    raise SystemExit(main())
