"""在共享 GPU 锁内逐组回放实验 07/08 的冻结请求，比较 SSD 写入时机。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
capacity = import_module("experiment_raa.08_kv_capacity.run_capacity")
ssd, joint = capacity.ssd, capacity.joint
gpu_lock = import_module("experiment_raa.12_ssd_write.gpu_lock")
WORKLOADS = ("frozen-78", "pressure")
MODES = ("eager", "on_evict")
GROUPS = [(workload, epoch, mode) for workload in WORKLOADS
          for epoch, modes in enumerate((MODES, MODES[::-1])) for mode in modes]
CPU_MIB = 14


def source_digests():
    """记录本次运行所用 connector、启动器与冻结计划生成器的 SHA-256。"""
    files = [*ROOT.glob("memmgr/short_term/*.py"), Path(__file__), Path(gpu_lock.__file__), Path(capacity.__file__),
             Path(ssd.__file__), Path(joint.__file__)]
    return {str(path.relative_to(ROOT)): joint.digest(path) for path in sorted(files)}


def freeze(args):
    """校验 args.source 的 07 计划、模型和 08 热点计划；为 args.output 建立清单。"""
    reference = ssd.load(args.source / "environment.json")
    pressure_ref = ssd.load(args.pressure_source / "environment.json")
    if any(ssd.load(source / "complete.json")["status"] != "complete"
           for source in (args.source, args.pressure_source)):
        raise ValueError("07/08 参考运行未完成")
    args.output.mkdir(parents=True)
    (plan, _), artifacts = ssd.freeze(args.source, args.output)
    if artifacts != reference["artifacts_sha256"]:
        raise ValueError("07 冻结工件摘要不符")
    pressure_path = "plans/pressure.json"
    joint.save(args.output / pressure_path, capacity.pressure_plan(plan))
    artifacts[pressure_path] = joint.digest(args.output / pressure_path)
    if (artifacts != pressure_ref["artifacts_sha256"]
            or artifacts[pressure_path] != joint.digest(args.pressure_source / pressure_path)
            or pressure_ref["cpu_bytes"] != CPU_MIB * 2**20):
        raise ValueError("08 热点计划、模型或 CPU 容量不符")
    filesystem = subprocess.check_output(["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "-T", args.output], text=True).strip()
    gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    if gpu != reference["gpu"] or filesystem != reference["filesystem"]:
        raise ValueError("GPU 或实际盘层文件系统与 07 不符")
    fixed = {key: reference["arguments"][key] for key in ("seed", "env", "server")}
    return dict(**fixed, packages=reference["packages"], groups=GROUPS, artifacts_sha256=artifacts,
                source_sha256=source_digests(), model_config_sha256=reference["model_config_sha256"],
                reference_sha256={str(source): joint.digest(source / "environment.json")
                                  for source in (args.source, args.pressure_source)},
                filesystem=filesystem, gpu=gpu, kv_bytes_per_block=reference["kv_bytes_per_block"],
                gpu_bytes=joint.GPU_BYTES, cpu_bytes=CPU_MIB * 2**20, ssd_bytes=ssd.SSD_BYTES)


def capture(args, directory, mode, plan, warmup, lock_fd):
    """启动 directory 的独立服务，用 mode 回放 warmup 和 plan；lock_fd 随服务持有。"""
    warm = directory / "warmup"
    warm.mkdir()
    argv, cache = capacity.launch_args(args, directory, "ssd", "learned")
    transfer = json.loads(argv[-1])
    transfer["kv_connector_extra_config"]["policy_config"]["ssd_write"] = mode
    argv[-1] = json.dumps(ssd.transfer_config(dict(args.server, max_model_len=joint.MAX_MODEL_LEN, max_num_seqs=1,
        kv_cache=transfer["kv_connector_extra_config"]["policy_config"]), ROOT))
    argv[-2:-2] = ["--shutdown-timeout", "20"]
    joint.save(directory / "server.json", dict(argv=argv, env=args.env))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    log, audit_path = directory / "server.log", directory / "audit.jsonl"
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=os.environ | args.env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
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
                        ssd.drain(client, directory, warm, True)
                        ssd.snapshot(parent, io_stream, "formal_start")
                        formal.set()
                        joint.capture(client, plan, directory, args.seed, sampler, "joint", audit)
                        ssd.drain(client, directory, directory, True)
                finally:
                    stop.set()
                sampler.result()
                ssd.snapshot(parent, io_stream, "formal_end")
                joint.save(directory / "disk.json", ssd.disk_usage(cache, audit_path))
                client.get("/health").raise_for_status()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
    if process.returncode not in (0, -signal.SIGTERM):
        raise RuntimeError(f"服务关闭失败 ({process.returncode}): {log}")
    with audit_path.open(encoding="utf-8") as audit:
        terminal = None
        for line in audit:
            terminal = json.loads(line)
    if terminal is None or terminal["kind"] != "ssd_shutdown" or terminal["success"] is not True:
        raise RuntimeError(f"缺少 SSD 排空与关闭证据: {audit_path}")


def main():
    """从项目根目录运行一次；调用者用 flock 包住本次执行的单个服务组。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".agent/kv-joint-ssd/output"))
    parser.add_argument("--pressure-source", type=Path, default=Path(".agent/kv-capacity/pressure-14"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    lock_fd = gpu_lock.inherited_lock(ROOT / ".agent/kv-experiment-gpu.lock")
    for name in ("source", "pressure_source", "output"):
        setattr(args, name, ssd.local_path(getattr(args, name), ROOT))
    env_path = args.output / "environment.json"
    if not args.output.exists():
        joint.save(env_path, freeze(args))
    env = ssd.load(env_path)
    if (env["groups"] != [list(group) for group in GROUPS] or env["source_sha256"] != source_digests()
            or any(metadata.version(name) != version for name, version in env["packages"].items())
            or any(joint.digest(args.output / name) != digest for name, digest in env["artifacts_sha256"].items())):
        raise ValueError("组间源码、版本、顺序或冻结工件改变")
    args.seed, args.env, args.server = (env[key] for key in ("seed", "env", "server"))
    args.model = joint.local_snapshot(args.server["model"], args.server["revision"])
    args.cpu_mib = CPU_MIB
    if joint.digest(args.model / "config.json") != env["model_config_sha256"]:
        raise ValueError("Qwen 模型规格摘要不符")
    for workload, epoch, mode in GROUPS:
        directory = args.output / workload / f"epoch-{epoch}" / mode
        if (directory / "complete.json").exists():
            continue
        ssd.require_idle_gpu()
        directory.mkdir(parents=True)
        warmup = ssd.load(args.output / "plans/warmup.json")
        plan = ssd.load(args.output / ("plans/eval.json" if workload == "frozen-78" else "plans/pressure.json"))
        print(f"开始 {workload}/{epoch}/{mode}", flush=True)
        capture(args, directory, mode, plan, warmup, lock_fd)
        ssd.mark_complete(directory, workload=workload, epoch=epoch, mode=mode,
                          requests=len(plan), warmup_requests=len(warmup))
        print(f"完成 {workload}/{epoch}/{mode}", flush=True)
        if (workload, epoch, mode) == GROUPS[-1]:
            ssd.mark_complete(args.output, groups=GROUPS)
        return
    raise ValueError("所有组已经完成")


if __name__ == "__main__":
    main()
