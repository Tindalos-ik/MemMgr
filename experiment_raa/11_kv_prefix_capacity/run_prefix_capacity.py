"""以实验 08 冻结输入单独采集 14 MiB CPU-LRU 或 SSD-LRU。"""

import argparse
from importlib import import_module, metadata
from pathlib import Path
from shutil import copyfile
import subprocess
import sys


capacity = import_module("experiment_raa.08_kv_capacity.run_capacity")
review08 = import_module("experiment_raa.08_kv_capacity.review_capacity")
ssd, joint, ROOT = capacity.ssd, capacity.joint, capacity.ROOT
GROUPS = ("cpu-lru", "ssd-lru")
INPUTS = ("plans/eval.json", "plans/warmup.json")


def source_digests(reference):
    """记录 reference 涉及的采集器及当前正式短期 KV 模块的字节摘要。"""
    paths = set(reference["source_sha256"])
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "memmgr/short_term").glob("*.py"))
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    return {name: review08.digest(ROOT / name) for name in sorted(paths)}


def frozen_plans(root, hashes):
    """从 root 校验 hashes，读取 78 条正式请求与 1,025-token 预热。"""
    for name in INPUTS:
        if review08.digest(root / name) != hashes[name]:
            raise ValueError(f"冻结输入摘要不符: {root / name}")
    plan, warmup = (ssd.load(root / name) for name in INPUTS)
    if len(plan) != 78 or len(warmup) != 1 or len(warmup[0]["prompt_token_ids"]) != 1025:
        raise ValueError("实验 08 的正式请求数或预热长度不符")
    return plan, warmup


def prepare(args):
    """按 args.historical 的实验 08 冻结环境创建全新输出目录。"""
    reference_path = args.historical / "environment.json"
    reference = ssd.load(reference_path)
    if reference["workload"] != "frozen-78" or reference["cpu_bytes"] != 14 * capacity.MIB:
        raise ValueError("历史输入须为实验 08 的 14 MiB 冻结组")
    hashes = {name: reference["artifacts_sha256"][name] for name in INPUTS}
    frozen_plans(args.historical, hashes)
    args.output.mkdir(parents=True)
    (args.output / "plans").mkdir()
    for name in INPUTS:
        copyfile(args.historical / name, args.output / name)
    frozen_plans(args.output, hashes)
    environment = dict(reference, artifacts_sha256=hashes, historical=str(args.historical),
                       historical_environment_sha256=review08.digest(reference_path),
                       historical_review_sha256=review08.digest(args.historical / "review.json"),
                       source_sha256=source_digests(reference), groups=GROUPS)
    joint.save(args.output / "environment.json", environment)


def collect(args):
    """在调用方持有共享 GPU 锁期间，用 args 从启动到停止采集一个独立组。"""
    environment = ssd.load(args.output / "environment.json")
    if any(metadata.version(name) != version for name, version in environment["packages"].items()):
        raise ValueError("当前软件版本与实验 08 不一致")
    if source_digests(environment) != environment["source_sha256"]:
        raise ValueError("正式源码或采集器在组间发生改变")
    filesystem = subprocess.check_output(
        ["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "-T", args.output], text=True).strip()
    gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    if filesystem != environment["filesystem"] or gpu != environment["gpu"]:
        raise ValueError("盘层文件系统或 GPU 身份与实验 08 不一致")
    for name in ("seed", "env", "server"):
        setattr(args, name, environment["arguments"][name])
    args.model = joint.local_snapshot(args.server["model"], args.server["revision"])
    if review08.digest(args.model / "config.json") != environment["model_config_sha256"]:
        raise ValueError("模型配置摘要与实验 08 不一致")
    args.cpu_mib = 14
    plan, warmup = frozen_plans(args.output, environment["artifacts_sha256"])
    capacity.collect_group(args, 0, args.group, plan, warmup)
    if not (args.output / "epoch-0" / args.group / "complete.json").is_file():
        raise RuntimeError(f"组未完成，请查阅原始日志: {args.output / 'epoch-0' / args.group}")


def main():
    """读取冻结输入路径，按 --group 决定准备数据或运行独立服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", default=".agent/kv-capacity/frozen-14")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--port", type=int, default=18111)
    args = parser.parse_args()
    args.output = ssd.local_path(args.output, ROOT)
    args.historical = ssd.local_path(args.historical, ROOT)
    if args.group:
        collect(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
