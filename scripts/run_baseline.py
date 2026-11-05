"""按固定计划执行三档检索与七组端到端初测。

用法：在 WSL 项目根目录运行 `python scripts/run_baseline.py [--config CONFIG] [--output-root DIR]`。
脚本会启动索引、检索和生成任务，并在 `.agent/executions/execution-*` 中保存计划、
配置、命令、日志及完整性审查记录。调用前需准备模型、数据和本机服务；
该入口会实际运行测量任务。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def code_snapshot() -> dict[str, str]:
    """返回应用及执行脚本的 SHA256，供整轮实验前后检查实际冻结源码。"""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            [
                *Path("memmgr").rglob("*.py"),
                *Path("scripts").glob("*.py"),
                *Path("scripts").glob("*.sh"),
            ]
        )
    }


def find_artifact_paths(log_path: Path, output_dir: str) -> list[Path]:
    """按命令日志中的标准输出路径取得该命令生成的运行目录。"""
    artifact_prefix = f"{output_dir}/"
    return [
        Path(line.strip())
        for line in log_path.read_text().splitlines()
        if line.strip().startswith(artifact_prefix)
    ]


def copy_command_log(run_dir: Path, log_path: Path) -> None:
    """将计划命令的完整标准输出副本写入运行目录logs/。"""
    run_logs = run_dir / "logs"
    run_logs.mkdir(exist_ok=True)
    (run_logs / "command.log").write_text(log_path.read_text())


def review_run_artifact(
    execution_dir: Path,
    log_path: Path,
    artifact_prefix: str,
    child_environment: dict[str, str],
    frozen_code: dict[str, str],
    label: str,
    command_args: list[str],
    record: dict,
) -> None:
    """对子命令打印的最后一个运行目录执行对应完整性复核。

    审查器日志保存在执行目录；运行目录同时获得冻结源码摘要，记录其版本边界。
    """
    candidates = [
        line.strip()
        for line in log_path.read_text().splitlines()
        if line.strip().startswith(artifact_prefix)
    ]
    if candidates:
        record["run_directory"] = candidates[-1]
        review_path = execution_dir / f"{label}.integrity.log"
        with review_path.open("w") as review:
            guard_script = (
                "scripts/review_run.py"
                if command_args[0] == "run"
                else "scripts/review_retrieval.py"
            )
            guard = subprocess.run(
                [sys.executable, guard_script, candidates[-1]],
                stdout=review,
                stderr=subprocess.STDOUT,
                env=child_environment,
                pass_fds=(int(child_environment["MEMMGR_RESOURCE_LOCK_FD"]),)
                if "MEMMGR_RESOURCE_LOCK_FD" in child_environment
                else (),
                timeout=60,
            )
        record["integrity_returncode"] = guard.returncode
        Path(candidates[-1], "code-snapshot.json").write_text(
            json.dumps(frozen_code, indent=2) + "\n"
        )
    else:
        record["integrity_returncode"] = 1
        record["integrity_error"] = "Run did not print its artifact directory"


def execute(
    execution_dir: Path,
    frozen_code: dict[str, str],
    child_environment: dict[str, str],
    records: list[dict],
    label: str,
    config: dict,
    command_args: list[str],
    timeout: int = 1200,
) -> bool:
    """执行一条计划命令并保存命令、运行目录和完整性结果。

    参数包含执行目录、源码摘要、子进程环境、累计记录、标签、逐命令配置、
    CLI参数和超时秒数。返回子进程退出码是否为0；完整性失败会中止计划，
    超时会在记录中保存无返回码状态。
    """
    if code_snapshot() != frozen_code:
        raise RuntimeError(
            "Application code changed during formal execution; preserve "
            "completed runs and start a new execution after review"
        )

    config_path = execution_dir / f"{label}.config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if command_args[0] == "index":
        command = [
            sys.executable,
            "scripts/index_measured.py",
            "--config",
            str(config_path),
            *command_args[1:],
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "memmgr",
            "--config",
            str(config_path),
            *command_args,
        ]

    started = time.monotonic()
    record = {
        "label": label,
        "command": command,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log_path = execution_dir / f"{label}.log"
    with log_path.open("w") as log:
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_environment,
                pass_fds=(int(child_environment["MEMMGR_RESOURCE_LOCK_FD"]),)
                if "MEMMGR_RESOURCE_LOCK_FD" in child_environment
                else (),
                timeout=timeout,
            )
            record["returncode"] = completed.returncode
        except subprocess.TimeoutExpired as error:
            record.update(returncode=None, failure=repr(error))

    artifact_prefix = f"{config['experiment']['output_dir']}/"
    artifact_paths = find_artifact_paths(log_path, config["experiment"]["output_dir"])
    if artifact_paths:
        record["run_directory"] = str(artifact_paths[-1])
        copy_command_log(artifact_paths[-1], log_path)

    if command_args[0] in ["run", "retrieval"] and record.get("returncode") == 0:
        review_run_artifact(
            execution_dir,
            log_path,
            artifact_prefix,
            child_environment,
            frozen_code,
            label,
            command_args,
            record,
        )

    record["wall_seconds"] = time.monotonic() - started
    records.append(record)
    (execution_dir / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record), flush=True)

    if record.get("integrity_returncode", 0) != 0:
        raise RuntimeError(
            f"Measurement integrity failed for {label}; preserve this "
            "execution and start a new one after the instrumentation fix"
        )
    return record.get("returncode") == 0


def main() -> None:
    """创建带源码快照的执行目录并按固定顺序运行全部正式条件。

    计划使用30个检索问题、8个问答问题，各重复两次并预热一次；
    服务端调度维持单序列，客户端并发条件为配置中的1、2、4。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--output-root", default=".agent/executions")
    args = parser.parse_args()
    execution_dir = Path(args.output_root) / (
        "execution-" + time.strftime("%Y%m%d-%H%M%S")
    )
    execution_dir.mkdir(parents=True)
    base_config = json.loads(Path(args.config).read_text())
    server_config = base_config.get("_web_snapshot", {}).get("server") or json.loads(
        Path("config/server.json").read_text()
    )
    frozen_code = code_snapshot()
    base_config["_code_snapshot"] = frozen_code
    (execution_dir / "code-snapshot.json").write_text(
        json.dumps(frozen_code, indent=2) + "\n"
    )

    e2e_conditions = [
        {"context_tokens": budget, "top_k": top_k, "concurrency": concurrency}
        for budget, top_k, concurrency in [
            (512, 3, 1),
            (1024, 3, 1),
            (1536, 3, 1),
            (1024, 1, 1),
            (1024, 5, 1),
            (1024, 3, 2),
            (1024, 3, 4),
        ]
    ]
    plan = {
        "queries_retrieval": base_config["dataset"]["query_count"],
        "queries_e2e": min(
            base_config["experiment"]["queries"], base_config["dataset"]["query_count"]
        ),
        "repeats": base_config["experiment"]["repeats"],
        "warmups": base_config["experiment"]["warmups"],
        "server_max_num_seqs": server_config["max_num_seqs"],
        "server_max_model_len": base_config["llm"]["max_model_len"],
        "note": (
            f"Client concurrency2/4 includes queueing; actual configured server max_num_seqs={server_config['max_num_seqs']}. "
            "The seven E2E rows change one dimension relative to1024/k3/c1."
        ),
        "retrieval_tiers": ["small", "medium", "full"],
        "e2e": e2e_conditions,
    }
    (execution_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)

    child_environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
    )
    records = []
    for tier in plan["retrieval_tiers"]:
        if (
            execute(
                execution_dir,
                frozen_code,
                child_environment,
                records,
                f"index-{tier}",
                base_config,
                ["index", "--tier", tier],
            )
            is False
        ):
            raise RuntimeError(f"Index failed: {tier}")
        if not execute(
            execution_dir,
            frozen_code,
            child_environment,
            records,
            f"retrieval-{tier}",
            base_config,
            ["retrieval", "--tier", tier, "--concurrency", "1"],
        ):
            raise RuntimeError(f"Retrieval failed: {tier}")

    for index, condition in enumerate(plan["e2e"], 1):
        config = json.loads(json.dumps(base_config))
        config["experiment"].update(condition)
        if not execute(
            execution_dir,
            frozen_code,
            child_environment,
            records,
            f"e2e-{index}",
            config,
            ["run", "--tier", "small", "--concurrency", str(condition["concurrency"])],
            timeout=900,
        ):
            raise RuntimeError(f"RAG failed: e2e-{index}")

    print(execution_dir, flush=True)


if __name__ == "__main__":
    main()
