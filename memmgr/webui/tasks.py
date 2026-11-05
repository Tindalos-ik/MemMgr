"""在独立进程组中执行受约束任务，以文件锁保护共享资源并持久化真实退出状态。"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .config import (
    ROOT,
    STATE,
    configs,
    effective_config,
    read_json,
    save_json,
    safe_name,
)


@contextlib.contextmanager
def resource_lock(blocking=False):
    """取得跨后端/任务进程的共享排他锁；blocking 指定是否等待。"""
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "resource.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def task_path(task_id):
    """由 task_id 安全取得本系统拥有的任务目录。"""
    return STATE / "tasks" / safe_name(task_id)


def process_identity(pid):
    """读取 pid 的内核 start_ticks 与 boot_id，避免用可受 WSL 时钟影响的墙钟作为进程身份。"""
    stat = Path(f"/proc/{int(pid)}/stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    return {
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_matches(record):
    """用 record 的 PID、内核启动标识与精确命令检查自有任务，防止 PID 重用误取消。"""
    import psutil

    try:
        process = psutil.Process(record["pid"])
        return (
            process_identity(record["pid"]) == record["process_identity"]
            and "memmgr.webui.tasks" in process.cmdline()
            and record["id"] in process.cmdline()
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        KeyError,
        FileNotFoundError,
        ProcessLookupError,
    ):
        return False


def owned_group_members(record):
    """依据 record 的独立 PGID 与命令配置路径证明归属，返回仍存活的自有进程。"""
    import psutil

    try:
        if process_identity(record["pid"]) != record.get("process_identity"):
            return []
    except FileNotFoundError:
        pass
    members = []
    for process in psutil.process_iter(["pid", "cmdline", "status"]):
        try:
            if (
                os.getpgid(process.pid) == record["pid"]
                and process.info["status"] != psutil.STATUS_ZOMBIE
            ):
                members.append(process)
        except (ProcessLookupError, psutil.Error):
            continue
    proof = any(
        str(task_path(record["id"])) in " ".join(p.info.get("cmdline") or [])
        or (
            "memmgr.webui.tasks" in (p.info.get("cmdline") or [])
            and record["id"] in (p.info.get("cmdline") or [])
        )
        for p in members
    )
    if members and not proof:
        raise RuntimeError("Cannot prove remaining process group ownership")
    return members


def finish_group(record, include_leader=False):
    """等待 record 自有子组退出，超时后只对核验身份的存活成员 SIGKILL。"""
    import psutil

    members = [
        p for p in owned_group_members(record) if include_leader or p.pid != os.getpid()
    ]
    _, alive = psutil.wait_procs(members, timeout=5)
    for process in alive:
        if process.is_running() and os.getpgid(process.pid) == record["pid"]:
            process.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    alive = [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    if alive:
        raise RuntimeError(
            f"Owned processes have not stopped: {[p.pid for p in alive]}"
        )


def read_task(task_id):
    """读取 task_id 的持久状态，异常退出且已消失的执行器精确标为 interrupted。"""
    path = task_path(task_id) / "task.json"
    record = read_json(path)
    if (
        record["status"] in ("queued", "running")
        and not process_matches(record)
        and time.time() - record["created_at"] > 3
    ):
        record = read_json(path)
        if record["status"] in ("queued", "running"):
            members = owned_group_members(record)
            if members:
                os.killpg(record["pid"], signal.SIGTERM)
                finish_group(record, include_leader=True)
            cancellation = task_path(task_id) / "cancel.json"
            record.update(
                status="cancelled" if cancellation.exists() else "interrupted",
                finished_at=time.time(),
                error="Cancellation completed after owned group exit"
                if cancellation.exists()
                else "Owned task process exited without a terminal state",
            )
            save_json(path, record)
    log = task_path(task_id) / "task.log"
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            record["progress"] = {
                "source": "task.log",
                "bytes": log.stat().st_size,
                "tail": stream.read().decode("utf-8", errors="replace"),
            }
    record["elapsed_seconds"] = (record.get("finished_at") or time.time()) - (
        record.get("started_at") or record["created_at"]
    )
    counts = {"formal": 0, "warmup": 0, "ok": 0, "failed": 0, "usage_missing": 0}
    from .records import complete_rows, incomplete_tail
    import hashlib

    run_ids = []
    has_incomplete_tail = False
    for request_file in task_path(task_id).glob("runs/*/requests.jsonl"):
        run_ids.append(
            hashlib.sha256(str(request_file.parent.resolve()).encode()).hexdigest()[:24]
        )
        has_incomplete_tail = has_incomplete_tail or incomplete_tail(request_file)
        for row in complete_rows(request_file):
            if row.get("phase") in counts:
                counts[row["phase"]] += 1
            if row.get("status") in ("ok", "failed", "usage_missing"):
                counts[row["status"]] += 1
    record.setdefault("progress", {})["request_counts"] = counts
    record["progress"]["stage"] = record.get("stage", record["status"])
    record["progress"]["incomplete_request_tail"] = has_incomplete_tail
    record["run_ids"] = run_ids
    record["warnings"] = (
        [
            f"{counts['usage_missing']} requests completed transport with missing server usage; excluded from status=ok measured successes"
        ]
        if counts["usage_missing"]
        else []
    )
    if record["status"] in ("succeeded", "failed", "cancelled") and process_matches(
        record
    ):
        record["status"] = "running"
        record["stage"] = "finalizing"
        record["progress"]["stage"] = "finalizing"
    return record


def start_task(request):
    """根据 request 固定配置并启动自有执行器，返回持久任务记录。"""
    import psutil
    import uuid

    if request.workspace == "formal":
        raise ValueError(
            "Tasks must use an isolated workspace; formal data/results are read-only"
        )
    from .config import workspace_root

    root = workspace_root(request.workspace)
    if request.kind in ("prepare", "rebuild") and not request.confirm:
        if request.kind == "rebuild" or (root / "metadata.json").exists():
            raise ValueError("Overwrite requires confirm=true")
    with resource_lock():
        task_id = uuid.uuid4().hex
        directory = task_path(task_id)
        snapshot = configs()
        from .config import validate_configs

        validate_configs(snapshot)
        from scripts.run_baseline import code_snapshot

        snapshot["source_sha256"] = code_snapshot()
        from .runtime import capture_runtime

        snapshot["runtime"] = capture_runtime(
            snapshot, require_llm=request.kind in ("rag", "baseline")
        )
        directory.mkdir(parents=True)
        config = effective_config(snapshot, request.workspace, directory / "runs")
        save_json(directory / "snapshot.json", snapshot)
        save_json(directory / "config.json", config)
        record = dict(
            id=task_id,
            **request.model_dump(),
            status="queued",
            created_at=time.time(),
            started_at=None,
            finished_at=None,
            error=None,
            snapshot=snapshot,
            effective_config=config,
            output=str(directory / "runs"),
        )
        save_json(directory / "task.json", record)
        with (directory / "task.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "memmgr.webui.tasks", task_id],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(
                    os.environ,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    OMP_NUM_THREADS="2",
                ),
            )
        record.update(
            pid=process.pid,
            process_created_at=psutil.Process(process.pid).create_time(),
            process_identity=process_identity(process.pid),
        )
        save_json(directory / "task.json", record)
        import threading

        threading.Thread(
            target=monitor_task, args=(task_id, process), daemon=True
        ).start()
    return record


def cancel_task(task_id):
    """取消 task_id 且身份核验通过的自有进程组，不操作其他服务。"""
    record = read_task(task_id)
    if record["status"] not in ("queued", "running"):
        raise ValueError("Task is already terminal")
    if not process_matches(record):
        raise ValueError("Owned task process is no longer alive")
    if os.getpgid(record["pid"]) != record["pid"]:
        raise RuntimeError("Owned process group identity mismatch")
    save_json(task_path(task_id) / "cancel.json", {"requested_at": time.time()})
    os.killpg(record["pid"], signal.SIGTERM)
    return {"id": task_id, "cancellation_requested": True}


def monitor_task(task_id, process):
    """等待本后端拥有的 process 实际退出，为 task_id 保存真实退出码并清理异常遗留组。"""
    code = process.wait()
    path = task_path(task_id) / "task.json"
    record = read_json(path)
    if record["status"] in ("queued", "running"):
        if owned_group_members(record):
            os.killpg(record["pid"], signal.SIGTERM)
            finish_group(record, include_leader=True)
        requested = (task_path(task_id) / "cancel.json").exists()
        record.update(
            status="cancelled" if requested else "interrupted",
            finished_at=time.time(),
            error=f"Owned runner exited without terminal state: returncode={code}",
        )
    record["process_returncode"] = code
    save_json(path, record)


def execute_task(task_id):
    """执行 task_id 的冻结配置与现有业务流程，所有终态在退出前持久化。"""
    directory = task_path(task_id)
    path = directory / "task.json"
    record = read_json(path)
    config = read_json(directory / "config.json")

    def cancelled(signum, frame):
        """响应 signum 取消；frame 为 Python 信号栈，保存真实取消终态。"""
        record.update(stage="cancelling")
        save_json(path, record)
        finish_group(record)
        record.update(
            status="cancelled",
            finished_at=time.time(),
            error=f"Cancelled by signal {signum}",
        )
        save_json(path, record)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancelled)
    try:
        with resource_lock(blocking=True) as lock:
            record.update(read_json(path))
            record.update(status="running", started_at=time.time())
            record["stage"] = record["kind"]
            save_json(path, record)
            from scripts.run_baseline import code_snapshot

            snapshot = read_json(directory / "snapshot.json")
            if code_snapshot() != snapshot["source_sha256"]:
                raise RuntimeError(
                    "Application or script source changed after task snapshot"
                )
            kind = record["kind"]
            if kind == "rebuild":
                from ..long_term.retrieval import open_collection

                collection, manifest = open_collection(config, record["tier"])
                import chromadb

                client = chromadb.HttpClient(
                    host=config["chroma"]["host"], port=config["chroma"]["port"]
                )
                identity = manifest["identity"]
                if (
                    not Path(identity["dataset"])
                    .resolve()
                    .is_relative_to(STATE / "workspaces" / record["workspace"])
                ):
                    raise ValueError("Rebuild collection is outside selected workspace")
                client.delete_collection(collection.name)
                (
                    Path(config["dataset"]["root"])
                    / "index-manifests"
                    / f"{collection.name}.json"
                ).unlink()
                kind = "index"
            if kind == "prepare":
                from ..long_term.data import prepare

                result = prepare(config)
                print(json.dumps(result), flush=True)
                manifests = Path(config["dataset"]["root"]) / "index-manifests"
                for manifest_file in manifests.glob("*.json"):
                    manifest_file.unlink()
                save_json(directory / "prepare.json", result)
                command = None
            elif kind == "baseline":
                command = [
                    sys.executable,
                    "scripts/run_baseline.py",
                    "--config",
                    str(directory / "config.json"),
                    "--output-root",
                    str(directory / "execution"),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "memmgr",
                    "--config",
                    str(directory / "config.json"),
                    {"rag": "run"}.get(kind, kind),
                    "--tier",
                    record["tier"],
                ]
                if kind in ("rag", "retrieval"):
                    command += ["--concurrency", str(record["concurrency"])]
            if command:
                record["argv"] = command
                save_json(path, record)
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    pass_fds=(lock.fileno(),),
                    env=dict(os.environ, MEMMGR_RESOURCE_LOCK_FD=str(lock.fileno())),
                )
                from .records import complete_rows

                for failure_file in (directory / "runs").glob("*/failures.jsonl"):
                    for failure in complete_rows(failure_file):
                        if failure.get("status") == "failed":
                            raise RuntimeError(
                                f"Experiment request failed: {failure.get('error')}; source={failure_file}"
                            )
            record.update(status="succeeded", finished_at=time.time(), returncode=0)
            save_json(path, record)
    except Exception as exc:
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        from .records import complete_rows

        failures = (
            row
            for failure_file in (directory / "runs").glob("*/failures.jsonl")
            for row in complete_rows(failure_file)
            if row.get("status") == "failed"
        )
        first_failure = next(failures, None)
        if first_failure is not None:
            error += f"; request_error={first_failure['error']}"
        record.update(
            status="failed",
            finished_at=time.time(),
            error=error,
            returncode=getattr(exc, "returncode", None),
        )
        save_json(path, record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    execute_task(parser.parse_args().task_id)
