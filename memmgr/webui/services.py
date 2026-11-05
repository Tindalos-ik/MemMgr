"""以固定配置和真实健康检查启动本机 vLLM/Chroma，保留独立进程身份及原始日志。"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from ..inference.service_status import health_snapshot
from .config import ROOT, STATE, configs, read_json, save_json
from .tasks import process_identity

NAMES = ("vllm", "chroma")


def service_path(name):
    """校验固定服务 name，返回专用状态目录。"""
    if name not in NAMES:
        raise ValueError("Unknown service; expected vllm or chroma")
    return STATE / "services" / name


def alive(pid, identity):
    """按 pid 与 identity 的内核启动标识检查进程存活，排除 PID 复用及僵尸。"""
    import psutil

    try:
        return (
            process_identity(pid) == identity
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        )
    except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
        return False


def service_group_members(record):
    """按 record 的独立 session/PGID 和内核启动身份核验仍存活的自有组成员。"""
    import psutil

    leader = record["service_pid"]
    identity = record["service_identity"]
    if process_identity(os.getpid())["boot_id"] != identity["boot_id"]:
        return []
    try:
        actual = process_identity(leader)
    except (FileNotFoundError, ProcessLookupError):
        actual = None
    if actual is not None and actual != identity:
        raise RuntimeError("Service process identity changed; refusing group cleanup")
    members = []
    for process in psutil.process_iter(["pid", "status"]):
        try:
            if os.getpgid(process.pid) != leader:
                continue
            member_identity = process_identity(process.pid)
            if (
                os.getsid(process.pid) != leader
                or member_identity["boot_id"] != identity["boot_id"]
                or member_identity["start_ticks"] < identity["start_ticks"]
            ):
                raise RuntimeError(
                    "Cannot prove service session ownership; refusing group cleanup"
                )
            if process.info["status"] != psutil.STATUS_ZOMBIE:
                members.append(process)
        except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
            continue
    return members


def finish_service_group(record):
    """终止 record 核验通过的自有进程组，根进程已退出时也等待残留成员全部停止。"""
    for signum, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not service_group_members(record):
            return
        try:
            os.killpg(record["service_pid"], signum)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not service_group_members(record):
                return
            time.sleep(0.1)
    remaining = [process.pid for process in service_group_members(record)]
    if remaining:
        raise RuntimeError(f"Owned service group has not stopped: {remaining}")


def endpoint(snapshot, name):
    """由固定 snapshot 返回 name 的本机监听地址。"""
    config = snapshot["server"] if name == "vllm" else snapshot["baseline"]["chroma"]
    host, port = config["host"], config["port"]
    if host not in ("localhost", "127.0.0.1"):
        raise ValueError("Only loopback service binding is allowed")
    return host, port


def occupied(snapshot, name):
    """检查 snapshot 中 name 的 TCP 端口是否已有监听者。"""
    with socket.socket() as stream:
        stream.settimeout(1)
        return stream.connect_ex(endpoint(snapshot, name)) == 0


def health(snapshot, name):
    """对 snapshot 中 name 使用共享真实健康接口，保留原始失败原因。"""
    return asyncio.run(health_snapshot(snapshot["baseline"], (name,)))[name]


def service_configuration(snapshot, name):
    """提取 snapshot 中 name 实际启动相关配置，排除实验/embedding 的无关更改。"""
    if name == "vllm":
        server = snapshot["server"]
        return {"server": server, "model": snapshot["models"][server["model"]]}
    return {
        "endpoint": endpoint(snapshot, name),
        "container": snapshot["baseline"]["resources"]["chroma_container"],
    }


def log_tail(directory):
    """从 directory 的专用日志读取末尾有界字节，保留实际子进程错误。"""
    path = directory / "service.log"
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        return stream.read(16384).decode("utf-8", errors="replace")


def local_vllm_processes():
    """发现实际本机 vLLM API 进程的完整 argv 与内核身份，阻止跨端口重复占用 GPU。"""
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "cmdline", "status"]):
        command = process.info["cmdline"] or []
        if (
            "vllm.entrypoints.openai.api_server" not in command
            or process.info["status"] == psutil.STATUS_ZOMBIE
        ):
            continue
        try:
            options = {}
            for index, item in enumerate(command):
                if item.startswith("--"):
                    key, separator, value = item.partition("=")
                    options[key] = (
                        value
                        if separator
                        else command[index + 1]
                        if index + 1 < len(command)
                        else None
                    )
            host = options.get("--host", "127.0.0.1")
            if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
                continue
            found.append(
                {
                    "pid": process.pid,
                    "process_identity": process_identity(process.pid),
                    "argv": command,
                    "host": "127.0.0.1" if host == "0.0.0.0" else host,
                    "port": int(options.get("--port", 8000)),
                }
            )
        except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
            continue
    return found


def validate_service_configuration(snapshot, name):
    """复用固定 schema 校验 snapshot 中 name 所需配置，仅解析 vLLM 的生成模型。"""
    from .schema import ChromaConfig, ServerConfig, ModelSnapshot, LLMConfig

    if name == "chroma":
        ChromaConfig.model_validate(snapshot["baseline"]["chroma"])
        if snapshot["baseline"]["resources"]["chroma_container"] != "memmgr-chroma-1":
            raise ValueError(
                "Only memmgr-chroma-1 is supported by the constrained Compose entry"
            )
        return None
    server = snapshot["server"]
    llm = snapshot["baseline"]["llm"]
    ServerConfig.model_validate(server)
    LLMConfig.model_validate(llm)
    model = snapshot["models"][server["model"]]
    ModelSnapshot.model_validate(model)
    if (
        model["revision"] != server["revision"]
        or llm["model"] != server["model"]
        or llm["tokenizer"] != server["model"]
        or llm["revision"] != server["revision"]
        or llm["max_model_len"] != server["max_model_len"]
        or llm["base_url"] != f"http://{server['host']}:{server['port']}/v1"
    ):
        raise ValueError(
            "Server and LLM model/tokenizer/revision/context/endpoint must agree"
        )
    from ..inference.models import local_snapshot

    return local_snapshot(server["model"], model["revision"])


def chroma_container(snapshot):
    """核验 snapshot 所选 Compose Chroma 容器的标签、端口与运行状态，返回真实 ID。"""
    inspected = subprocess.run(
        ["docker", "inspect", snapshot["baseline"]["resources"]["chroma_container"]],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if inspected.returncode:
        raise RuntimeError(inspected.stderr or inspected.stdout)
    container = json.loads(inspected.stdout)[0]
    labels = container["Config"]["Labels"]
    bindings = container["NetworkSettings"]["Ports"].get("8000/tcp") or []
    host, port = endpoint(snapshot, "chroma")
    if (
        labels.get("com.docker.compose.project") != "memmgr"
        or labels.get("com.docker.compose.service") != "chroma"
        or not container["State"]["Running"]
        or not any(
            item["HostIp"] == host and item["HostPort"] == str(port)
            for item in bindings
        )
    ):
        raise RuntimeError(
            "Cannot identify the configured running Compose Chroma container and loopback port"
        )
    return container["Id"]


def service_status(name):
    """重新发现 name 的真实健康及自有启动进程，返回持久状态与配置快照。"""
    directory = service_path(name)
    path = directory / "service.json"
    current = configs()
    record = (
        read_json(path)
        if path.exists()
        else dict(
            name=name,
            status="stopped",
            stage="not_started",
            owned=False,
            started_at=None,
            error=None,
        )
    )
    snapshot = record.get("snapshot", current)
    observed = health(snapshot, name)
    driver_alive = bool(
        record.get("driver_pid")
        and alive(record["driver_pid"], record["driver_identity"])
    )
    service_alive = bool(
        record.get("service_pid")
        and alive(record["service_pid"], record["service_identity"])
    )
    if record.get("stage") == "cleanup_failed":
        return {
            **record,
            "health": observed,
            "log_source": "service-" + name,
            "owned": service_alive,
            "elapsed_seconds": time.time() - record["started_at"],
        }
    if record["status"] == "starting" and not driver_alive:
        record = read_json(path)
        if record["status"] == "starting":
            record.update(
                status="failed",
                stage="driver_exited",
                error="Startup driver exited without a terminal state; "
                + log_tail(directory),
                finished_at=time.time(),
            )
            save_json(path, record)
    elif (
        record["status"] == "running"
        and record.get("service_pid")
        and not service_alive
    ):
        record.update(
            status="failed",
            stage="process_exited",
            error="Owned service process exited; see service log",
            finished_at=time.time(),
        )
        save_json(path, record)
    if observed["status"] == "ok" and record["status"] not in ("starting", "failed"):
        record.update(status="running", stage="healthy", error=None)
    elif record["status"] == "running" and observed["status"] != "ok":
        record.update(
            status="unavailable", stage="health_failed", error=observed["error"]
        )
    elif record["status"] not in ("starting", "failed"):
        record.update(
            status="unavailable" if occupied(snapshot, name) else "stopped",
            error=observed.get("error"),
        )
        if name == "chroma":
            try:
                diagnostic = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if diagnostic.returncode:
                    record.update(
                        status="unavailable",
                        error=diagnostic.stderr.strip() or diagnostic.stdout.strip(),
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                record.update(
                    status="unavailable", error=f"{type(exc).__name__}: {exc}"
                )
    verified = False
    if name == "chroma" and (observed["status"] == "ok" or record.get("container_id")):
        try:
            actual_container = chroma_container(snapshot)
            verified = True
            record["owned"] = actual_container == record.get("container_id")
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            record.update(owned=False, ownership_error=f"{type(exc).__name__}: {exc}")
            if observed["status"] == "ok" and record["status"] != "failed":
                record.update(
                    status="unavailable",
                    stage="identity_unverified",
                    error=record["ownership_error"],
                )
    matches = service_configuration(snapshot, name) == service_configuration(
        current, name
    )
    if name == "chroma" and record.get("compose_sha256"):
        matches = (
            matches
            and hashlib.sha256((ROOT / "compose.yaml").read_bytes()).hexdigest()
            == record["compose_sha256"]
        )
    record.update(
        health=observed,
        configured_health=health(current, name) if snapshot != current else observed,
        log_source="service-" + name,
        owned=service_alive if name == "vllm" else record.get("owned", False),
        configuration_matches=matches,
    )
    record["verified_running"] = verified
    record["configured_service_identified"] = verified and endpoint(
        snapshot, name
    ) == endpoint(current, name)
    if name == "vllm":
        processes = local_vllm_processes()
        record["running_configuration"] = processes
        record["verified_running"] = bool(processes) or service_alive
        record["configured_service_identified"] = any(
            (item["host"], item["port"]) == endpoint(current, name)
            for item in processes
        )
        if not record["verified_running"] and observed["status"] == "ok":
            record.update(
                status="unavailable",
                stage="identity_unverified",
                error="Cannot identify a local vLLM process for the healthy endpoint",
            )
        if processes and not service_alive and record["status"] != "starting":
            actual = next(
                (
                    item
                    for item in processes
                    if (item["host"], item["port"]) == endpoint(current, name)
                ),
                processes[0],
            )
            actual_config = {
                **current,
                "baseline": {
                    **current["baseline"],
                    "llm": {
                        **current["baseline"]["llm"],
                        "base_url": f"http://{actual['host']}:{actual['port']}/v1",
                    },
                },
            }
            observed = health(actual_config, name)
            record.update(health=observed, owned=False)
            if record["status"] != "failed":
                record.update(
                    status="running" if observed["status"] == "ok" else "unavailable",
                    stage="external_process",
                    error=observed.get("error"),
                )
            from .runtime import capture_runtime

            runtime = capture_runtime(current)
            record.update(
                configuration_matches=runtime["configuration_matches"],
                configuration_differences=runtime["differences"],
                runtime=runtime,
            )
        record["restart_required"] = (
            bool(processes) and not record["configuration_matches"]
        )
        if record["restart_required"]:
            record["message"] = (
                "A local vLLM instance is still running with different configuration. Handle the existing service manually before applying the saved configuration; START will not launch a second GPU instance."
            )
    record["elapsed_seconds"] = (
        ((record.get("finished_at") or time.time()) - record["started_at"])
        if record.get("started_at")
        else None
    )
    return record


def fail(directory, record, exc):
    """将 directory 中 record 的失败 exc 原文写入状态和专用日志。"""
    message = f"{type(exc).__name__}: {exc}"
    with (directory / "service.log").open("a") as stream:
        stream.write(message + "\n")
    record.update(
        status="failed", stage="failed", error=message, finished_at=time.time()
    )
    save_json(directory / "service.json", record)


def start_service(name):
    """启动固定 name；同步拒绝冲突，将共享锁交给独立 runner 直到真实健康或失败。"""
    directory = service_path(name)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (STATE / "resource.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = service_status(name)
        if existing.get("stage") == "cleanup_failed":
            raise BlockingIOError(existing["error"])
        if existing["status"] == "starting":
            raise BlockingIOError("Service startup is already in progress")
        snapshot = configs()
        if (
            existing["health"]["status"] == "ok"
            and existing["verified_running"]
            and (
                not occupied(snapshot, name)
                or existing["configured_service_identified"]
            )
        ):
            return {**existing, "status": "running", "result": "already_running"}
        attempt_id = uuid.uuid4().hex
        record = dict(
            name=name,
            status="starting",
            stage="validating",
            owned=False,
            started_at=time.time(),
            finished_at=None,
            error=None,
            snapshot=snapshot,
            log_source="service-" + name,
            attempt_id=attempt_id,
        )
        fresh_log = directory / (".service-" + attempt_id + ".log")
        fresh_log.write_text("")
        fresh_log.replace(directory / "service.log")
        try:
            if occupied(snapshot, name):
                raise BlockingIOError(
                    f"Refusing startup: {endpoint(snapshot, name)} is occupied by an unknown/unhealthy listener; {existing['configured_health'].get('error')}"
                )
            if name == "vllm" and (
                existing.get("running_configuration") or existing.get("owned")
            ):
                raise BlockingIOError(
                    "A local vLLM process is still alive; handle the existing service manually before starting another instance. "
                    + str(existing.get("running_configuration"))
                )
            resolved_model = validate_service_configuration(snapshot, name)
            if resolved_model is not None:
                record["resolved_model_path"] = str(resolved_model)
            if name == "chroma" and endpoint(snapshot, name) != ("127.0.0.1", 8001):
                raise ValueError("compose.yaml only supports Chroma at 127.0.0.1:8001")
            if name == "chroma":
                record["compose_sha256"] = hashlib.sha256(
                    (ROOT / "compose.yaml").read_bytes()
                ).hexdigest()
            for key in ("server", "models"):
                save_json(directory / (key + ".json"), snapshot[key])
            read_gate, write_gate = os.pipe()
            try:
                with (directory / "service.log").open("ab") as log:
                    driver = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "memmgr.webui.services",
                            name,
                            str(lock.fileno()),
                            str(read_gate),
                        ],
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock.fileno(), read_gate),
                    )
                record.update(
                    driver_pid=driver.pid,
                    driver_identity=process_identity(driver.pid),
                    stage="launching",
                )
                save_json(directory / "service.json", record)
                os.write(write_gate, b"1")
            finally:
                os.close(read_gate)
                os.close(write_gate)
            import threading

            threading.Thread(target=driver.wait, daemon=True).start()
            return {
                **record,
                "health": existing["health"],
                "elapsed_seconds": 0,
                "result": "starting",
            }
        except Exception as exc:
            fail(directory, record, exc)
            raise
    finally:
        # close 保持继承给 runner 的同一文件描述锁，禁止在父进程显式解锁。
        lock.close()


def run(name, lock_fd, gate_fd):
    """为 name 执行实际启动并持有 lock_fd；gate_fd 保证父进程已写入进程身份。"""
    if os.read(gate_fd, 1) != b"1":
        raise RuntimeError("Backend exited before startup handoff")
    os.close(gate_fd)
    directory = service_path(name)
    record = read_json(directory / "service.json")
    process = None
    try:
        snapshot = record["snapshot"]
        if occupied(snapshot, name):
            raise RuntimeError(
                f"Service port became occupied before launch: {endpoint(snapshot, name)}"
            )
        if name == "vllm" and local_vllm_processes():
            raise RuntimeError(
                "A local vLLM process appeared before launch; refusing a second GPU instance"
            )
        if (
            name == "chroma"
            and record.get("compose_sha256")
            and hashlib.sha256((ROOT / "compose.yaml").read_bytes()).hexdigest()
            != record["compose_sha256"]
        ):
            raise RuntimeError("compose.yaml changed after startup snapshot")
        argv = (
            [
                "bash",
                str(ROOT / "scripts" / "start-vllm.sh"),
                str(directory / "server.json"),
                str(directory / "models.json"),
            ]
            if name == "vllm"
            else ["bash", str(ROOT / "scripts" / "chroma.sh"), "up"]
        )
        environment = {
            **os.environ,
            "PYTHON": sys.executable,
            "PROJECT_ROOT": str(ROOT),
        }
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        record.update(
            service_pid=process.pid,
            service_identity=process_identity(process.pid),
            stage="waiting_for_health",
        )
        save_json(directory / "service.json", record)
        deadline = time.monotonic() + (600 if name == "vllm" else 200)
        while True:
            code = process.poll()
            if code is not None and (name == "vllm" or code != 0):
                raise RuntimeError(
                    f"{name} startup exited with code {code};\n" + log_tail(directory)
                )
            observed = health(snapshot, name)
            if observed["status"] == "ok" and (name == "vllm" or code == 0):
                if name == "chroma":
                    record["container_id"] = chroma_container(snapshot)
                    record.pop("service_pid", None)
                    record.pop("service_identity", None)
                record.update(
                    status="running",
                    stage="healthy",
                    health=observed,
                    owned=True,
                    finished_at=time.time(),
                )
                save_json(directory / "service.json", record)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{name} health deadline exceeded: {observed.get('error')}"
                )
            time.sleep(1)
    except Exception as exc:
        traceback.print_exc()
        if process is not None:
            try:
                finish_service_group(record)
                process.wait(timeout=5)
            except Exception as cleanup_error:
                traceback.print_exc()
                record.update(
                    status="unavailable",
                    stage="cleanup_failed",
                    error=f"{type(exc).__name__}: {exc}\nCleanup error: {type(cleanup_error).__name__}: {cleanup_error}",
                )
                save_json(directory / "service.json", record)
                raise
        fail(directory, record, exc)
        raise
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=NAMES)
    parser.add_argument("lock_fd", type=int)
    parser.add_argument("gate_fd", type=int)
    arguments = parser.parse_args()
    run(arguments.name, arguments.lock_fd, arguments.gate_fd)
