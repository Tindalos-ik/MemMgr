"""使用临时代码/config 副本验证真实服务启动边界；显式执行 python tests/web_service_start.py。

无 GPU 条件通过 CUDA_VISIBLE_DEVICES='' 提供；不停止现有服务，不写正式数据或结果。
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    """取得本机当前可绑定端口，调用方立即使用该端口。"""
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def log_recovery():
    """在临时副本中实际断开/恢复 Docker daemon，检查原始诊断与 RFC3339 订阅恢复。"""
    sys.path.insert(0, str(Path.cwd()))
    from memmgr.web_config import STATE, configs, read_json
    from memmgr.web_logs import ChromaLogs, docker_timestamp

    container = configs()["baseline"]["resources"]["chroma_container"]
    path = STATE / "chroma.log"
    path.unlink(missing_ok=True)
    original_host = os.environ.get("DOCKER_HOST")
    os.environ["DOCKER_HOST"] = "unix:///tmp/memmgr-test-no-docker-daemon.sock"
    failed = ChromaLogs(container)
    failed.process.wait(timeout=15)
    failed.thread.join(timeout=15)
    diagnostic = path.read_bytes()
    assert failed.process.returncode != 0, diagnostic
    assert diagnostic and not any(
        docker_timestamp(line) for line in diagnostic.splitlines()
    ), diagnostic
    failed.close()
    if original_host is None:
        del os.environ["DOCKER_HOST"]
    else:
        os.environ["DOCKER_HOST"] = original_host
    recovered = ChromaLogs(container)
    try:
        initial = read_json(STATE / "chroma-log-status.json")
        assert "--since" not in initial["argv"], initial
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            content = path.read_bytes()
            if any(docker_timestamp(line) for line in content.splitlines()):
                break
            if recovered.process.poll() is not None:
                raise RuntimeError(content.decode(errors="replace"))
            time.sleep(0.1)
        assert diagnostic in content, content
        assert any(docker_timestamp(line) for line in content.splitlines()), content
    finally:
        recovered.close()
    # 再取得真实 daemon 诊断，证明日志末行为错误时仍从较早的真实 timestamp 续读。
    os.environ["DOCKER_HOST"] = "unix:///tmp/memmgr-test-no-docker-daemon.sock"
    failed_again = ChromaLogs(container)
    failed_again.process.wait(timeout=15)
    failed_again.thread.join(timeout=15)
    failed_again.close()
    if original_host is None:
        del os.environ["DOCKER_HOST"]
    else:
        os.environ["DOCKER_HOST"] = original_host
    final = ChromaLogs(container)
    try:
        status = read_json(STATE / "chroma-log-status.json")
        since = status["argv"][status["argv"].index("--since") + 1]
        assert docker_timestamp((since + " log").encode()) == since, status
        time.sleep(0.5)
        assert final.process.poll() is None, path.read_text()
        print(
            "docker_log_recovery",
            json.dumps(status),
            "diagnostic_bytes",
            len(diagnostic),
            flush=True,
        )
    finally:
        final.close()


class ServiceStartup(unittest.TestCase):
    """在真实隔离后端上检查 API、防重入、未知监听者和无 GPU 启动失败。"""

    def setUp(self):
        """创建当前代码/config 的临时副本并启动无可见 GPU 的独立后端。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="memmgr-service-start-")
        self.root = Path(self.temporary.name)
        for name in ("memmgr", "config", "scripts"):
            shutil.copytree(
                ROOT / name,
                self.root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(ROOT / "compose.yaml", self.root / "compose.yaml")
        self.vllm_port = free_port()
        server_path = self.root / "config/server.json"
        baseline_path = self.root / "config/baseline.json"
        server = json.loads(server_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        server["port"] = self.vllm_port
        baseline["llm"]["base_url"] = f"http://127.0.0.1:{self.vllm_port}/v1"
        server_path.write_text(json.dumps(server))
        baseline_path.write_text(json.dumps(baseline))
        self.environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(self.root),
        }
        self.backend_log = (self.root / "backend-output.log").open("wb")
        self.port = free_port()
        self.command = [
            sys.executable,
            "-m",
            "uvicorn",
            "memmgr.web:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.backend = subprocess.Popen(
            self.command,
            cwd=self.root,
            env=self.environment,
            stdout=self.backend_log,
            stderr=subprocess.STDOUT,
        )
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}", timeout=30, trust_env=False
        )
        self.wait_backend()
        self.listener = None

    def wait_backend(self):
        """等待本测试后端实际监听，进程退出时立即报告其原始输出。"""
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if self.backend.poll() is not None:
                self.fail((self.root / "backend-output.log").read_text())
            try:
                response = self.client.get("/api/logs")
                response.raise_for_status()
                return
            except httpx.ConnectError:
                time.sleep(0.1)
        self.fail("Isolated backend startup deadline exceeded")

    def tearDown(self):
        """结束本测试拥有的后端及无关 HTTP 监听者，删除临时副本。"""
        if self.listener is not None:
            self.listener.terminate()
            self.listener.wait(timeout=10)
        self.backend.terminate()
        self.backend.wait(timeout=30)
        self.client.close()
        self.backend_log.close()
        self.temporary.cleanup()

    def wait_terminal(self, name):
        """等待 name 的真实启动驱动写出终态；不重复任何启动请求。"""
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            record = (
                self.client.get("/api/services")
                .raise_for_status()
                .json()["services"][name]
            )
            if record["status"] != "starting":
                return record
            time.sleep(0.2)
        self.fail("Service startup deadline exceeded")

    def test_unknown_listener_and_origin(self):
        """实际 http.server 占用 vLLM 端口时拒绝启动，验证同源写安全及日志轮换游标。"""
        self.listener = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(self.vllm_port),
                "--bind",
                "127.0.0.1",
            ],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get(
                    f"http://127.0.0.1:{self.vllm_port}", trust_env=False
                ).raise_for_status()
                break
            except httpx.ConnectError:
                if time.monotonic() > deadline:
                    self.fail("Unrelated HTTP listener failed to start")
                time.sleep(0.05)
        response = self.client.post("/api/services/vllm/start", json={})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("occupied", response.text)
        record = (
            self.client.get("/api/services")
            .raise_for_status()
            .json()["services"]["vllm"]
        )
        self.assertEqual(record["status"], "failed", record)
        self.assertFalse(record["owned"])
        log = self.client.get("/api/logs/service-vllm").raise_for_status().json()
        self.assertIn("occupied", log["text"])
        second = self.client.post("/api/services/vllm/start", json={})
        self.assertEqual(second.status_code, 409, second.text)
        rotated = (
            self.client.get(
                "/api/logs/service-vllm",
                params={"cursor": log["cursor"], "identity": log["identity"]},
            )
            .raise_for_status()
            .json()
        )
        self.assertTrue(rotated["reset"])
        self.assertNotEqual(log["identity"], rotated["identity"])
        self.assertIn("occupied", rotated["text"])
        self.assertEqual(
            self.client.post(
                "/api/services/vllm/start",
                json={},
                headers={"Origin": "https://example.com"},
            ).status_code,
            403,
        )
        print("unknown_listener", response.status_code, record["error"], flush=True)

    def test_real_no_gpu_start_and_restart(self):
        """通过 API 实际启动无 GPU 的 vLLM，验证共享锁、防重复、真实失败日志及后端重启发现。"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(self.client.post, "/api/services/vllm/start", json={})
                for _ in range(2)
            ]
            responses = [future.result() for future in futures]
        if any(response.status_code == 200 for response in responses) and all(
            response.status_code in (200, 409) for response in responses
        ):
            # 现存外部 vLLM 的另一端口仍健康时，API 必须识别它，禁止另起 GPU 实例。
            record = next(
                response.json() for response in responses if response.status_code == 200
            )
            self.assertEqual(record["result"], "already_running", record)
            self.assertTrue(record["restart_required"], record)
            self.assertFalse(record["configuration_matches"], record)
            self.assertTrue(
                all(
                    item["port"] != self.vllm_port
                    for item in record["running_configuration"]
                )
            )
            with (self.root / ".agent/web/resource.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                blocked = self.client.post("/api/services/vllm/start", json={})
                self.assertEqual(blocked.status_code, 409, blocked.text)
                saved = {
                    name: json.loads(
                        (self.root / "config" / f"{name}.json").read_text()
                    )
                    for name in ("baseline", "server", "models")
                }
                self.assertEqual(
                    self.client.put("/api/config", json=saved).status_code, 409
                )
            self.backend.terminate()
            self.backend.wait(timeout=30)
            self.backend = subprocess.Popen(
                self.command,
                cwd=self.root,
                env=self.environment,
                stdout=self.backend_log,
                stderr=subprocess.STDOUT,
            )
            self.wait_backend()
            discovered = (
                self.client.get("/api/services")
                .raise_for_status()
                .json()["services"]["vllm"]
            )
            self.assertEqual(
                discovered["running_configuration"], record["running_configuration"]
            )
            # 独立验证 memmgr.server 在无可见 GPU 时的真实错误，不绕过 API 的防重复保护。
            failed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "memmgr.server",
                    "--server",
                    str(self.root / "config/server.json"),
                    "--models",
                    str(self.root / "config/models.json"),
                ],
                cwd=self.root,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=150,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Traceback", failed.stderr)
            print(
                "cross_port_already_running",
                discovered["running_configuration"],
                "no_visible_gpu_exit",
                failed.returncode,
                failed.stderr[-2500:],
                flush=True,
            )
            return
        self.assertEqual(
            sorted(response.status_code for response in responses),
            [202, 409],
            [response.text for response in responses],
        )
        accepted = next(
            response.json() for response in responses if response.status_code == 202
        )
        self.assertEqual(accepted["status"], "starting")
        saved = {
            name: json.loads((self.root / "config" / f"{name}.json").read_text())
            for name in ("baseline", "server", "models")
        }
        busy = self.client.put("/api/config", json=saved)
        self.assertEqual(busy.status_code, 409, busy.text)
        # 后端退出保持启动 driver；无 GPU 子进程必须由真实依赖自行失败。
        self.backend.terminate()
        self.backend.wait(timeout=30)
        self.backend = subprocess.Popen(
            self.command,
            cwd=self.root,
            env=self.environment,
            stdout=self.backend_log,
            stderr=subprocess.STDOUT,
        )
        self.wait_backend()
        record = self.wait_terminal("vllm")
        self.assertEqual(record["status"], "failed", record)
        self.assertIn("exited with code", record["error"])
        self.assertEqual(record["driver_identity"], accepted["driver_identity"])
        self.assertFalse(record["owned"])
        log = self.client.get("/api/logs/service-vllm").raise_for_status().json()
        self.assertTrue(log["available"])
        self.assertIn("Traceback", log["text"])
        print(
            "no_visible_gpu",
            record["driver_pid"],
            record["status"],
            record["error"][-4000:],
            flush=True,
        )

    def test_compose_driver_missing_docker(self):
        """直接运行同一持久 driver，在真实缺 docker 条件下验证 Compose 退出码与错误日志。"""
        directory = self.root / ".agent/web/services/chroma"
        directory.mkdir(parents=True)
        snapshot = {
            name: json.loads((self.root / "config" / f"{name}.json").read_text())
            for name in ("baseline", "server", "models")
        }
        # 独立 driver 的探测端口为空；正式 8001 监听者和持久卷完全不动。
        snapshot["baseline"]["chroma"]["port"] = free_port()
        record = dict(
            name="chroma",
            status="starting",
            stage="launching",
            owned=False,
            started_at=time.time(),
            error=None,
            snapshot=snapshot,
        )
        path = directory / "service.json"
        path.write_text(json.dumps(record))
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "bash").symlink_to(shutil.which("bash"))
        environment = {**self.environment, "PATH": str(binary)}
        read_gate, write_gate = os.pipe()
        with (
            (self.root / ".agent/web/resource.lock").open("a") as lock,
            (directory / "service.log").open("wb") as log,
        ):
            fcntl.flock(lock, fcntl.LOCK_EX)
            driver = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "memmgr.web_services",
                    "chroma",
                    str(lock.fileno()),
                    str(read_gate),
                ],
                cwd=self.root,
                env=environment,
                pass_fds=(lock.fileno(), read_gate),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            stat = Path(f"/proc/{driver.pid}/stat").read_text()
            record.update(
                driver_pid=driver.pid,
                driver_identity={
                    "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id")
                    .read_text()
                    .strip(),
                },
            )
            path.write_text(json.dumps(record))
            os.write(write_gate, b"1")
            os.close(read_gate)
            os.close(write_gate)
            self.assertNotEqual(driver.wait(timeout=40), 0)
        persisted = json.loads(path.read_text())
        output = (directory / "service.log").read_text()
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("docker: command not found", output)
        self.assertIn("code 127", persisted["error"])
        self.assertEqual(
            self.client.get("/api/services")
            .raise_for_status()
            .json()["services"]["chroma"]["status"],
            "failed",
        )
        print("missing_docker", persisted["status"], persisted["error"], flush=True)

    def test_real_docker_log_recovery(self):
        """真实 Docker daemon 不可达后恢复订阅，保留原错误并读取已有容器真实输出。"""
        self.backend.terminate()
        self.backend.wait(timeout=30)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--log-recovery"],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=75,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout, end="", flush=True)

    def test_only_required_service_configuration(self):
        """真实解析生成缓存，验证服务校验不依赖 embedding 且实验设置不会触发重启差异。"""
        code = '''"""在临时副本中检查服务启动实际所需配置。"""
import copy
from memmgr.web_config import configs
from memmgr.web_services import service_configuration, validate_service_configuration
original = configs()
changed = copy.deepcopy(original)
changed["baseline"]["experiment"]["top_k"] += 1
changed["models"].pop("BAAI/bge-small-en-v1.5")
assert service_configuration(original, "vllm") == service_configuration(changed, "vllm")
assert service_configuration(original, "chroma") == service_configuration(changed, "chroma")
path = validate_service_configuration(changed, "vllm")
assert path.is_dir() and path.name == changed["server"]["revision"], path
changed["models"] = {}
assert validate_service_configuration(changed, "chroma") is None
print("required_service_configuration", str(path), "Chroma does not resolve models")
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout, end="", flush=True)


class ServiceGroupLifecycle(unittest.TestCase):
    """仅用真实独立 session 与 sleep 检查失败启动的残留进程清理边界。"""

    def test_exited_leader_cleanup(self):
        """leader 实际退出1且子进程继续时清理整组，并核实另一独立 session 未受影响。"""
        sys.path.insert(0, str(ROOT))
        from memmgr.web_services import (
            alive,
            finish_service_group,
            service_group_members,
        )
        from memmgr.web_tasks import process_identity

        sentinel = subprocess.Popen(["sleep", "120"], start_new_session=True)
        sentinel_identity = process_identity(sentinel.pid)
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import os, subprocess; child = subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL); print(child.pid, flush=True); os._exit(1)',
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        record = {
            "service_pid": leader.pid,
            "service_identity": process_identity(leader.pid),
        }
        try:
            child_pid = int(leader.stdout.readline())
            child_identity = process_identity(child_pid)
            self.assertEqual(leader.wait(timeout=10), 1)
            self.assertTrue(alive(child_pid, child_identity))
            self.assertEqual(os.getsid(child_pid), leader.pid)
            self.assertEqual(os.getpgid(child_pid), leader.pid)
            self.assertEqual(
                [item.pid for item in service_group_members(record)], [child_pid]
            )
            finish_service_group(record)
            self.assertEqual(service_group_members(record), [])
            self.assertFalse(alive(child_pid, child_identity))
            self.assertTrue(alive(sentinel.pid, sentinel_identity))
            print(
                "exited_leader_group_cleanup",
                record,
                "child",
                child_pid,
                "external_sentinel_alive",
                sentinel.pid,
                flush=True,
            )
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=10)
            finish_service_group(record)
            leader.wait(timeout=10)
            leader.stdout.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--log-recovery"]:
        log_recovery()
    else:
        unittest.main()
