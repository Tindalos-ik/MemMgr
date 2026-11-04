"""对真实隔离任务执行取消、SIGKILL 与后端冷重启验收，不使用模拟进程。

显式运行：MEMMGR_WEB_URL=http://127.0.0.1:8765 MEMMGR_TEST_WORKSPACE=<自行准备并索引的隔离工作区>
MEMMGR_ALLOW_BACKEND_RESTART=1 python tests/web_task_lifecycle.py。
该测试会终止当前自有 Web 后端，测试后需重新启动持久服务。
"""

from __future__ import annotations
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]


class LiveTaskLifecycle(unittest.TestCase):
    """只对 API 返回且身份核验通过的自有任务进程组实施真实信号测试。"""

    @classmethod
    def setUpClass(cls):
        """为 cls 连接实际后端；独占测试必须通过环境变量显式启用。"""
        if os.environ.get("MEMMGR_ALLOW_BACKEND_RESTART") != "1":
            raise RuntimeError("Explicit MEMMGR_ALLOW_BACKEND_RESTART=1 is required")
        cls.url = os.environ["MEMMGR_WEB_URL"]
        cls.workspace = os.environ["MEMMGR_TEST_WORKSPACE"]
        cls.client = httpx.Client(base_url=cls.url, timeout=30, trust_env=False)
        cls.created = []
        cls.restarted_backend = None

    @classmethod
    def tearDownClass(cls):
        """取消 cls 创建的剩余真实任务，关闭本测试启动的有限生命周期后端。"""
        for task in cls.created:
            try:
                record = cls.client.get("/api/tasks/" + task["id"]).json()
                if record.get("status") in ("queued", "running"):
                    cls.client.post(
                        "/api/tasks/" + task["id"] + "/cancel"
                    ).raise_for_status()
            except httpx.TransportError:
                pass
        cls.client.close()
        if cls.restarted_backend is not None:
            cls.restarted_backend.terminate()
            cls.restarted_backend.wait(timeout=30)

    def create(self):
        """创建真实已建索引 workspace 的 retrieval 任务并登记测试归属。"""
        response = self.client.post(
            "/api/tasks",
            json={"kind": "retrieval", "workspace": self.workspace, "tier": "small"},
        )
        response.raise_for_status()
        task = response.json()
        self.created.append(task)
        return task

    def wait_state(self, task, terminal=False, child=False):
        """等待 task 的实际运行/终态；terminal 控制终态，child 要求已启动业务子进程。"""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            record = (
                self.client.get("/api/tasks/" + task["id"]).raise_for_status().json()
            )
            if terminal and record["status"] not in ("queued", "running"):
                return record
            if not terminal and record["status"] == "running":
                if not child or psutil.Process(task["pid"]).children(recursive=True):
                    return record
            if not terminal and record["status"] in (
                "failed",
                "cancelled",
                "interrupted",
            ):
                self.fail(json.dumps(record, ensure_ascii=False))
            time.sleep(0.2)
        self.fail("Actual task state deadline exceeded")

    def assert_group_gone(self, task):
        """核实 task 的真实进程组无存活成员，僵尸进程不视为继续执行。"""
        remaining = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if (
                    os.getpgid(process.pid) == task["pid"]
                    and process.info["status"] != psutil.STATUS_ZOMBIE
                ):
                    remaining.append(process.pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
        self.assertEqual(remaining, [])

    def test_1_immediate_queued_cancel(self):
        """创建后立即取消，验证早于信号 handler 安装的取消也持久记录 cancelled。"""
        task = self.create()
        self.client.post("/api/tasks/" + task["id"] + "/cancel").raise_for_status()
        record = self.wait_state(task, terminal=True)
        self.assertEqual(record["status"], "cancelled", record)
        self.assert_group_gone(task)
        print("queued_cancel", task["id"], record["status"], flush=True)

    def test_2_running_group_cancel(self):
        """业务子进程实际启动后取消，验证 cancelled 之前整组已经结束。"""
        task = self.create()
        self.wait_state(task, child=True)
        self.client.post("/api/tasks/" + task["id"] + "/cancel").raise_for_status()
        record = self.wait_state(task, terminal=True)
        self.assertEqual(record["status"], "cancelled", record)
        self.assert_group_gone(task)
        print("running_cancel", task["id"], record["status"], flush=True)

    def test_3_killed_task_recovery(self):
        """只 SIGKILL 真实自有 runner，验证恢复器清除遗留业务组并记录 interrupted。"""
        task = self.create()
        self.wait_state(task, child=True)
        process = psutil.Process(task["pid"])
        self.assertIn("memmgr.web_tasks", process.cmdline())
        self.assertIn(task["id"], process.cmdline())
        stat = Path(f"/proc/{process.pid}/stat").read_text()
        self.assertEqual(
            int(stat[stat.rfind(")") + 2 :].split()[19]),
            task["process_identity"]["start_ticks"],
        )
        self.assertEqual(
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            task["process_identity"]["boot_id"],
        )
        child = process.children(recursive=True)[0]
        descriptor = child.environ()["MEMMGR_RESOURCE_LOCK_FD"]
        self.assertEqual(
            Path(f"/proc/{child.pid}/fd/{descriptor}").stat().st_ino,
            (ROOT / ".agent/web/resource.lock").stat().st_ino,
        )
        process.send_signal(signal.SIGKILL)
        record = self.wait_state(task, terminal=True)
        self.assertEqual(record["status"], "interrupted", record)
        self.assert_group_gone(task)
        print("killed_task", task["id"], record["status"], flush=True)

    def test_4_cold_backend_and_stream_recovery(self):
        """SIGKILL 后端后冷启动，真实流式请求恢复为 interrupted，独立任务跨后端重启存活。"""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        stream_started = __import__("threading").Event()
        request_id = {}

        def stream_request():
            """发送真实 QA 并在收到 request SSE 后通知主线程，此时服务已保存请求记录。"""
            try:
                with httpx.Client(timeout=120, trust_env=False) as client:
                    with client.stream(
                        "POST",
                        self.url + "/api/qa",
                        json={
                            "question": "Does DdrB participate in DNA repair?",
                            "workspace": self.workspace,
                            "tier": "small",
                        },
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line.startswith("data: ") and not request_id:
                                data = json.loads(line[6:])
                                if "id" in data:
                                    request_id["id"] = data["id"]
                                    stream_started.set()
            except httpx.TransportError:
                return

        future = pool.submit(stream_request)
        self.assertTrue(stream_started.wait(timeout=60))
        health = self.client.get("/api/health").raise_for_status().json()
        backend = psutil.Process(health["backend"]["pid"])
        self.assertIn("memmgr.web:app", backend.cmdline())
        backend.send_signal(signal.SIGKILL)
        future.result(timeout=30)
        pool.shutdown(wait=True)
        self.start_backend()
        record = (
            self.client.get("/api/requests/" + request_id["id"])
            .raise_for_status()
            .json()
        )
        self.assertEqual(record["status"], "interrupted", record)
        print("cold_qa", record["id"], record["status"], flush=True)
        task = self.create()
        self.wait_state(task, child=True)
        type(self).restarted_backend.kill()
        type(self).restarted_backend.wait(timeout=10)
        type(self).restarted_backend = None
        self.start_backend()
        record = self.client.get("/api/tasks/" + task["id"]).raise_for_status().json()
        self.assertIn(record["status"], ("running", "succeeded"), record)
        if record["status"] == "running":
            self.client.post("/api/tasks/" + task["id"] + "/cancel").raise_for_status()
            record = self.wait_state(task, terminal=True)
        self.assert_group_gone(task)
        print("cold_task_survival", task["id"], record["status"], flush=True)

    def test_5_rag_budget_failure_stops_batch(self):
        """以合法128 token预算触发真实问题超预算，验证原错误、停止后续repeat及整组退出。"""
        import copy

        saved = self.client.get("/api/config").raise_for_status().json()
        changed = copy.deepcopy(saved)
        changed["baseline"]["experiment"].update(
            context_tokens=128, queries=8, repeats=2, warmups=0
        )
        root = (ROOT / ".agent/web/workspaces" / self.workspace).resolve()
        if not root.is_relative_to((ROOT / ".agent/web/workspaces").resolve()):
            raise ValueError("Test workspace escapes isolated root")
        queries_path = root / "small/queries.jsonl"
        original_queries = queries_path.read_bytes()
        queries = [json.loads(line) for line in original_queries.splitlines()]
        with (root / "small/corpus.jsonl").open() as corpus:
            source_document = json.loads(next(corpus))
        long_question = source_document["title"] + "\n" + source_document["text"]
        sys.path.insert(0, str(ROOT))
        from transformers import AutoTokenizer

        from memmgr.models import local_snapshot
        from memmgr.prompt import assemble_prompt

        tokenizer = AutoTokenizer.from_pretrained(
            local_snapshot(
                saved["baseline"]["llm"]["model"],
                saved["baseline"]["llm"]["revision"],
            ),
            local_files_only=True,
        )
        with self.assertRaisesRegex(
            ValueError, "Even the empty-evidence prompt does not fit"
        ):
            assemble_prompt(tokenizer, long_question, [], 128)
        try:
            queries[0]["text"] = long_question
            queries_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queries),
                encoding="utf-8",
            )
            self.client.put("/api/config", json=changed).raise_for_status()
            task = (
                self.client.post(
                    "/api/tasks",
                    json={
                        "kind": "rag",
                        "workspace": self.workspace,
                        "tier": "small",
                        "concurrency": 1,
                    },
                )
                .raise_for_status()
                .json()
            )
            self.created.append(task)
            record = self.wait_state(task, terminal=True)
            self.assertEqual(record["status"], "failed", record)
            self.assertIn(
                "Even the empty-evidence prompt does not fit", record["error"]
            )
            runs = list(Path(task["output"]).glob("*"))
            self.assertEqual(len(runs), 1)
            rows = [
                json.loads(line)
                for line in (runs[0] / "requests.jsonl").read_text().splitlines()
            ]
            failures = [row for row in rows if row["status"] == "failed"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual({row["repeat"] for row in rows}, {1})
            self.assertFalse((runs[0] / "repeat-2.json").exists())
            self.assert_group_gone(task)
            before = {
                path.name: path.stat().st_size for path in runs[0].glob("*.jsonl")
            }
            time.sleep(1)
            self.assertEqual(
                before,
                {path.name: path.stat().st_size for path in runs[0].glob("*.jsonl")},
            )
            print(
                "rag_fail_fast",
                task["id"],
                record["status"],
                failures[0]["error"],
                flush=True,
            )
        finally:
            queries_path.write_bytes(original_queries)
            self.client.put("/api/config", json=saved).raise_for_status()

    def test_6_baseline_first_index_error_stops_plan(self):
        """损坏自有真实索引清单身份，验证完整baseline首个index失败即停止、保留原错误。"""
        workspace = next(
            row
            for row in self.client.get("/api/workspaces")
            .raise_for_status()
            .json()["items"]
            if row["name"] == self.workspace
        )
        name = workspace["tiers"]["small"]["collection"]
        manifest_path = Path(workspace["path"]) / "index-manifests" / f"{name}.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        try:
            manifest["identity"]["corpus_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            task = (
                self.client.post(
                    "/api/tasks", json={"kind": "baseline", "workspace": self.workspace}
                )
                .raise_for_status()
                .json()
            )
            self.created.append(task)
            record = self.wait_state(task, terminal=True)
            self.assertEqual(record["status"], "failed", record)
            self.assertIn("Existing index manifest identity differs", record["error"])
            commands_path = next(
                (Path(task["output"]).parent / "execution").glob("*/commands.json")
            )
            commands = json.loads(commands_path.read_text())
            self.assertEqual(
                [command["label"] for command in commands], ["index-small"]
            )
            self.assertNotEqual(commands[0]["returncode"], 0)
            self.assert_group_gone(task)
            print(
                "baseline_first_error",
                task["id"],
                record["status"],
                commands[0]["returncode"],
                flush=True,
            )
        finally:
            manifest_path.write_bytes(original)

    def start_backend(self):
        """启动测试所有的真实后端子进程，并等待实际 health HTTP 成功。"""
        log_path = ROOT / ".agent/web/lifecycle-backend.log"
        with log_path.open("ab") as log:
            type(self).restarted_backend = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "memmgr.web:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.restarted_backend.poll() is not None:
                self.fail(f"Actual backend exited: {log_path}")
            try:
                if self.client.get("/api/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.2)
        self.fail("Actual cold backend did not become ready")


if __name__ == "__main__":
    unittest.main(verbosity=2)
