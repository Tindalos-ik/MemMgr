"""仅跟随已配置 Chroma 容器的真实 Docker 日志，提供可轮换的有界本地日志文件。"""

from __future__ import annotations
import subprocess
import re
import threading
from pathlib import Path
from .config import STATE, save_json, read_json


def docker_timestamp(line):
    """仅提取 line 中真实 Docker RFC3339 日志前缀；诊断正文不会成为 --since 参数。"""
    from datetime import datetime

    token = line.split(b" ", 1)[0]
    if not re.fullmatch(
        rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})",
        token,
    ):
        return None
    value = token.decode("ascii")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


class ChromaLogs:
    """管理专有 Docker logs 进程及后台读流，container 为允许的已配置容器名。"""

    def __init__(self, container):
        """为 container 构建受约束 argv；禁止 shell 与任意其他 Docker 操作。"""
        self.path = STATE / "chroma.log"
        self.status = STATE / "chroma-log-status.json"
        import psutil
        from .tasks import process_identity

        if self.status.exists():
            previous = read_json(self.status)
            if previous.get("pid") and previous.get("process_identity"):
                try:
                    old = psutil.Process(previous["pid"])
                    command = old.cmdline()
                    if (
                        process_identity(old.pid) == previous["process_identity"]
                        and command == previous["argv"]
                    ):
                        old.terminate()
                        old.wait(timeout=5)
                except (psutil.NoSuchProcess, FileNotFoundError):
                    pass
        self.stop_requested = False
        argv = ["docker", "logs", "--follow", "--timestamps"]
        self.last_line = None
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as stream:
                stream.seek(max(0, self.path.stat().st_size - 65536))
                lines = stream.read().splitlines()
            self.last_line = next(
                (
                    line
                    for line in reversed(lines)
                    if docker_timestamp(line) is not None
                ),
                None,
            )
        if self.last_line is not None:
            argv += ["--since", docker_timestamp(self.last_line)]
        else:
            argv += ["--tail", "500"]
        argv.append(container)
        self.process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        save_json(
            self.status,
            {
                "available": True,
                "container": container,
                "argv": argv,
                "pid": self.process.pid,
                "process_identity": process_identity(self.process.pid),
            },
        )
        self.thread = threading.Thread(target=self.follow, daemon=True)
        self.thread.start()

    def follow(self):
        """把真实进程输出逐行追加至受限文件；重启时只排除已持久化的边界重复行。"""
        with self.path.open("ab") as stream:
            boundary = self.last_line
            for line in iter(self.process.stdout.readline, b""):
                if boundary is not None and line.rstrip(b"\n") == boundary:
                    boundary = None
                    continue
                if stream.tell() >= 2 * 1024 * 1024:
                    stream.close()
                    self.path.replace(self.path.with_suffix(".log.1"))
                    stream = self.path.open("ab")
                stream.write(line)
                stream.flush()
            stream.close()
        code = self.process.wait()
        if not self.stop_requested:
            save_json(
                self.status,
                {
                    "available": False,
                    "reason": f"docker logs exited with code {code}; actual error is retained in chroma.log",
                    "returncode": code,
                },
            )

    def close(self):
        """终止本实例启动的日志读取进程并等待线程退出，保留真实服务运行。"""
        self.stop_requested = True
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.thread.join(timeout=5)
