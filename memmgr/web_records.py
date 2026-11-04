"""只读取完整写入的真实 JSONL 行，并显式报告尚未提交的末行。"""

from pathlib import Path
import json


def complete_rows(path):
    """按 path 顺序产出以换行结束的 JSON 记录；未完成末行留待下一次读取。"""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                return
            yield json.loads(line)


def incomplete_tail(path):
    """检查 path 的真实最后一个字节，返回是否存在尚未结束的 JSONL 末行。"""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return False
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        return stream.read(1) != b"\n"
