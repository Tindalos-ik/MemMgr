"""共享请求批次的真实失败持久化与受控取消，保留普通 CLI 的原有测量语义。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


def append_request(output_directory, record):
    """将 record 原始请求写入 output_directory；失败时同步保存同一真实错误行。"""
    encoded = json.dumps(record, ensure_ascii=False) + "\n"
    with (Path(output_directory) / "requests.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(encoded)
    if record["status"] == "failed":
        with (Path(output_directory) / "failures.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(encoded)


async def gather_requests(requests, fail_fast=False):
    """协调 requests 协程；fail_fast 时首错取消并等待全部在途协程结束，再传播原异常。"""
    if not fail_fast:
        return await asyncio.gather(*requests)
    tasks = [asyncio.create_task(request) for request in requests]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
