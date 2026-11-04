"""读取 OpenAI 兼容的流式响应，保留正文、结束原因和服务端 token 用量。

本模块只处理一次已获应用接纳的生成请求；应用排队、检索与结果落盘由批量测量模块负责。
"""

from __future__ import annotations

import json
import time


async def stream_completion(http, endpoint, payload, admitted_start, on_delta=None):
    """发送一次流式请求，返回正文、两类首次内容等待、结束信息和生成耗时。

    http 为已配置连接数与超时的 AsyncClient；endpoint 为 chat/completions 地址，
    payload 保留模型、采样参数和 include_usage。admitted_start 是应用接纳请求的
    单调时钟，用于计算含检索与提示词准备的首次内容等待。
    on_delta 为可选异步正文回调，接收本次真实正文块；默认保持批量调用行为。

    返回 (answer, e2e_ttft, model_ttft, finish_reason, usage, chunks, model_total)。
    首个非空 content delta 才触发计时，空角色包和用量包只更新相应状态；每个内容块
    可以包含多个 token。缺失用量返回 None，由调用者决定 usage_missing 状态。
    HTTP、JSON 或传输异常直接传播给单请求边界记录。
    """
    answer = ""
    e2e_ttft = None
    model_ttft = None
    finish_reason = None
    usage = None
    content_chunks = 0
    model_started = time.perf_counter()
    stream_started = model_started

    async with http.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw_event = line[6:]
            if raw_event == "[DONE]":
                break
            event = json.loads(raw_event)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content", "")
                if content:
                    # 首次到达正文时分别度量应用接纳后等待与生成接口内等待。
                    if e2e_ttft is None:
                        e2e_ttft = time.perf_counter() - admitted_start
                        model_ttft = time.perf_counter() - stream_started
                    answer += content
                    content_chunks += 1
                    if on_delta is not None:
                        await on_delta(content)
                finish_reason = choice.get("finish_reason") or finish_reason

    model_total = time.perf_counter() - model_started
    return (
        answer,
        e2e_ttft,
        model_ttft,
        finish_reason,
        usage,
        content_chunks,
        model_total,
    )
