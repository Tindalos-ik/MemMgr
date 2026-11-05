"""共享本机 vLLM、Chroma 真实健康请求，逐服务保留不可达原因。"""

import httpx


async def health_snapshot(config, names=("vllm", "chroma")):
    """按 config 中地址查询 names 指定的健康接口，返回逐服务 HTTP 状态及原始错误。"""
    result = {}
    urls = {
        "vllm": config["llm"]["base_url"].removesuffix("/v1") + "/health",
        "chroma": f"http://{config['chroma']['host']}:{config['chroma']['port']}/api/v2/heartbeat",
    }
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        for name in names:
            url = urls[name]
            try:
                response = await client.get(url)
                response.raise_for_status()
                result[name] = {
                    "status": "ok",
                    "url": url,
                    "http_status": response.status_code,
                }
            except (httpx.HTTPError, OSError) as exc:
                result[name] = {
                    "status": "unavailable",
                    "url": url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return result
