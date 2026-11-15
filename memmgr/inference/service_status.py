"""共享本机 vLLM、Chroma 真实健康请求，逐服务保留不可达原因。"""

import asyncio

import httpx


async def health_snapshot(config, names=("vllm", "chroma")):
    """按 config 中地址查询 names 指定的健康接口，返回逐服务 HTTP 状态及原始错误。"""
    urls = {
        "vllm": config["llm"]["base_url"].removesuffix("/v1") + "/health",
        "chroma": f"http://{config['chroma']['host']}:{config['chroma']['port']}/api/v2/heartbeat",
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(5, connect=1), trust_env=False
    ) as client:
        async def probe(name):
            """请求 name 的真实健康接口，返回服务名与健康状态或原始错误。"""
            url = urls[name]
            try:
                response = await client.get(url)
                response.raise_for_status()
                return name, {
                    "status": "ok",
                    "url": url,
                    "http_status": response.status_code,
                }
            except (httpx.HTTPError, OSError) as exc:
                return name, {
                    "status": "unavailable",
                    "url": url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return dict(await asyncio.gather(*(probe(name) for name in names)))
