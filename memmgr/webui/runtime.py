"""采集实际 vLLM 进程参数和模型接口，阻止未重启服务与已保存配置混用。"""

from __future__ import annotations

import json
import time

import httpx
import psutil

from ..inference.models import local_snapshot


def capture_runtime(snapshot, require_llm=False):
    """读取 snapshot 所选本机服务的实际参数；require_llm 为真要求服务与固定配置一致。"""
    server = snapshot["server"]
    candidates = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        command = process.info.get("cmdline") or []
        if "vllm.entrypoints.openai.api_server" not in command:
            continue
        options = {}
        for i, arg in enumerate(command):
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    options[key] = value
                else:
                    options[arg] = (
                        command[i + 1]
                        if i + 1 < len(command) and not command[i + 1].startswith("--")
                        else True
                    )
        if str(options.get("--port", "8000")) != str(server["port"]):
            continue
        environment = process.environ()
        candidates.append(
            {
                "pid": process.pid,
                "created_at": process.info["create_time"],
                "argv": command,
                "options": options,
                "environment": {
                    key: environment.get(key)
                    for key in (
                        "VLLM_USE_V2_MODEL_RUNNER",
                        "VLLM_USE_FLASHINFER_SAMPLER",
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                        "VLLM_BATCH_INVARIANT",
                    )
                },
            }
        )
    runtime = {
        "captured_at": time.time(),
        "processes": candidates,
        "models": None,
        "error": None,
        "configuration_matches": False,
        "differences": [],
    }
    try:
        response = httpx.get(
            snapshot["baseline"]["llm"]["base_url"] + "/models",
            timeout=5,
            trust_env=False,
        )
        response.raise_for_status()
        runtime["models"] = response.json()
    except (httpx.HTTPError, OSError) as exc:
        runtime["error"] = f"{type(exc).__name__}: {exc}"
        if require_llm:
            raise RuntimeError(runtime["error"]) from exc
        return runtime
    if len(candidates) != 1:
        runtime["differences"].append(
            "Cannot identify exactly one local vLLM server process"
        )
    else:
        actual = candidates[0]
        options = actual["options"]
        for key, flag in {
            "host": "--host",
            "port": "--port",
            "max_model_len": "--max-model-len",
            "max_num_seqs": "--max-num-seqs",
            "gpu_memory_utilization": "--gpu-memory-utilization",
        }.items():
            observed = options.get(flag)
            expected = server[key]
            equal = (
                float(observed) == float(expected)
                if isinstance(expected, (int, float)) and observed is not None
                else observed == expected
            )
            if not equal:
                runtime["differences"].append(
                    {"field": key, "configured": expected, "actual": observed}
                )
        dtype = {"half": "float16", "float": "float32"}.get(
            options.get("--dtype"), options.get("--dtype")
        )
        if dtype != server["dtype"]:
            runtime["differences"].append(
                {"field": "dtype", "configured": server["dtype"], "actual": dtype}
            )
        if bool(options.get("--enforce-eager", False)) != server["enforce_eager"]:
            runtime["differences"].append(
                {
                    "field": "enforce_eager",
                    "configured": server["enforce_eager"],
                    "actual": bool(options.get("--enforce-eager", False)),
                }
            )
        model = options.get("--model")
        from ..short_term.config import KVConfig, transfer_config
        from .config import ROOT

        kv = KVConfig.model_validate(server.get("kv_cache", {}))
        expected_transfer = transfer_config(server, ROOT)
        actual_transfer = options.get("--kv-transfer-config")
        if (json.loads(actual_transfer) if actual_transfer else None) != expected_transfer:
            runtime["differences"].append("KV transfer configuration differs")
        expected_gpu_bytes = str(kv.gpu_bytes) if kv.gpu_bytes else None
        if options.get("--kv-cache-memory-bytes") != expected_gpu_bytes:
            runtime["differences"].append("GPU KV memory budget differs")
        if expected_transfer is not None and (
            options.get("--block-size") != str(kv.block_tokens)
            or "--enable-prefix-caching" not in options
            or psutil.Process(actual["pid"]).environ().get("VLLM_KV_CACHE_LAYOUT") != "LBHNC"
        ):
            runtime["differences"].append("KV block size, prefix caching or layout differs")
        configured = snapshot["models"][server["model"]]
        expected_snapshot = str(local_snapshot(server["model"], configured["revision"]))
        if model == server["model"]:
            if options.get("--revision") != configured["revision"]:
                runtime["differences"].append("Actual model revision differs")
        elif model != expected_snapshot:
            runtime["differences"].append(
                {
                    "field": "model",
                    "configured": expected_snapshot,
                    "actual": model,
                }
            )
        for key, value in actual["environment"].items():
            expected = ("1" if kv.batch_invariant else "0") if key == "VLLM_BATCH_INVARIANT" else server[key]
            if value != expected:
                runtime["differences"].append(
                    {"field": key, "configured": expected, "actual": value}
                )
    advertised = runtime["models"].get("data", [])
    if not any(
        model["id"] == server["model"]
        and model.get("max_model_len") == server["max_model_len"]
        for model in advertised
    ):
        runtime["differences"].append(
            "Advertised model name/context differs from saved configuration"
        )
    runtime["configuration_matches"] = not runtime["differences"]
    if require_llm and runtime["differences"]:
        raise ValueError(
            f"Saved server configuration requires vLLM restart before generation: {runtime['differences']}"
        )
    return runtime
