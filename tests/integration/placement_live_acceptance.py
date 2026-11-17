"""使用真实 GPU 和 WebAPI 验收联合放置配置、问答服务与访问边界。

独占运行；--server 指向第 06 组生成的 placement 服务配置。
脚本只修改临时项目副本中的配置，并在退出时恢复原始内容。
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_raa/06_kv_placement"))

from run_placement import request, save
from run_ssd import stop_service


def wait_ready(process, port, endpoint):
    """等待自有 process 在 port 提供真实 endpoint，提前退出则立即报错。"""
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned service exited before readiness")
        with socket.socket() as probe:
            listening = probe.connect_ex(("127.0.0.1", port)) == 0
        if listening:
            with httpx.Client(trust_env=False, timeout=60) as client:
                client.get(f"http://127.0.0.1:{port}{endpoint}").raise_for_status()
            return
        time.sleep(.5)
    raise TimeoutError("Owned service readiness timed out")


def main():
    """启动 --server 的真实独占服务，在 --output 保存验收并恢复临时配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--web-port", type=int, default=18765)
    parser.add_argument("--chroma-port", type=int, choices=(8001,), help="使用项目固定端口的已有 small 索引检查真实 CLI 与 Web 检索问答")
    args = parser.parse_args()
    for name in ("server", "runtime", "prompts", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output.is_relative_to(ROOT / ".agent"):
        raise ValueError("Acceptance output must stay in the project .agent directory")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(args.runtime / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"), TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"), PYTHONHASHSEED="0",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    originals = {name: (ROOT / f"config/{name}.json").read_bytes() for name in ("baseline", "server")}
    server = json.loads(args.server.read_text())
    if server["kv_cache"]["policy"] != "placement":
        raise ValueError("Acceptance requires the real placement policy")
    server["port"] = args.port
    server["kv_cache"].update(ssd_root=str(args.output / "ssd-cache"), audit_path=str(args.output / "audit.jsonl"))
    baseline = json.loads(originals["baseline"])
    baseline["llm"]["base_url"] = f"http://127.0.0.1:{args.port}/v1"
    if args.chroma_port is not None:
        baseline["chroma"]["port"] = args.chroma_port
    processes = []
    logs = []
    try:
        save(ROOT / "config/server.json", server)
        save(ROOT / "config/baseline.json", baseline)
        for port in (args.port, args.web_port):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        commands = [([sys.executable, "-m", "memmgr.inference.server"], args.port, "/health", "vllm.log"),
                    ([sys.executable, "-m", "uvicorn", "memmgr.webui.app:app", "--host", "127.0.0.1",
                      "--port", str(args.web_port)], args.web_port, "/api/config", "web.log")]
        for command, port, endpoint, log_name in commands:
            log = (args.output / log_name).open("x")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            wait_ready(process, port, endpoint)
        from memmgr.webui.runtime import capture_runtime

        snapshot = {"baseline": baseline, "server": server,
                    "models": json.loads((ROOT / "config/models.json").read_text())}
        runtime = capture_runtime(snapshot, require_llm=True)
        save(args.output / "runtime.json", runtime)
        from memmgr.short_term.placement import PlacementCosts

        profile_path = Path(server["kv_cache"]["placement_profile"])
        profile = json.loads(profile_path.read_text())
        costs = PlacementCosts(profile_path, profile["context"])
        try:
            costs.seconds("cpu_read", costs.max_blocks + 1)
        except ValueError:
            pass
        else:
            raise ValueError("Uncalibrated prefix length was accepted")
        changed = dict(profile["context"], cpu_blocks=profile["context"]["cpu_blocks"] + 1)
        try:
            PlacementCosts(profile_path, changed)
        except ValueError:
            pass
        else:
            raise ValueError("Mismatched real runtime identity was accepted")
        inputs = json.loads(args.prompts.read_text())
        target = next(row["prompt_token_ids"] for row in inputs if len(row["prompt_token_ids"]) == 1536)
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", trust_env=False, timeout=180) as client:
            completions = [request(client, target, f"acceptance-{index}", 32) for index in range(3)]
        if any(row["generated_tokens"] != completions[0]["generated_tokens"] for row in completions):
            raise ValueError("Repeated real inference output differs")
        save(args.output / "completions.json", completions)
        with httpx.Client(base_url=f"http://127.0.0.1:{args.web_port}", trust_env=False, timeout=60) as client:
            schema = client.get("/api/config/schema").raise_for_status().json()
            if schema["values"]["server"]["kv_cache"]["policy"] != "placement":
                raise ValueError("Web configuration did not expose the actual placement policy")
            saved = client.put("/api/config", json=schema["values"])
            (args.output / "configuration-response.json").write_text(saved.text)
            saved.raise_for_status()
            metrics = client.get("/api/kv").raise_for_status().json()
            save(args.output / "kv.json", metrics)
            if not metrics["metrics"] or metrics["configured"]["policy"] != "placement":
                raise ValueError("Web did not return real engine KV metrics")
            if args.chroma_port is not None:
                query = inputs[0]["messages"][1]["content"].split("\nEvidence:")[0].removeprefix("Question: ")
                cli = json.loads(json.dumps(baseline))
                cli["experiment"]["output_dir"] = str(args.output / "cli")
                save(args.output / "cli-config.json", cli)
                completed = subprocess.run([sys.executable, "-m", "memmgr", "--config", str(args.output / "cli-config.json"),
                                            "ask", "--tier", "small", "--query", query], cwd=ROOT,
                                           capture_output=True, text=True, timeout=240)
                (args.output / "cli.log").write_text(completed.stdout + completed.stderr)
                completed.check_returncode()
                records = [json.loads(line) for path in (args.output / "cli").glob("*/requests.jsonl")
                           for line in path.read_text().splitlines()]
                if len(records) != 1 or not records[0]["request_success"] or not records[0]["answer"]:
                    raise ValueError("Real CLI retrieval answer failed")
                response = client.post("/api/qa", json=dict(workspace="formal", tier="small", question=query, context_tokens=1024))
                response.raise_for_status()
                (args.output / "web-qa.sse").write_text(response.text)
                if "event: done" not in response.text or "event: error" in response.text:
                    raise ValueError("Real Web retrieval answer failed")
            rejected = []
            for field, value in (("placement_profile", "../../outside.json"), ("tier", "cpu"), ("policy", "unknown")):
                invalid = json.loads(json.dumps(schema["values"]))
                invalid["server"]["kv_cache"][field] = value
                response = client.put("/api/config", json=invalid)
                if response.status_code != 422:
                    raise ValueError(f"Web accepted an invalid {field}")
                rejected.append(field)
            save(args.output / "summary.json", dict(runtime_match=runtime["configuration_matches"],
                 real_completions=len(completions), output_equal=True, configuration_roundtrip=True,
                 engine_metrics=True, uncalibrated_length_rejected=True,
                 mismatched_cost_identity_rejected=True, cli_and_web_rag=args.chroma_port is not None, rejected=rejected))
    finally:
        for process in reversed(processes):
            stop_service(process)
        for log in logs:
            log.close()
        for name, value in originals.items():
            (ROOT / f"config/{name}.json").write_bytes(value)


if __name__ == "__main__":
    main()
