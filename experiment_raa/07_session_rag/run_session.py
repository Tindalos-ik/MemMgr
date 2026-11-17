"""经正式 WebAPI 对照多轮 RAG 的三级 KV 策略，并集中验收真实生命周期。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_raa/06_kv_placement"))
sys.path.insert(0, str(ROOT / "tests/integration"))

from run_placement import calibration_plan, calibrate, counters, digest, run_service, save
from run_ssd import monitor_io, stop_service
from review_placement import audit
from memmgr.short_term.placement import PlacementCosts
from memmgr.short_term.reuse import Predictor
from session_live_acceptance import acceptance, completed, idle


def launch(command, port, endpoint, log):
    """启动自有 command，以 port/endpoint 等待真实就绪，标准输出写入 log。"""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 600
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Owned service exited: {command}")
                with socket.socket() as probe:
                    listening = probe.connect_ex(("127.0.0.1", port)) == 0
                if listening:
                    response = client.get(endpoint)
                    if response.status_code == 200:
                        return process
                    if response.status_code != 503:
                        response.raise_for_status()
                time.sleep(.5)
        raise TimeoutError(f"Service readiness timed out: {command}")
    except BaseException:
        stop_service(process)
        raise


def plan(pool):
    """从 pool 的独立评价问题构造两会话、四轮、三种证据变化的固定顺序。"""
    ids = list(dict.fromkeys(row["query_id"] for row in pool))
    queries = {row["query_id"]: row["messages"][1]["content"].split("\nEvidence:")[0].removeprefix("Question: ")
               for row in pool}
    selected = ids[len(ids) // 2:]
    result = []
    for scenario in ("same", "partial", "changed"):
        for turn in range(4):
            for conversation in range(2):
                query_id = selected[(conversation + (2 * turn if scenario == "changed" else 0)) % len(selected)]
                query = queries[query_id]
                body = dict(workspace="formal", tier="small", query=query,
                            question=f"What does the supplied evidence establish about this claim: {query} Cite the supporting document.",
                            context_tokens=1536, top_k=1 if scenario == "partial" and turn % 2 else 3)
                result.append(dict(index=len(result), scenario=scenario, conversation=conversation,
                                   turn=turn, query_id=query_id, body=body))
    return result


def run_mode(args, name, mode, trace, verify=False):
    """按 args/name 启动 mode，回放 trace；verify 指定额外生命周期与重算检查。"""
    output = args.output / name
    output.mkdir()
    server = json.loads((ROOT / "config/server.json").read_text())
    server.update(port=args.port, gpu_memory_utilization=.65)
    server["kv_cache"] = dict(tier="ssd", policy=mode, batch_invariant=True,
                              gpu_bytes=58 * 2**20, cpu_bytes=128 * 2**20, ssd_bytes=224 * 2**20,
                              ssd_root=str(output / "cache"), audit_path=str(output / "audit.jsonl"),
                              model_dir=str(args.output / "model") if mode != "lru" else "",
                              placement_profile=str(args.output / "costs.json") if mode == "placement" else "")
    baseline = json.loads((ROOT / "config/baseline.json").read_text())
    baseline["llm"].update(base_url=f"http://127.0.0.1:{args.port}/v1", max_tokens=96)
    save(ROOT / "config/server.json", server)
    save(ROOT / "config/baseline.json", baseline)
    save(output / "server.json", server)
    save(output / "baseline.json", baseline)
    processes, logs, records = [], [], []
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    sampler = None
    try:
        commands = [([sys.executable, "-m", "memmgr.inference.server"], args.port, "/health", "vllm.log"),
                    ([sys.executable, "-m", "uvicorn", "memmgr.webui.app:app", "--host", "127.0.0.1",
                      "--port", str(args.web_port)], args.web_port, "/api/config", "web.log")]
        for command, port, endpoint, filename in commands:
            log = (output / filename).open("x")
            logs.append(log)
            processes.append(launch(command, port, endpoint, log))
        sampler = pool.submit(monitor_io, processes[0].pid, stop, output / "resources.jsonl")
        with httpx.Client(base_url=f"http://127.0.0.1:{args.web_port}", timeout=240, trust_env=False) as client, \
                httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=180, trust_env=False) as engine:
            # 各组使用相同输入预热；验收请求放在评价之后，保持策略初态可比较。
            warm = completed(client, dict(trace[0]["body"], question="What does the current evidence support? Cite a source."))
            save(output / "warmup.json", warm)
            raw = idle(engine, output / "initial.prom")
            before = counters(raw)
            active = {}
            with (output / "requests.jsonl").open("x") as stream:
                for row in trace:
                    key = (row["scenario"], row["conversation"])
                    if key not in active:
                        active[key] = client.post("/api/sessions", json=dict(workspace="formal", tier="small")) \
                            .raise_for_status().json()["id"]
                    body = dict(row["body"], session_id=active[key])
                    result = completed(client, body)
                    raw = idle(engine, output / f"{row['index']:04}.prom")
                    after = counters(raw)
                    delta = {key: after[key] - before.get(key, 0) for key in after}
                    record = dict(row, response=result["record"], client_ttft_seconds=result["client_ttft_seconds"],
                                  metrics=delta)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    records.append(record)
                    before = after
                    if row["index"] % 8 == 7:
                        print(json.dumps(dict(run=name, completed=row["index"] + 1, total=len(trace))), flush=True)
            for session_id in active.values():
                client.delete(f"/api/sessions/{session_id}").raise_for_status()
            idle(engine, output / "final.prom")
            if verify:
                verified = acceptance(client, engine, output / "acceptance")
                stop_service(processes.pop())
                log = (output / "web-restart.log").open("x")
                logs.append(log)
                command, port, endpoint, _ = commands[1]
                processes.append(launch(command, port, endpoint, log))
                session_id = verified["resume_session_id"]
                restored = client.get(f"/api/sessions/{session_id}").raise_for_status().json()
                if restored["turns"] != verified["resume_turns"]:
                    raise ValueError("Backend restart lost completed session history")
                followup = completed(client, dict(trace[0]["body"], session_id=session_id))
                if not followup["record"]["prompt"]["history_request_ids"]:
                    raise ValueError("Restarted backend did not consume saved history")
                save(output / "acceptance/restarted.json", followup)
                verified["backend_restart_history"] = True
                save(output / "acceptance/summary.json", verified)
                client.delete(f"/api/sessions/{session_id}").raise_for_status()
                recomputations = []
                for index in (0, 12, 20):
                    actual = records[index]["response"]
                    idle(engine)
                    previous = counters(engine.get("/metrics").raise_for_status().text)
                    response = engine.post("/v1/chat/completions", json=dict(
                        model=server["model"], messages=actual["prompt"]["messages"], temperature=0, seed=7,
                        max_tokens=96, cache_salt=f"recompute-{name}-{index}"))
                    response.raise_for_status()
                    recomputed = response.json()
                    current = counters(idle(engine))
                    computed = current["vllm:request_prefill_kv_computed_tokens_sum"] - previous["vllm:request_prefill_kv_computed_tokens_sum"]
                    observation = dict(index=index, computed_tokens=computed, response=recomputed)
                    save(output / f"acceptance/recomputed-{index}.json", observation)
                    if (computed != actual["usage"]["prompt_tokens"]
                            or recomputed["choices"][0]["message"]["content"] != actual["answer"]
                            or any(recomputed["usage"][key] != actual["usage"][key]
                                   for key in ("prompt_tokens", "completion_tokens", "total_tokens"))):
                        raise ValueError("Actual full recomputation differs from the cached session answer")
                    recomputations.append(observation)
                save(output / "acceptance/recomputed.json", recomputations)
            idle(engine)
    finally:
        stop.set()
        try:
            if sampler is not None:
                sampler.result(timeout=15)
        finally:
            pool.shutdown(wait=True)
            for process in reversed(processes):
                stop_service(process)
            for log in logs:
                log.close()


def review(root):
    """从 root 原始响应、指标和真实块事件复核同输入、来源、容量及策略时延。"""
    profile = json.loads((root / "costs.json").read_text())
    costs = PlacementCosts(root / "costs.json", profile["context"])
    metadata = json.loads((root / "model/model.json").read_text())
    predictor = Predictor(root / "model", metadata["identity"])
    groups, mismatches = [], []
    for epoch in range(2):
        reference = [json.loads(line) for line in (root / f"e{epoch}-lru/requests.jsonl").read_text().splitlines()]
        for mode in ("lru", "learned", "placement"):
            directory = root / f"e{epoch}-{mode}"
            records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
            if len(records) != 24:
                raise ValueError("Incomplete session experiment")
            before = counters((directory / "initial.prom").read_text())
            events = [json.loads(line) for line in (directory / "audit.jsonl").read_text().splitlines()]
            begins = [event for event in events if event["kind"] == "begin"]
            linkage = {}
            times, e2e, saved, computed, cited = [], [], [], 0, 0
            ids = set()
            for record, original in zip(records, reference):
                actual, expected = record["response"], original["response"]
                after = counters((directory / f"{record['index']:04}.prom").read_text())
                delta = {key: after[key] - before.get(key, 0) for key in after}
                if delta != record["metrics"] or delta["vllm:request_success_total"] != 1:
                    raise ValueError("Actual request boundaries or counters differ")
                token_count = actual["usage"]["prompt_tokens"]
                balanced = sum(delta.get(key, 0) for key in ("vllm:prefix_cache_hits_total",
                    "vllm:external_prefix_cache_hits_total", "vllm:request_prefill_kv_computed_tokens_sum"))
                if balanced != token_count or delta["vllm:num_preemptions_total"] != 0:
                    raise ValueError("Actual input tokens do not balance across cache and compute")
                if actual["prompt"]["messages"] != expected["prompt"]["messages"]:
                    raise ValueError("Policies received different conversation prompts")
                if actual["answer"] != expected["answer"] or actual["usage"] != expected["usage"]:
                    mismatches.append(dict(epoch=epoch, mode=mode, index=record["index"]))
                times.append(actual["model_ttft_seconds"] * 1000)
                e2e.append(actual["e2e_ttft_seconds"] * 1000)
                saved.append((expected["model_ttft_seconds"] - actual["model_ttft_seconds"]) * 1000)
                computed += delta["vllm:request_prefill_kv_computed_tokens_sum"]
                cited += actual["citations"]["has_citations"]
                matches = [event for event in begins if event["request_id"].startswith("chatcmpl-" + actual["id"] + "-")]
                if len(matches) != 1 or matches[0]["prompt_sha256"] != actual["prompt"]["input_sha256"]:
                    raise ValueError("Engine request identity or actual prompt hash differs")
                linkage[actual["id"]] = matches[0]["request_id"]
                ids.add(matches[0]["request_id"])
                before = after
            # 完整块复核使用评价期间的独立审计前缀，避免验收额外请求进入本组统计。
            last = next(index for index, event in enumerate(events)
                        if event["kind"] == "request_done" and event["request_id"] == linkage[records[-1]["response"]["id"]])
            save(directory / "request-linkage.json", linkage)
            end = next(index for index in range(last, len(events)) if events[index]["kind"] == "residency")
            scoped = directory / "review-scope"
            scoped.mkdir(exist_ok=True)
            (scoped / "audit.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events[:end + 1]))
            try:
                checked = audit(scoped, costs, predictor, ids)
            finally:
                (scoped / "audit.jsonl").unlink()
                scoped.rmdir()
            groups.append(dict(epoch=epoch, mode=mode, requests=len(records), citations_present=cited,
                               model_ttft_median_ms=statistics.median(times),
                               model_ttft_p95_ms=sorted(times)[math.ceil(len(times) * .95) - 1],
                               e2e_ttft_median_ms=statistics.median(e2e),
                               paired_model_saved_median_ms=statistics.median(saved), computed_tokens=computed, **checked))
    if mismatches:
        raise ValueError(f"Real policy outputs differ: {mismatches}")
    summary = dict(groups=groups, output_mismatches=mismatches, requests=sum(row["requests"] for row in groups),
                   model_sha256=predictor.sha256, costs_sha256=costs.sha256)
    save(root / "summary.json", summary)
    return summary


def main():
    """按参数复用可信模型、采集真实成本，运行两轮策略对照并恢复临时配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--web-port", type=int, default=18765)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    for name in ("model", "runtime", "prompts", "predictor", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output.is_relative_to(ROOT / ".agent"):
        raise ValueError("Results must stay in the project .agent directory")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(args.runtime / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_CACHE_ROOT=str(args.runtime / "vllm-cache"), TRITON_CACHE_DIR=str(args.runtime / "triton-cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(args.runtime / "torch-cache"), PYTHONHASHSEED="0",
                      VLLM_LOG_STATS_INTERVAL="0.5", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    source = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "memmgr").rglob("*.py")}
    source.update({str(path.relative_to(ROOT)): digest(path)
                   for folder in ("05_kv_ssd_restore", "06_kv_placement", "07_session_rag")
                   for path in (ROOT / "experiment_raa" / folder).glob("*.py")})
    test_path = ROOT / "tests/integration/session_live_acceptance.py"
    source[str(test_path.relative_to(ROOT))] = digest(test_path)
    save(args.output / "environment.json", dict(arguments=vars(args), source=source,
         prompts_sha256=digest(args.prompts), predictor_source=str(args.predictor)))
    shutil.copytree(args.predictor, args.output / "model")
    pool = json.loads(args.prompts.read_text())
    collect = calibration_plan(pool)
    save(args.output / "collect-trace.json", collect)
    originals = {name: (ROOT / f"config/{name}.json").read_bytes() for name in ("server", "baseline")}
    try:
        run_service(args, "collect", collect, "lru")
        calibrate(args.output / "collect", args.output / "costs.json")
        trace = plan(pool)
        save(args.output / "trace.json", trace)
        for epoch, modes in enumerate((("lru", "learned", "placement"), ("placement", "learned", "lru"))):
            for mode in modes:
                run_mode(args, f"e{epoch}-{mode}", mode, trace, verify=epoch == 0 and mode == "placement")
        if any(digest(ROOT / name) != value for name, value in source.items()):
            raise ValueError("Source changed during the frozen experiment")
        print(json.dumps(review(args.output)), flush=True)
    finally:
        for name, raw in originals.items():
            (ROOT / f"config/{name}.json").write_bytes(raw)


if __name__ == "__main__":
    main()
