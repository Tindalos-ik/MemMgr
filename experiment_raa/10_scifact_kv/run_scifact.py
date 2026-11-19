"""在独立 vLLM 上按正反策略顺序回放真实 SciFact RAG 输入。"""

import argparse
from collections import Counter
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

from memmgr.inference.models import local_snapshot
from memmgr.long_term.index import sha256_file
from memmgr.short_term.config import local_path, transfer_config

from .prepare_scifact import BLOCK, ROOT, read, replay_plan, save

joint = importlib.import_module("experiment_raa.06_kv_joint.run_joint")
metrics_06 = importlib.import_module("experiment_raa.06_kv_joint.review_joint").METRICS
POLICIES = ("lru", "admission", "learned")
ORDERS = {"forward": POLICIES, "reverse": POLICIES[::-1]}
MAX_LEN = 1088
MAX_OUTPUT = 64
SEED = 53
GPU_BYTES = 32 * 2**20
CPU_BYTES = 56 * 2**20
MODEL_HASHES = {
    "model.json": "4bdd0df8cb08f250a89c297b3f8469349a817799006a1bcf3d60217549e019e6",
    "model.joblib": "5c97e4045088bd5e47046f3282b31a2ee6bd10ce942e178bf75993a7efe60373",
}
TIMED_MODULE = "experiment_raa.10_scifact_kv.timed_connector"
SERVER_ENV = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                  VLLM_BATCH_INVARIANT="1", VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
                  VLLM_LOG_STATS_INTERVAL="0.5", VLLM_KV_CACHE_LAYOUT="LBHNC", VLLM_SERVER_DEV_MODE="1")


def run_config(inputs, model_dir):
    """核对 inputs 和 model_dir 的固定模型与阈值，记录回放所用环境。"""
    server = read(ROOT / "config/server.json")
    paths = [Path(__file__).with_name(name) for name in ("prepare_scifact.py", "run_scifact.py", "timed_connector.py")]
    packages = {name: version(name) for name in ("vllm", "torch", "transformers", "scikit-learn")}
    model_metadata = read(model_dir / "model.json")
    if (packages["vllm"] != "0.29.0" or model_metadata["threshold"] != 0.35
            or {name: sha256_file(model_dir / name) for name in MODEL_HASHES} != MODEL_HASHES):
        raise ValueError("Expected experiment 06's original frozen model and vLLM 0.29.0")
    llm = read(inputs / "inputs.json")["config"]["llm"]
    if (server["model"], server["revision"], server["dtype"]) != (llm["model"], llm["revision"], "float16"):
        raise ValueError("Retrieval chat tokenizer and replay model do not match")
    snapshot = local_snapshot(server["model"], server["revision"])
    model = read(snapshot / "config.json")
    block_bytes = (model["num_hidden_layers"] * 2 * model["num_key_value_heads"]
                   * (model["hidden_size"] // model["num_attention_heads"]) * 2 * BLOCK)
    return dict(inputs=str(inputs), inputs_sha256=sha256_file(inputs / "inputs.json"), model_dir=str(model_dir),
                model_hashes=MODEL_HASHES, model_threshold=model_metadata["threshold"], model_snapshot=str(snapshot),
                model_config_sha256=sha256_file(snapshot / "config.json"), server=server, seed=SEED,
                max_output=MAX_OUTPUT, max_len=MAX_LEN, packages=packages, block_bytes=block_bytes,
                gpu_bytes=GPU_BYTES, cpu_bytes=CPU_BYTES, orders={key: list(order) for key, order in ORDERS.items()},
                env=SERVER_ENV, source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in paths})


def server_command(config, directory, port, policy):
    """为 policy 从 config 构造本机 port 和 directory 的独立真实服务命令。"""
    options = dict(tier="cpu", policy=policy, block_tokens=BLOCK, batch_invariant=True, restore="always",
                   cpu_bytes=CPU_BYTES, gpu_bytes=GPU_BYTES, audit_path=str(directory / "audit.jsonl"))
    if policy != "lru":
        options["model_dir"] = config["model_dir"]
    server = dict(config["server"], max_model_len=MAX_LEN, max_num_seqs=1, kv_cache=options)
    transfer = transfer_config(server, ROOT)
    transfer.update(kv_connector="TimedConnector", kv_connector_module_path=TIMED_MODULE)
    transfer["kv_connector_extra_config"].update(spec_name="TimedCPUSpec", spec_module_path=TIMED_MODULE)
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", config["model_snapshot"],
            "--served-model-name", "scifact-replay", "--host", "127.0.0.1", "--port", str(port),
            "--dtype", server["dtype"], "--max-model-len", str(MAX_LEN), "--max-num-seqs", "1",
            "--block-size", str(BLOCK), "--kv-cache-memory-bytes", str(GPU_BYTES),
            "--gpu-memory-utilization", "0.8", "--enforce-eager", "--enable-prefix-caching",
            "--generation-config", "vllm", "--no-enable-log-requests", "--kv-transfer-config", json.dumps(transfer)]


def generate(client, token_ids, request_id):
    """发送项目模板 token_ids，用 request_id 关联审计并读出真实流式 token ID。"""
    payload = dict(model="scifact-replay", prompt=token_ids, max_tokens=MAX_OUTPUT, temperature=0,
                   seed=SEED, stream=True, request_id=request_id, logprobs=1,
                   return_tokens_as_token_ids=True, stream_options={"include_usage": True})
    generated, text = [], []
    first_token = first_text = finish = usage = None
    done = False
    started = time.perf_counter()
    with client.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            event = json.loads(line[6:])
            now = time.perf_counter() - started
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event["choices"]:
                finish = choice["finish_reason"] or finish
                content = choice["text"]
                if content and first_text is None:
                    first_text = now
                text.append(content)
                if choice["logprobs"] and (ids := choice["logprobs"]["tokens"]):
                    if first_token is None:
                        first_token = now
                    for token in ids:
                        if not token.startswith("token_id:") or not token[9:].isdigit():
                            raise ValueError(f"Expected a numeric output token ID: {token!r}")
                        generated.append(int(token[9:]))
    total = time.perf_counter() - started
    if (not done or first_token is None or finish not in ("length", "stop") or usage is None
            or usage["prompt_tokens"] != len(token_ids) or usage["completion_tokens"] != len(generated)
            or not 1 <= len(generated) <= MAX_OUTPUT or (finish == "length" and len(generated) != MAX_OUTPUT)):
        raise RuntimeError(f"Incomplete stream, usage or output token IDs: {request_id}, {finish}, {usage}, {len(generated)}")
    return dict(client_ttft_seconds=first_token, first_text_seconds=first_text, total_seconds=total,
                generated_token_ids=generated, answer="".join(text), finish_reason=finish, usage=usage)


def new_events(audit):
    """读取 audit 当前可用的完整新行，未完成的末行留待下次。"""
    events = []
    while True:
        position = audit.tell()
        line = audit.readline()
        if not line.endswith("\n"):
            audit.seek(position)
            return events
        events.append(json.loads(line))


def prom(client):
    """获取 client 的原始 Prometheus 文本及复核所需计数器。"""
    raw, values = joint.metrics(client)
    relevant = {key: values.get(name, 0) if key in ("load_bytes", "store_bytes") else values[name]
                for key, name in metrics_06.items()}
    return raw, relevant


def drain(client, audit, before, block_bytes, length):
    """等待 client 发布请求终点和全部迁移；用 audit、before 和 length 界定增量。"""
    deadline = time.monotonic() + 25
    events = []
    while time.monotonic() < deadline:
        events.extend(new_events(audit))
        raw, counters = prom(client)
        delta = {key: value - before[key] for key, value in counters.items()}
        if any(value < 0 for value in delta.values()) or delta["requests"] > 1:
            raise RuntimeError(f"Nonexclusive request metrics: {delta}")
        counts = Counter(event["kind"] for event in events)
        if counts["timing"] == 1 and delta["requests"] == delta["ttft_count"] == 1:
            if counts["begin"] != 1 or counts["lookup"] != 1:
                raise RuntimeError(f"Unexpected connector events: {counts}")
            blocks = Counter()
            for event in events:
                if event["kind"] in ("store", "stored", "loaded"):
                    blocks[event["kind"]] += len(event["keys"])
            lookup = next(event for event in events if event["kind"] == "lookup")
            if (blocks["store"] == blocks["stored"] and blocks["loaded"] * BLOCK == lookup["admitted"]
                    and delta["store_bytes"] == blocks["stored"] * block_bytes
                    and delta["load_bytes"] == blocks["loaded"] * block_bytes):
                if (delta["queries"] != length or delta["preemptions"] != 0 or delta["ttft"] <= 0
                        or delta["gpu"] + delta["cpu"] + delta["computed"] != length):
                    raise RuntimeError(f"Unexpected request metrics: {delta}")
                return raw, counters, delta, events
        time.sleep(0.05)
    raise RuntimeError(f"Request metrics or transfers did not settle: {delta}, {counts}")


def capture(client, audit, directory, plan, block_bytes):
    """重置 client 两层缓存，再将 plan 的实际请求和原始指标保存到 directory。"""
    directory.mkdir()
    reset = client.post("/reset_prefix_cache", params={"reset_external": "true"})
    reset.raise_for_status()
    events = new_events(audit)
    if reset.json() != {"success": True} or len(events) != 1 or events[0]["kind"] != "reset":
        raise RuntimeError(f"Incomplete native cache reset: {reset.text}, {events}")
    save(directory / "reset.json", reset.json())
    raw, before = prom(client)
    (directory / "initial.prom").write_text(raw, encoding="utf-8")
    with (directory / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for row in plan:
            index = row["index"]
            client_id = f"{directory.name}-{index:02}"
            tokens = row["prompt_token_ids"]
            if len(tokens) + MAX_OUTPUT > MAX_LEN:
                raise ValueError(f"Input exceeds the fixed generation budget: {client_id}")
            result = generate(client, tokens, client_id)
            raw, before, delta, events = drain(client, audit, before, block_bytes, len(tokens))
            begin = events[0]
            if (begin["kind"] != "begin" or begin["prompt_sha256"] != row["prompt_sha256"]
                    or begin["clock"] != index or any(event["request_id"] != begin["request_id"] for event in events)
                    or (index == 0 and (delta["gpu"] or delta["cpu"]))):
                raise RuntimeError(f"Request or cache reset mismatch: {client_id}")
            (directory / f"{index:02}.prom").write_text(raw, encoding="utf-8")
            record = {key: value for key, value in row.items() if key != "prompt_token_ids"}
            record.update(result, client_request_id=client_id, request_id=begin["request_id"], metrics=delta)
            stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
            stream.flush()
    (directory / "final.prom").write_text(raw, encoding="utf-8")


def run_policy(config, directory, port, policy, inputs, sequences):
    """用 config 在 port 启动 policy，执行 inputs 的 sequences 后关闭真实进程。"""
    directory.mkdir()
    argv = server_command(config, directory, port, policy)
    save(directory / "server.json", dict(argv=argv, env=SERVER_ENV, policy=policy,
                                         sequences=[sequence["id"] for sequence in sequences]))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    log = directory / "server.log"
    print(f"启动 {directory}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=os.environ | SERVER_ENV, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False) as client:
                joint.ready(client, process, port, log)
                audit_path = directory / "audit.jsonl"
                with audit_path.open(encoding="utf-8") as audit:
                    if [event["kind"] for event in new_events(audit)] != ["init", "context"]:
                        raise RuntimeError(f"Incomplete scheduler startup audit: {audit_path}")
                    for sequence in sequences:
                        capture(client, audit, directory / sequence["id"], replay_plan(inputs, sequence),
                                config["block_bytes"])
                        print(f"已采集 {directory}/{sequence['id']}", flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if audit_path.with_name("audit.jsonl.1").exists():
        raise RuntimeError(f"Audit rolled over: {audit_path}")


def main():
    """读取固定输入和阶段选项，每次只运行一个顺序；调用方持有 Linux GPU 锁。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs")
    parser.add_argument("output")
    parser.add_argument("--model-dir", default=".agent/kv-joint/output/model")
    parser.add_argument("--port", type=int, default=18111)
    parser.add_argument("--epoch", choices=ORDERS, required=True)
    parser.add_argument("--sequence", help="只执行这一条真实序列，用于单独检查采集与复核")
    args = parser.parse_args()
    inputs_path, output, model_dir = (local_path(path, ROOT) for path in (args.inputs, args.output, args.model_dir))
    config = run_config(inputs_path, model_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.exists():
        if read(manifest) != config:
            raise ValueError("Replay environment or frozen inputs changed between epochs")
    else:
        save(manifest, config)
    inputs = read(inputs_path / "inputs.json")
    sequences = [seq for seq in inputs["sequences"] if args.sequence is None or seq["id"] == args.sequence]
    if not sequences:
        parser.error(f"Unknown sequence: {args.sequence}")
    if args.epoch == "reverse":
        sequences.reverse()
    epoch = output / args.epoch
    epoch.mkdir()
    groups = ORDERS[args.epoch]
    save(epoch / "start.json", dict(order=list(groups), sequences=[seq["id"] for seq in sequences]))
    for policy in groups:
        run_policy(config, epoch / policy, args.port, policy, inputs, sequences)
    save(epoch / "complete.json", read(epoch / "start.json"))


if __name__ == "__main__":
    main()
