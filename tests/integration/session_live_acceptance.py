"""通过真实 WebAPI 验收多轮 RAG、会话隔离、预算与取消；需独占真实服务。

运行 python tests/integration/session_live_acceptance.py --output .agent/session-acceptance。
服务由调用者启动；本脚本只创建会话、查询已有索引及保存原始响应。
"""

import argparse
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_raa/06_kv_placement"))

from run_placement import save


def idle(client, output=None):
    """等待 client 的真实引擎没有运行或排队请求，可将原始指标写入 output。"""
    deadline = time.monotonic() + 30
    stable = 0
    while time.monotonic() < deadline:
        raw = client.get("/metrics").raise_for_status().text
        gauges = {line.split()[0]: float(line.split()[-1]) for line in raw.splitlines()
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))}
        if len(gauges) < 2:
            raise ValueError("Engine did not expose real running/waiting gauges")
        stable = stable + 1 if all(value == 0 for value in gauges.values()) else 0
        if stable >= 3:
            if output:
                output.write_text(raw)
            return raw
        time.sleep(.35)
    raise TimeoutError("Engine did not release running or waiting requests")


def ask(client, body, cancel=None):
    """用 client 发送 body，解析真实 SSE；cancel 可在指定事件触发真实取消或断开。"""
    events, name, started, first = {}, None, time.perf_counter(), None
    with client.stream("POST", "/api/qa", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                value = json.loads(line[6:])
                events[name] = value
                if name == "delta" and first is None:
                    first = time.perf_counter() - started
                if cancel and name == cancel:
                    if cancel == "disconnect":
                        break
                    busy = client.post("/api/qa", json=body)
                    ending = client.delete(f"/api/sessions/{body['session_id']}")
                    if busy.status_code != 409 or ending.status_code != 409:
                        raise ValueError("Active generation did not protect its resources")
                    client.post(f"/api/sessions/{body['session_id']}/cancel", json={}).raise_for_status()
                    cancel = None
                elif cancel == "disconnect" and name == "delta":
                    break
    return dict(events=events, client_ttft_seconds=first, client_total_seconds=time.perf_counter() - started)


def completed(client, body):
    """经 client 完成 body，并核对实际历史、当前证据、用量和持久请求记录。"""
    result = ask(client, body)
    if "error" in result["events"] or "done" not in result["events"]:
        raise ValueError(f"Real QA failed: {result['events'].get('error')}")
    record = result["events"]["done"]
    if record["status"] != "ok" or not record["answer"] or not record["citations"]["source_ids_valid"]:
        raise ValueError("Incomplete answer or invalid actual citation source")
    if record["usage"]["prompt_tokens"] != record["prompt"]["input_tokens_local"]:
        raise ValueError("Local and engine prompt token counts differ")
    if client.get(f"/api/requests/{record['id']}").raise_for_status().json() != record:
        raise ValueError("SSE completion differs from the persisted request")
    if body.get("session_id"):
        session = client.get(f"/api/sessions/{body['session_id']}").raise_for_status().json()
        if session["turns"][-1] != record["id"] or len(session["turns"]) != record["turn"]:
            raise ValueError("Session commit differs from the actual completed turn")
    result["record"] = record
    return result


def interrupted(client, engine, body, event):
    """在 body 的 event 阶段取消或断开，检查真实清理及历史未追加。"""
    before = client.get(f"/api/sessions/{body['session_id']}").raise_for_status().json()["turns"]
    result = ask(client, body, cancel=event)
    request_id = result["events"]["request"]["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        record = client.get(f"/api/requests/{request_id}").raise_for_status().json()
        if record["status"] != "running":
            break
        time.sleep(.1)
    if record["status"] != "interrupted" or "done" in result["events"]:
        raise ValueError("Cancelled real stream was accepted as a complete turn")
    idle(engine)
    after = client.get(f"/api/sessions/{body['session_id']}").raise_for_status().json()["turns"]
    if before != after:
        raise ValueError("Cancelled generation changed committed history")
    return dict(request_id=request_id, trigger=event, status=record["status"], history_unchanged=True)


def acceptance(client, engine, output):
    """使用 client/engine 验收双会话与真实取消，将记录保存到 output。"""
    output.mkdir()
    sessions = [client.post("/api/sessions", json={"workspace": "formal", "tier": "small"})
                .raise_for_status().json() for _ in range(2)]
    topics = ["The DdrB protein from Deinococcus radiodurans is an alternative SSB.",
              "A vegetarian diet improves cardiovascular risk factors in type 2 diabetes."]
    markers = ["SESSION_ALDER_1701", "SESSION_BIRCH_1702"]
    results = [[], []]
    for turn in range(2):
        for index, session in enumerate(sessions):
            question = (f"Topic label {markers[index]}. What does the evidence establish about: {topics[index]}"
                        if turn == 0 else "Which topic did I ask about, and what does the current evidence support? Cite a document.")
            body = dict(session_id=session["id"], workspace="formal", tier="small", query=topics[index],
                        question=question, context_tokens=1536, top_k=2)
            if turn == 1 and index == 0:
                del body["query"]
            result = completed(client, body)
            prompt = result["record"]["prompt"]
            if markers[1 - index] in prompt["rendered_prompt"]:
                raise ValueError("A conversation received the other session's private marker")
            if turn and (markers[index] not in prompt["rendered_prompt"]
                         or prompt["history_request_ids"] != [results[index][0]["record"]["id"]]):
                raise ValueError("Follow-up omitted its own actual complete history")
            results[index].append(result)
            idle(engine)
    body = dict(session_id=sessions[0]["id"], workspace="formal", tier="small", query=topics[0],
                question="Explain the current evidence in detail and cite its sources.", context_tokens=1536)
    cancellations = [interrupted(client, engine, body, event) for event in ("prompt", "delta", "disconnect")]
    resumed = completed(client, body)
    idle(engine)
    mismatch = client.post("/api/qa", json=dict(body, tier="medium"))
    if mismatch.status_code != 409:
        raise ValueError("Session accepted a different index tier")
    missing = client.post("/api/qa", json=dict(body, session_id="0" * 32))
    if missing.status_code != 404:
        raise ValueError("Unknown session was accepted")
    before = client.get(f"/api/sessions/{sessions[0]['id']}").raise_for_status().json()["turns"]
    oversized = ask(client, dict(body, question="overlong question " * 600, context_tokens=128))
    if "error" not in oversized["events"] or "done" in oversized["events"]:
        raise ValueError("Over-budget question was accepted")
    if client.get(f"/api/sessions/{sessions[0]['id']}").json()["turns"] != before:
        raise ValueError("Failed prompt changed history")
    client.delete(f"/api/sessions/{sessions[1]['id']}").raise_for_status()
    ended = client.get(f"/api/sessions/{sessions[1]['id']}").json()
    if ended["status"] != "ended" or ended["turns"]:
        raise ValueError("Ended session retained active history ownership")
    if client.post("/api/qa", json=dict(body, session_id=sessions[1]["id"])).status_code != 409:
        raise ValueError("Ended session accepted another turn")
    for index, records in enumerate(results):
        save(output / f"session-{index}.json", records)
    save(output / "resumed.json", resumed)
    save(output / "cancellations.json", cancellations)
    save(output / "budget-error.json", oversized)
    result = dict(two_sessions=True, own_history=True, cross_session_marker_absent=True,
                  cancellations=cancellations, after_cancel_success=True, budget_rejected=True,
                  tier_change_rejected=True, unknown_session_rejected=True, ended_session_rejected=True,
                  resume_session_id=sessions[0]["id"], resume_turns=before)
    save(output / "summary.json", result)
    return result


def main():
    """按命令行的真实 WebAPI、引擎地址与 output 执行集中验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", default="http://127.0.0.1:18765")
    parser.add_argument("--engine", default="http://127.0.0.1:18086")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with httpx.Client(base_url=args.web, timeout=240, trust_env=False) as client, \
            httpx.Client(base_url=args.engine, timeout=180, trust_env=False) as engine:
        acceptance(client, engine, args.output)


if __name__ == "__main__":
    main()
