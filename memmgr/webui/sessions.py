"""以原子 JSON 维护本机会话，只把完整成功的请求加入有界历史。"""

import re
import time
import uuid

from fastapi import HTTPException

from .config import STATE, read_json, safe_name, save_json, workspace_root


def identity(snapshot):
    """从 snapshot 提取会话使用的模型、数值模式和上下文身份。"""
    server = snapshot["server"]
    return dict(models=snapshot["models"], **{
        key: server[key] for key in ("model", "revision", "dtype", "max_model_len")
    }, batch_invariant=server.get("kv_cache", {}).get("batch_invariant", False))


def session_path(session_id):
    """将 session_id 限定为会话目录中的单个 JSON 文件。"""
    return STATE / "sessions" / f"{safe_name(session_id)}.json"


def read_session(session_id):
    """读取 session_id 的真实会话，未知标识由文件边界返回 404。"""
    return read_json(session_path(session_id))


def create_session(workspace, tier, snapshot):
    """为 workspace/tier 创建绑定 snapshot 模型身份的独立会话。"""
    workspace_root(workspace)
    session = dict(id=uuid.uuid4().hex, workspace=workspace, tier=tier,
                   identity=identity(snapshot), status="open", turns=[], created_at=time.time())
    save_json(session_path(session["id"]), session)
    return session


def check_session(session_id, workspace, tier, snapshot):
    """要求 session_id 仍开放，且 workspace/tier 和 snapshot 与创建时一致。"""
    session = read_session(session_id)
    if session["status"] != "open":
        raise HTTPException(409, "Session has ended")
    if (session["workspace"], session["tier"], session["identity"]) != (workspace, tier, identity(snapshot)):
        raise HTTPException(409, "Session workspace, tier or model identity changed; create a new session")
    return session


def history_messages(session, tokenizer, budget):
    """从 session 最近完整轮次提取消息；tokenizer 约束历史最多占 budget 的一半。"""
    from ..inference.prompt import chat_input_ids

    messages, retained = [], []
    for request_id in reversed(session["turns"][-32:]):
        record = read_json(STATE / "requests" / f"{safe_name(request_id)}.json")
        if record["status"] != "ok" or record["session_id"] != session["id"]:
            raise ValueError("Session contains an incomplete or unrelated turn")
        question = record["input"].get("question") or record["input"]["query"]
        pair = [{"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": record["answer"]}]
        candidate = pair + messages
        if len(chat_input_ids(tokenizer, [{"role": "system", "content": ""}, *candidate])) > budget // 2:
            break
        messages = candidate
        retained.insert(0, request_id)
    return messages, retained


def citations(answer, included):
    """核对 answer 中 [DOC id] 标记是否属于当前实际 included 证据，返回来源检查结果。"""
    cited = list(dict.fromkeys(re.findall(r"\[DOC\s+([^\]\s]+)\]", answer)))
    unknown = sorted(set(cited) - set(included))
    if unknown:
        raise ValueError(f"Answer cited documents absent from current evidence: {unknown}")
    return dict(cited_doc_ids=cited, allowed_doc_ids=list(included),
                has_citations=bool(cited), source_ids_valid=True)


def commit_turn(session, record):
    """把完整成功的 record 加入 session；调用者须持有共享资源锁。"""
    if record["status"] != "ok" or not record["answer"]:
        raise ValueError("Only complete successful answers can enter session history")
    session["turns"].append(record["id"])
    session["updated_at"] = time.time()
    save_json(session_path(session["id"]), session)


def end_session(session_id):
    """结束 session_id 并清空历史引用；请求审计记录继续保留。"""
    session = read_session(session_id)
    session.update(status="ended", turns=[], ended_at=time.time())
    save_json(session_path(session_id), session)
    return session
