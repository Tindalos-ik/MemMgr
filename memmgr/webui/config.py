"""验证 WebUI 配置及本机文件边界，生成任务固定配置。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..inference.models import local_snapshot

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".agent" / "web"
NAMES = ("baseline", "models", "server")


class TaskInput(BaseModel):
    """任务请求；kind、workspace、tier 和 concurrency 指定受约束的现有流程。"""

    model_config = ConfigDict(extra="forbid")
    kind: str = Field(pattern="^(prepare|index|rebuild|rag|retrieval|baseline)$")
    workspace: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    tier: str = Field(default="small", pattern="^(small|medium|full)$")
    concurrency: int = Field(default=1, ge=1, le=4)
    confirm: bool = False


class QueryInput(BaseModel):
    """真实检索或问答参数；query/question 为输入正文，workspace/tier 选择索引。"""

    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, max_length=20000)
    question: str | None = Field(default=None, max_length=20000)
    workspace: str = "formal"
    tier: str = Field(default="small", pattern="^(small|medium|full)$")
    top_k: int = Field(default=3, ge=1, le=100)
    context_tokens: int = Field(default=1024, ge=128, le=32768)

    @model_validator(mode="after")
    def require_text(self):
        """校验 self 至少提供一个非空输入文本。"""
        if not (self.query or self.question or "").strip():
            raise ValueError("query or question must not be empty")
        return self


def safe_name(value):
    """校验 value 为单个安全名称，禁止路径分隔及父目录。"""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
        raise ValueError("Invalid name")
    return value


def read_json(path):
    """读取 path 指定的真实 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    """将 value 原子写入 path，避免进程中断留下半份记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configs():
    """返回三个已保存配置，键为 baseline、models、server。"""
    transaction = STATE / "config-transaction.json"
    if transaction.exists():
        return read_json(transaction)
    return {name: read_json(ROOT / "config" / f"{name}.json") for name in NAMES}


def save_configs(values):
    """以 values 的完整三配置事务提交，崩溃期间由事务文件提供一致读取。"""
    validate_configs(values)
    transaction = STATE / "config-transaction.json"
    save_json(transaction, values)
    for name in NAMES:
        save_json(ROOT / "config" / f"{name}.json", values[name])
    transaction.unlink()


def validate_configs(all_configs):
    """校验 all_configs 类型、范围、跨文件模型身份和本机路径边界。"""
    from ..long_term.index import validate_chunk_config
    from .schema import BaselineConfig, ModelSnapshot, ServerConfig

    if not isinstance(all_configs, dict) or set(all_configs) != set(NAMES):
        raise ValueError("Require baseline, models and server")
    baseline, models, server = (all_configs[name] for name in NAMES)
    BaselineConfig.model_validate(baseline)
    ServerConfig.model_validate(server)
    from ..short_term.config import transfer_config

    transfer_config(server, ROOT)
    if not isinstance(models, dict) or set(models) != {
        "Qwen/Qwen2.5-1.5B-Instruct",
        "BAAI/bge-small-en-v1.5",
    }:
        raise ValueError("Only the two configured project models are supported")
    for item in models.values():
        ModelSnapshot.model_validate(item)
    current = configs()
    if baseline["resources"] != current["baseline"]["resources"]:
        raise ValueError("Resource process/container selectors are read-only")
    tiers = baseline["dataset"]["tiers"]
    if (
        set(tiers) != {"small", "medium", "full"}
        or not tiers["small"] <= tiers["medium"] <= tiers["full"]
    ):
        raise ValueError("Require ordered small, medium, full tier sizes")
    embedding = baseline["embedding"]
    validate_chunk_config(embedding["chunk_tokens"], embedding["chunk_overlap"])
    llm = baseline["llm"]
    if isinstance(llm["temperature"], bool):
        raise ValueError("temperature must be the numeric value 0")
    if llm["max_tokens"] >= llm["max_model_len"]:
        raise ValueError("max_tokens must be less than max_model_len")
    if (
        baseline["experiment"]["context_tokens"]
        > llm["max_model_len"] - llm["max_tokens"]
    ):
        raise ValueError("Context budget plus output exceeds server model length")
    for model, item in models.items():
        local_snapshot(model, item["revision"])
    for section in (embedding, llm):
        if (
            section["model"] not in models
            or section["revision"] != models[section["model"]]["revision"]
        ):
            raise ValueError("Model revision differs across configuration files")
    if (
        llm["tokenizer"] != llm["model"]
        or server["model"] != llm["model"]
        or server["revision"] != llm["revision"]
        or server["max_model_len"] != llm["max_model_len"]
    ):
        raise ValueError("Server and LLM model/tokenizer/revision/context must agree")
    if llm["base_url"] != f"http://{server['host']}:{server['port']}/v1":
        raise ValueError("LLM endpoint differs from server address")
    for key in ("prefix_caching", "chunked_prefill", "concurrency_semantics"):
        if server[key] != current["server"][key]:
            raise ValueError(f"{key} is read-only observed metadata")
    return all_configs


def workspace_root(name):
    """返回 name 工作区真实路径；formal 只用于读取正式语料。"""
    if name == "formal":
        return (ROOT / "data" / "scifact-baseline").resolve()
    safe_name(name)
    path = STATE / "workspaces" / name
    if not path.resolve().is_relative_to((STATE / "workspaces").resolve()):
        raise ValueError("Workspace escapes allowed directory")
    if not path.is_dir():
        raise FileNotFoundError(f"Workspace not found: {name}")
    return path.resolve()


def effective_config(snapshot, workspace, output):
    """由 snapshot 固定模型快照、workspace 数据与 output 输出目录生成实际业务配置。"""
    config = json.loads(json.dumps(snapshot["baseline"]))
    config["dataset"]["root"] = str(workspace_root(workspace))
    config["experiment"]["output_dir"] = str(output)
    config["_web_snapshot"] = snapshot
    config["_web_fail_fast"] = True
    config["_index_registry_path"] = str(
        Path(output).parent / "index-run-registry.json"
    )
    for section in ("embedding", "llm"):
        original = config[section]["model"]
        item = snapshot["models"][original]
        path = str(local_snapshot(original, item["revision"]))
        if section == "embedding":
            config[section]["snapshot_path"] = path
        else:
            config[section]["tokenizer"] = path
    return config
