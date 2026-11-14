"""校验分层 KV 配置并构造固定模块的 vLLM 接入参数。"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KVConfig(BaseModel):
    """配置缓存层、CPU 策略、可信本地模型与显式回迁成本表。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    tier: Literal["gpu", "cpu", "ssd"] = "gpu"
    policy: Literal["lru", "admission", "learned"] = "lru"
    batch_invariant: bool = False
    block_tokens: Literal[64] = 64
    gpu_bytes: int = Field(default=0, ge=0, le=64 * 2**30)
    cpu_bytes: int = Field(default=64 * 2**20, ge=4 * 2**20, le=4 * 2**30)
    ssd_root: str = ".agent/kv-cache"
    ssd_bytes: int = Field(default=4 * 2**30, ge=16 * 2**20, le=1024 * 2**30)
    model_dir: str = ""
    restore: Literal["always", "never", "gate"] = "always"
    cost_profile: str = ""
    margin_ms: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    audit_path: str = ".agent/kv-events.jsonl"

    @model_validator(mode="after")
    def compatible(self):
        """检查 self 中策略、模型、回迁方式和所选缓存层的组合。"""
        if self.tier == "gpu" and (self.policy != "lru" or self.restore != "always"):
            raise ValueError("GPU-only mode requires lru and always")
        if self.tier != "gpu" and not self.batch_invariant:
            raise ValueError("CPU/SSD KV requires batch_invariant=true for consistent cached and recomputed outputs")
        if self.policy != "lru" and not self.model_dir:
            raise ValueError("Learned policies require a trusted local model_dir")
        if self.restore == "gate" and not self.cost_profile:
            raise ValueError("Cost gate requires an explicit calibrated cost_profile")
        if self.tier == "ssd" and self.restore != "always":
            raise ValueError("SSD cost gating requires a separately validated promotion cost model")
        return self


def local_path(value, root):
    """将 value 解析为 root/.agent 内部路径，拒绝空路径和越界的链接。"""
    if not value:
        raise ValueError("Empty KV artifact path")
    path = (Path(root) / value).resolve()
    if not path.is_relative_to((Path(root) / ".agent").resolve()):
        raise ValueError("KV artifacts must stay within the project .agent directory")
    return path


def transfer_config(server, root):
    """按 server 的 KV 设置构造固定 connector 参数；root 是项目目录。"""
    config = KVConfig.model_validate(server.get("kv_cache", {}))
    options = config.model_dump()
    for field in ("ssd_root", "audit_path"):
        options[field] = str(local_path(options[field], root))
    for field in ("model_dir", "cost_profile"):
        if options[field]:
            path = local_path(options[field], root)
            if not path.exists():
                raise FileNotFoundError(path)
            options[field] = str(path)
    if config.tier == "gpu":
        return None
    if server["max_num_seqs"] != 1 or server["dtype"] != "float16":
        raise ValueError("KV policies currently require one sequence and float16")
    extra = dict(cpu_bytes_to_use=config.cpu_bytes, block_size=config.block_tokens,
                 policy_config=options, model_revision=server["revision"],
                 dtype=server["dtype"], max_model_len=server["max_model_len"],
                 gpu_bytes=config.gpu_bytes)
    if config.tier == "ssd":
        directory = str(Path(options["ssd_root"]) / "invariant-vllm-0.29.0")
        extra.update(spec_name="ManagedTieringSpec", spec_module_path="memmgr.short_term.connector",
                     secondary_tiers=[dict(type="ManagedFileSystemTier", module_path="memmgr.short_term.connector",
                                           root_dir=directory, n_read_threads=4, n_write_threads=4)])
    else:
        extra.update(spec_name="ManagedCPUSpec", spec_module_path="memmgr.short_term.connector")
    return dict(kv_connector="ManagedConnector", kv_role="kv_both",
                kv_connector_module_path="memmgr.short_term.connector", kv_connector_extra_config=extra)
