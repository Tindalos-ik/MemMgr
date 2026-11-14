"""定义 WebUI 暴露配置的固定类型、范围与路径 allowlist，规则不从可编辑文件推断。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from ..short_term.config import KVConfig


class StrictConfig(BaseModel):
    """所有配置对象拒绝额外字段和隐式字符串/布尔数字转换。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class DatasetConfig(StrictConfig):
    """固定正式只读数据根；实验规模与问题抽样参数使用明确范围。"""

    root: Literal["data/scifact-baseline"]
    tiers: dict[
        Literal["small", "medium", "full"], Annotated[int, Field(ge=1, le=5183)]
    ]
    seed: int = Field(ge=0, le=2147483647)
    query_count: int = Field(ge=1, le=300)


class EmbeddingConfig(StrictConfig):
    """CPU 文档/查询编码与 tokenizer 分块参数。"""

    model: Literal["BAAI/bge-small-en-v1.5"]
    revision: str = Field(pattern="^[a-f0-9]{40}$")
    device: Literal["cpu"]
    batch_size: int = Field(ge=1, le=128)
    chunk_tokens: int = Field(ge=1, le=512)
    chunk_overlap: int = Field(ge=0, le=511)
    query_instruction: str = Field(max_length=1000)


class ChromaConfig(StrictConfig):
    """仅允许项目已经部署的本机 Chroma 和数据集合前缀。"""

    host: Literal["127.0.0.1"]
    port: Literal[8001]
    collection_prefix: Literal["memmgr"]


class LLMConfig(StrictConfig):
    """本机生成端点、固定模型及真实请求/预算限制。"""

    base_url: str = Field(min_length=1, max_length=200)
    model: Literal["Qwen/Qwen2.5-1.5B-Instruct"]
    tokenizer: Literal["Qwen/Qwen2.5-1.5B-Instruct"]
    revision: str = Field(pattern="^[a-f0-9]{40}$")
    max_tokens: int = Field(ge=1, le=8192)
    max_model_len: int = Field(ge=256, le=32768)
    seed: int = Field(ge=0, le=2147483647)
    temperature: Literal[0]
    timeout: float = Field(ge=1, le=3600)


class ExperimentConfig(StrictConfig):
    """固定输出根并限制现有实验请求、并发、重复与预热数量。"""

    output_dir: Literal[".agent/runs"]
    top_k: int = Field(ge=1, le=100)
    context_tokens: int = Field(ge=128, le=32768)
    warmups: int = Field(ge=0, le=300)
    repeats: int = Field(ge=1, le=100)
    concurrency: int = Field(ge=1, le=4)
    queries: int = Field(ge=1, le=300)


class ResourceConfig(StrictConfig):
    """只采样指定容器、卷与正整数 PID，禁止任意命令或文件路径。"""

    sample_interval_seconds: float = Field(ge=0.1, le=60)
    docker_sample_interval_seconds: float = Field(ge=1, le=120)
    chroma_container: Literal["memmgr-chroma-1"]
    chroma_volume_name: Literal["memmgr_chroma_data"]
    vllm_pids: list[Annotated[int, Field(ge=1)]] = Field(max_length=16)


class BaselineConfig(StrictConfig):
    """完整 baseline 的固定配置结构，跨节约束由共用验证函数完成。"""

    dataset: DatasetConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    llm: LLMConfig
    experiment: ExperimentConfig
    resources: ResourceConfig


class ModelSnapshot(StrictConfig):
    """Hugging Face 固定 revision，快照位置使用运行环境的标准缓存。"""

    revision: str = Field(pattern="^[a-f0-9]{40}$")


class ServerConfig(StrictConfig):
    """启动脚本实际支持的 vLLM 设置与只读观测说明。"""

    model: Literal["Qwen/Qwen2.5-1.5B-Instruct"]
    revision: str = Field(pattern="^[a-f0-9]{40}$")
    dtype: Literal["float16", "bfloat16", "auto"]
    max_model_len: int = Field(ge=256, le=32768)
    max_num_seqs: int = Field(ge=1, le=32)
    enforce_eager: bool
    gpu_memory_utilization: float = Field(ge=0.1, le=0.95)
    host: Literal["127.0.0.1", "localhost"]
    port: int = Field(ge=1024, le=65535)
    VLLM_USE_V2_MODEL_RUNNER: Literal["0", "1"]
    VLLM_USE_FLASHINFER_SAMPLER: Literal["0", "1"]
    HF_HUB_OFFLINE: Literal["0", "1"]
    TRANSFORMERS_OFFLINE: Literal["0", "1"]
    prefix_caching: str
    chunked_prefill: str
    concurrency_semantics: str
    kv_cache: KVConfig = Field(default_factory=KVConfig)
