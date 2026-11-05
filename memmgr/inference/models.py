"""按模型 ID 和固定 revision 从 Hugging Face 标准缓存解析本地快照。"""

from pathlib import Path

from huggingface_hub import snapshot_download

IGNORED_MODEL_FILES = ["*.bin", "onnx/*", "openvino/*", "*.h5", "*.ot"]


def local_snapshot(model: str, revision: str) -> Path:
    """返回 model 在 revision 的本地快照；缓存位置遵循 Hugging Face 环境变量，缺失时直接报错。"""
    return Path(
        snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
            ignore_patterns=IGNORED_MODEL_FILES,
        )
    ).resolve()
