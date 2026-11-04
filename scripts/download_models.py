"""下载固定版本的模型并记录来源。

用法：在 WSL 项目根目录运行 `python scripts/download_models.py`。
脚本按 `config/models.json` 中的固定 revision 下载模型快照并更新缓存路径。
补齐缺失快照时访问 Hugging Face Hub；离线模式直接读取现有缓存。
快照 `cache_dir` 沿用当前部署的固定路径。
"""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """读取必要模型配置，下载两个固定快照并逐个保存缓存路径。

    revision 来自 config/models.json；文件、模型项或 revision 缺失时直接报错。
    """
    os.environ.setdefault("HF_HOME", "/home/tindalos/.cache/huggingface")
    output = Path("config/models.json")
    models = json.loads(output.read_text())

    for name in ["Qwen/Qwen2.5-1.5B-Instruct", "BAAI/bge-small-en-v1.5"]:
        item = models[name]
        item["snapshot_path"] = snapshot_download(
            name,
            revision=item["revision"],
            cache_dir="/home/tindalos/.cache/huggingface/hub",
            ignore_patterns=["*.bin", "onnx/*", "openvino/*", "*.h5", "*.ot"],
        )
        output.write_text(json.dumps(models, indent=2) + "\n")
        print(name, item, flush=True)


if __name__ == "__main__":
    main()
