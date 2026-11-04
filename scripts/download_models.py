"""将 config/models.json 中固定 revision 的模型下载到 Hugging Face 标准缓存。

用法：在项目根目录运行 `python scripts/download_models.py`。
缓存位置遵循 HF_HUB_CACHE、HF_HOME 等标准环境变量；下载不改写项目配置。
补齐缺失快照时访问 Hugging Face Hub；离线模式直接读取现有缓存。
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memmgr.models import IGNORED_MODEL_FILES


def main() -> None:
    """读取项目模型配置并逐个下载固定快照，输出模型来源及本次实际缓存路径。"""
    config = Path(__file__).resolve().parents[1] / "config" / "models.json"
    models = json.loads(config.read_text(encoding="utf-8"))
    for name, item in models.items():
        snapshot = snapshot_download(
            name,
            revision=item["revision"],
            ignore_patterns=IGNORED_MODEL_FILES,
        )
        print(name, {**item, "snapshot_path": str(Path(snapshot).resolve())}, flush=True)


if __name__ == "__main__":
    main()
