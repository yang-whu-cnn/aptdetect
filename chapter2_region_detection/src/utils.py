# src/utils.py
import os
import json
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: str) -> Dict[str, Any]:
    # 避免引入额外依赖，优先用 PyYAML；没有就报错提示
    try:
        import yaml
    except ImportError as e:
        raise ImportError("需要安装 PyYAML：pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
