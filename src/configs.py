import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".yaml", ".yml"}:
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("YAML config must be a mapping")
        return obj

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("JSON config must be an object")
        return obj

    raise ValueError("Unsupported config format; use .yaml/.yml or .json")
