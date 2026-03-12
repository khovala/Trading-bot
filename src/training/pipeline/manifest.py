from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def load_pipeline_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("pipeline.yaml must contain a 'stages' mapping")
    return manifest


def load_params(path: Path) -> dict[str, Any]:
    return load_yaml(path)
