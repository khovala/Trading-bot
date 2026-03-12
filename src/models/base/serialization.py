from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
