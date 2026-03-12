from __future__ import annotations

from pathlib import Path


def market_features_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/market/features/market_features.parquet"


def merged_train_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/merged/train.parquet"


def merged_validation_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/merged/validation.parquet"


def merged_test_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/merged/test.parquet"
