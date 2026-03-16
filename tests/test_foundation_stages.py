from __future__ import annotations

from pathlib import Path

from src.data.feature_store.io import write_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.data.ingestion.news.storage import interim_news_items_path
from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.foundation_models import TrainFoundationModelsStage, TrainNewsEncoderStage


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={
            "tracking": {"mlflow": {"enabled": False}},
            "models_v2": {
                "news_encoder": {
                    "encoder_name": "hashing_tfidf_v1",
                    "embedding_dim": 8,
                    "half_life_minutes": 180,
                    "min_tokens": 1,
                    "relevance_threshold": 0.0,
                },
                "foundation_forecasters": [
                    {"type": "chronos2", "enabled": True},
                    {"type": "patchtst", "enabled": True},
                ],
            },
            "stages": {},
        },
        stage_params={"enabled": False},
        run_id="test-run",
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_train_news_encoder_stage_outputs_embeddings(tmp_path: Path) -> None:
    news_path = interim_news_items_path(tmp_path)
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news_path.write_text(
        (
            '{"source":"rbc","title":"profit growth","cleaned_text":"profit growth expected",'
            '"published_at":"2026-01-01T10:00:00+00:00"}\n'
        ),
        encoding="utf-8",
    )

    result = TrainNewsEncoderStage(StageSpec(name="train_news_encoder", purpose="x")).run(
        _ctx(tmp_path, "train_news_encoder")
    )

    assert result.success is True
    assert (tmp_path / "data/processed/news/features/news_embeddings.parquet").exists()
    assert (tmp_path / "models/news_v2/metadata.json").exists()


def test_train_foundation_models_stage_serializes_models(tmp_path: Path) -> None:
    rows = [
        {"ticker": "SBER", "timestamp": "2026-01-01T10:00:00+00:00", "return_1": 0.01},
        {"ticker": "SBER", "timestamp": "2026-01-01T11:00:00+00:00", "return_1": -0.02},
    ]
    write_parquet(merged_train_parquet_path(tmp_path), rows)

    result = TrainFoundationModelsStage(StageSpec(name="train_foundation_models", purpose="x")).run(
        _ctx(tmp_path, "train_foundation_models")
    )

    assert result.success is True
    assert (tmp_path / "models/foundation/chronos2_wrapper.pkl").exists()
    assert (tmp_path / "models/foundation/patchtst_wrapper.pkl").exists()
    assert (tmp_path / "models/foundation/metadata.json").exists()
