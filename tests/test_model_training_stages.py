from __future__ import annotations

from pathlib import Path

from src.data.feature_store.io import write_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.data.ingestion.news.storage import news_features_parquet_path
from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.model_training import (
    TrainBaseModelsStage,
    TrainEnsembleModelStage,
    TrainNewsModelStage,
)


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={
            "models": {
                "prediction_horizon_minutes": 60,
                "weights": {
                    "tabular_regression": 0.2,
                    "rnn_regression": 0.2,
                    "binary_classifier": 0.2,
                    "multiclass_classifier": 0.2,
                    "news_model": 0.2,
                },
            },
            "features": {"news": {"schema_version": "v1"}, "merged": {"schema_version": "v1"}},
            "stages": {},
        },
        stage_params={"enabled": False},
        run_id="test-run",
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_training_stages_write_serialized_artifacts(tmp_path: Path) -> None:
    merged_rows = [{"ticker": "SBER", "timestamp": "2026-01-01T10:00:00+00:00", "return_1": 0.01}]
    news_rows = [{"ticker": "SBER", "timestamp_bucket": "2026-01-01T10:00:00+00:00", "sentiment_mean": 0.2}]
    write_parquet(merged_train_parquet_path(tmp_path), merged_rows)
    write_parquet(news_features_parquet_path(tmp_path), news_rows)

    base = TrainBaseModelsStage(StageSpec(name="train_base_models", purpose="x")).run(_ctx(tmp_path, "train_base_models"))
    news = TrainNewsModelStage(StageSpec(name="train_news_model", purpose="x")).run(_ctx(tmp_path, "train_news_model"))
    ens = TrainEnsembleModelStage(StageSpec(name="train_ensemble_model", purpose="x")).run(
        _ctx(tmp_path, "train_ensemble_model")
    )

    assert base.success is True
    assert news.success is True
    assert ens.success is True
    assert (tmp_path / "models/base/tabular_regression_baseline.pkl").exists()
    assert (tmp_path / "models/news/news_feature_model.pkl").exists()
    assert (tmp_path / "models/ensemble/weighted_ensemble.pkl").exists()
    assert (tmp_path / "artifacts/evaluation/ensemble_ablation.json").exists()
