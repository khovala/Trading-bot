from __future__ import annotations

from src.models.classification.binary_direction import BinaryDirectionClassifier
from src.models.classification.multiclass_action import MulticlassActionClassifier
from src.models.ensemble.weighted import WeightedEnsembleModel
from src.models.news.news_feature_model import NewsFeatureModel
from src.models.regression.gru_skeleton import GRURegressionSkeleton
from src.models.regression.tabular_baseline import TabularRegressionBaseline


def _rows() -> list[dict]:
    return [
        {"return_1": 0.01, "sentiment_mean": 0.5, "news_sentiment_mean": 0.5},
        {"return_1": -0.02, "sentiment_mean": -0.3, "news_sentiment_mean": -0.3},
        {"return_1": 0.00, "sentiment_mean": 0.0, "news_sentiment_mean": 0.0},
    ]


def test_baseline_models_emit_standardized_prediction_fields() -> None:
    rows = _rows()
    models = [
        TabularRegressionBaseline(),
        GRURegressionSkeleton(),
        BinaryDirectionClassifier(),
        MulticlassActionClassifier(),
        NewsFeatureModel(),
    ]
    for model in models:
        model.fit(rows)
        pred = model.predict(rows[:1])[0]
        assert hasattr(pred, "expected_return")
        assert hasattr(pred, "direction_probability_up")
        assert hasattr(pred, "direction_probability_down")
        assert hasattr(pred, "confidence")
        assert hasattr(pred, "prediction_horizon")
        assert hasattr(pred, "model_name")
        assert hasattr(pred, "model_version")


def test_weighted_ensemble_combines_predictions() -> None:
    rows = _rows()
    tab = TabularRegressionBaseline()
    bin_model = BinaryDirectionClassifier()
    tab.fit(rows)
    bin_model.fit(rows)
    preds = {
        tab.model_name: tab.predict(rows),
        bin_model.model_name: bin_model.predict(rows),
    }
    ensemble = WeightedEnsembleModel(
        weights={tab.model_name: 0.6, bin_model.model_name: 0.4},
    )
    ensemble.fit([])
    out = ensemble.combine(preds)
    assert len(out) == len(rows)


def test_weighted_ensemble_handles_negative_weights() -> None:
    ensemble = WeightedEnsembleModel(weights={"a": -1.0, "b": 0.0})
    metrics = ensemble.fit([])
    assert metrics["weight_count"] == 2.0
    assert ensemble.weights["a"] == 0.5
    assert ensemble.weights["b"] == 0.5
