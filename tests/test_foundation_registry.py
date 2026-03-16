from __future__ import annotations

from src.models.foundation.registry import build_forecasters_from_params, parse_forecaster_specs


def test_parse_and_build_forecasters_from_params() -> None:
    params = {
        "models_v2": {
            "foundation_forecasters": [
                {"type": "chronos2", "enabled": True, "prediction_horizon": "30m", "model_version": "x1"},
                {"type": "timesfm2", "enabled": False},
                {"type": "patchtst", "enabled": True},
                {"type": "unknown_model", "enabled": True},
            ]
        }
    }

    specs = parse_forecaster_specs(params)
    models = build_forecasters_from_params(params)

    assert len(specs) == 4
    assert len(models) == 2
    assert models[0].model_name == "chronos2_wrapper"
    assert models[0].prediction_horizon == "30m"
    assert models[1].model_name == "patchtst_wrapper"


def test_foundation_forecaster_fit_predict_contract() -> None:
    params = {"models_v2": {"foundation_forecasters": [{"type": "chronos2", "enabled": True}]}}
    model = build_forecasters_from_params(params)[0]
    rows = [
        {"ticker": "SBER", "return_1": 0.01, "rolling_volatility_20": 0.2},
        {"ticker": "SBER", "return_1": -0.02, "rolling_volatility_20": 0.3},
        {"ticker": "SBER", "return_1": 0.03, "rolling_volatility_20": 0.25},
    ]

    metrics = model.fit(rows, target_key="return_1")
    predictions = model.predict(rows[:2])

    assert metrics["train_samples"] == 3.0
    assert len(predictions) == 2
    assert predictions[0].model_name == "chronos2_wrapper"
    assert 0.0 <= predictions[0].confidence <= 1.0
