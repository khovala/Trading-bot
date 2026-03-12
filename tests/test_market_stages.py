from __future__ import annotations

from src.training.pipeline.registry import StageRegistry
from src.training.pipeline.runner import PipelineRunner
from src.training.stages.register import register_default_stages


def test_market_download_and_preprocess_stages(tmp_path) -> None:
    manifest = {
        "stages": {
            "download_market_data": {
                "purpose": "download",
                "deps": [],
                "outs": ["data/raw/market"],
                "reports": ["reports/download_market_data.json"],
            },
            "preprocess_market_data": {
                "purpose": "preprocess",
                "deps": ["data/raw/market"],
                "outs": ["data/interim/market"],
                "reports": ["reports/preprocess_market_data.json"],
            },
        }
    }
    params = {
        "global": {"seed": 1},
        "data": {
            "market": {
                "provider": "mock",
                "interval": "1min",
                "lookback_days": 1,
                "instruments": ["SBER"],
            }
        },
        "stages": {"preprocess_market_data": {"interval_minutes": 1}},
    }
    registry = StageRegistry()
    register_default_stages(registry)
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    download_result = runner.run_stage("download_market_data")
    preprocess_result = runner.run_stage("preprocess_market_data", fail_on_missing_inputs=True)

    assert download_result.success is True
    assert preprocess_result.success is True
    assert (tmp_path / "data/raw/market/instruments.json").exists()
    assert (tmp_path / "data/interim/market/instruments.json").exists()
    assert (tmp_path / "data/interim/market/candles/SBER.jsonl").exists()
