from __future__ import annotations

from types import SimpleNamespace

from src.training.tracking import mlflow_utils


def test_mlflow_config_from_params_prefers_stage_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        mlflow_utils,
        "get_settings",
        lambda: SimpleNamespace(mlflow_tracking_uri="http://default:5000", mlflow_experiment="default-exp"),
    )
    params = {"tracking": {"mlflow": {"tracking_uri": "http://params:5000", "experiment_name": "params-exp"}}}
    stage_params = {"tracking_uri": "http://stage:5000", "experiment_name": "stage-exp", "enabled": False}

    cfg = mlflow_utils.MLflowConfig.from_params(params=params, stage_params=stage_params)
    assert cfg.tracking_uri == "http://stage:5000"
    assert cfg.experiment_name == "stage-exp"
    assert cfg.enabled is False


def test_mlflow_tracker_run_and_logging_calls_mlflow(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class _RunCtx:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", lambda uri: calls.setdefault("uri", uri))
    monkeypatch.setattr(mlflow_utils.mlflow, "set_experiment", lambda exp: calls.setdefault("exp", exp))
    def _start_run(**kwargs):
        calls["start_run_kwargs"] = kwargs
        return _RunCtx()

    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", _start_run)
    monkeypatch.setattr(mlflow_utils.mlflow, "log_params", lambda p: calls.setdefault("params", p))
    monkeypatch.setattr(mlflow_utils.mlflow, "log_metrics", lambda m, step=None: calls.setdefault("metrics", (m, step)))
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda file_path, artifact_path=None: calls.setdefault("artifact", (file_path, artifact_path)),
    )

    tracker = mlflow_utils.MLflowTracker(
        config=mlflow_utils.MLflowConfig(tracking_uri="http://mlflow:5000", experiment_name="exp", enabled=True)
    )
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("x", encoding="utf-8")

    with tracker.run("stage-a", tags={"k": "v"}, nested=True) as run_id:
        assert run_id == "run-123"
        tracker.log_params({"a": 1, "nested": {"x": 1}})
        tracker.log_metrics({"m": 0.1}, step=2)
        tracker.log_artifact(str(artifact_file), artifact_path="reports")

    assert calls["uri"] == "http://mlflow:5000"
    assert calls["exp"] == "exp"
    assert calls["start_run_kwargs"]["nested"] is True
    assert calls["metrics"] == ({"m": 0.1}, 2)


def test_mlflow_tracker_disabled_noops(monkeypatch) -> None:
    called = {"start_run": 0}
    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", lambda **kwargs: called.__setitem__("start_run", 1))
    tracker = mlflow_utils.MLflowTracker(
        config=mlflow_utils.MLflowConfig(tracking_uri="http://mlflow:5000", experiment_name="exp", enabled=False)
    )
    with tracker.run("disabled-run") as run_id:
        assert run_id is None
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0})
    assert called["start_run"] == 0


def test_mlflow_tracker_fail_open_when_not_strict(monkeypatch) -> None:
    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", lambda uri: (_ for _ in ()).throw(RuntimeError("down")))
    tracker = mlflow_utils.MLflowTracker(
        config=mlflow_utils.MLflowConfig(
            tracking_uri="http://mlflow:5000",
            experiment_name="exp",
            enabled=True,
            strict=False,
        )
    )
    with tracker.run("fail-open") as run_id:
        assert run_id is None
    assert tracker.last_error is not None


def test_mlflow_tracker_strict_raises(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", lambda uri: (_ for _ in ()).throw(RuntimeError("down")))
    tracker = mlflow_utils.MLflowTracker(
        config=mlflow_utils.MLflowConfig(
            tracking_uri="http://mlflow:5000",
            experiment_name="exp",
            enabled=True,
            strict=True,
        )
    )
    with pytest.raises(RuntimeError):
        with tracker.run("strict"):
            pass
