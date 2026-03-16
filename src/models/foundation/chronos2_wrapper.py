from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.foundation.base_forecaster import FoundationForecasterBase


@dataclass
class Chronos2Wrapper(FoundationForecasterBase):
    model_name: str = "chronos2_wrapper"
    model_version: str = "v2"
    backend_name: str = "chronos2"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "rolling_volatility_20",
        "momentum_10",
        "volume_ratio_20",
        "news_weighted_sentiment_mean",
    )
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    calibration_alpha: float = 0.15
    backend_model_id: str = "amazon/chronos-t5-small"
    backend_context_length: int = 128
    backend_prediction_length: int = 1
    backend_num_samples: int = 20
    backend_stride: int = 64
    backend_max_predict_rows: int = 5000

    _chronos_pipeline: Any = None

    def _try_init_backend(self) -> bool:
        if self._chronos_pipeline is not None:
            return True
        try:
            from chronos import ChronosPipeline  # type: ignore
        except Exception:
            return False
        try:
            self._chronos_pipeline = ChronosPipeline.from_pretrained(self.backend_model_id)
        except Exception:
            return False
        return True

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        metrics = super().fit(rows, target_key=target_key)
        self.backend_available = self._try_init_backend()
        metrics["backend_available"] = 1.0 if self.backend_available else 0.0
        return metrics

    def _backend_predict_distribution(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if not self.backend_available or self._chronos_pipeline is None:
            return []
        if len(rows) < max(8, self.backend_context_length):
            return []
        try:
            import torch
        except Exception:
            return []

        limit = min(len(rows), self.backend_max_predict_rows)
        start_idx = len(rows) - limit
        subset = rows[start_idx:]
        series = [float(r.get("return_1", 0.0)) for r in subset]
        out = [{"0.05": 0.0, "0.5": 0.0, "0.95": 0.0} for _ in subset]
        step = max(1, self.backend_stride)
        for i in range(self.backend_context_length, len(series), step):
            context = series[max(0, i - self.backend_context_length) : i]
            if len(context) < 8:
                continue
            try:
                context_tensor = torch.tensor(context, dtype=torch.float32)
                forecast = self._chronos_pipeline.predict(  # type: ignore[operator]
                    context=context_tensor,
                    prediction_length=self.backend_prediction_length,
                    num_samples=self.backend_num_samples,
                )
                if hasattr(forecast, "detach"):
                    arr = forecast.detach().cpu().numpy()
                elif hasattr(forecast, "numpy"):
                    arr = forecast.numpy()
                else:
                    arr = forecast
                # Expected shape: [num_samples, prediction_length]
                values = [float(v[0]) for v in arr] if len(arr) else [0.0]
            except Exception:
                continue
            values_sorted = sorted(values)
            if not values_sorted:
                continue
            q05 = values_sorted[max(0, int(round((len(values_sorted) - 1) * 0.05)))]
            q50 = values_sorted[max(0, int(round((len(values_sorted) - 1) * 0.50)))]
            q95 = values_sorted[max(0, int(round((len(values_sorted) - 1) * 0.95)))]
            end = min(len(out), i + step)
            for j in range(i, end):
                out[j] = {"0.05": q05, "0.5": q50, "0.95": q95}
        # Prefix rows that were not part of subset via fallback neutral estimates.
        if start_idx > 0:
            pad = [{"0.05": self.mean_target - self.std_target, "0.5": self.mean_target, "0.95": self.mean_target + self.std_target}] * start_idx
            return pad + out
        return out

    def predict_distribution(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        backend = self._backend_predict_distribution(rows)
        if backend and len(backend) == len(rows):
            return backend
        return super().predict_distribution(rows)

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta.update(
            {
                "backend_model_id": self.backend_model_id,
                "backend_context_length": self.backend_context_length,
                "backend_prediction_length": self.backend_prediction_length,
                "backend_num_samples": self.backend_num_samples,
                "backend_stride": self.backend_stride,
                "backend_max_predict_rows": self.backend_max_predict_rows,
            }
        )
        return meta
