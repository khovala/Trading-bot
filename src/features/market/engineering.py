from __future__ import annotations

from datetime import datetime
import math

from src.data.schemas.market import CandleRecord


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def generate_market_features(candles_by_ticker: dict[str, list[CandleRecord]]) -> list[dict]:
    rows: list[dict] = []
    for ticker in sorted(candles_by_ticker):
        candles = sorted(candles_by_ticker[ticker], key=lambda x: x.timestamp)
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [float(c.volume) for c in candles]

        ema12 = None
        ema26 = None
        signal = None
        alpha12 = 2.0 / (12 + 1)
        alpha26 = 2.0 / (26 + 1)
        alpha9 = 2.0 / (9 + 1)

        prev_return_1 = 0.0
        prev_volatility = 0.0
        prev_rsi = 50.0

        for i, candle in enumerate(candles):
            close = closes[i]
            prev_close = closes[i - 1] if i > 0 else close
            ret = (close / prev_close - 1.0) if i > 0 and prev_close else 0.0
            log_ret = math.log(close / prev_close) if i > 0 and prev_close > 0 else 0.0

            win_returns = [math.log(closes[j] / closes[j - 1]) for j in range(max(1, i - 19), i + 1)]
            rolling_vol = _std(win_returns)

            momentum_10 = close / closes[i - 10] - 1.0 if i >= 10 and closes[i - 10] else 0.0

            gains: list[float] = []
            losses: list[float] = []
            for j in range(max(1, i - 13), i + 1):
                diff = closes[j] - closes[j - 1]
                gains.append(max(diff, 0.0))
                losses.append(abs(min(diff, 0.0)))
            avg_gain = _mean(gains)
            avg_loss = _mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
            rsi = 100.0 - (100.0 / (1.0 + rs)) if avg_loss > 0 else 100.0

            ema12 = close if ema12 is None else alpha12 * close + (1 - alpha12) * ema12
            ema26 = close if ema26 is None else alpha26 * close + (1 - alpha26) * ema26
            macd = ema12 - ema26
            signal = macd if signal is None else alpha9 * macd + (1 - alpha9) * signal

            trs: list[float] = []
            for j in range(max(0, i - 13), i + 1):
                prev = closes[j - 1] if j > 0 else closes[j]
                tr = max(highs[j] - lows[j], abs(highs[j] - prev), abs(lows[j] - prev))
                trs.append(tr)
            atr = _mean(trs)

            win_closes = closes[max(0, i - 19) : i + 1]
            close_mean = _mean(win_closes)
            close_std = _std(win_closes)
            zscore = (close - close_mean) / close_std if close_std > 0 else 0.0
            
            bb_upper = close_mean + 2 * close_std
            bb_lower = close_mean - 2 * close_std
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            
            mean_reversion_signal = 1.0 if abs(zscore) > 2.0 else (0.5 if abs(zscore) > 1.5 else 0.0)
            mean_reversion_direction = -1.0 if zscore > 1.5 else (1.0 if zscore < -1.5 else 0.0)

            win_volumes = volumes[max(0, i - 19) : i + 1]
            vol_mean = _mean(win_volumes)
            vol_std = _std(win_volumes)
            vol_ratio = volumes[i] / vol_mean if vol_mean > 0 else 0.0
            vol_z = (volumes[i] - vol_mean) / vol_std if vol_std > 0 else 0.0

            trend_regime = 1.0 if macd >= 0 else 0.0
            volatility_regime = 1.0 if rolling_vol > 0.01 else 0.0

            return_lag_1 = prev_return_1
            return_lag_2 = rows[-1].get("return_lag_1", 0.0) if rows and rows[-1].get("ticker") == ticker else 0.0
            return_lag_5 = rows[-1].get("return_lag_2", 0.0) if rows and rows[-1].get("ticker") == ticker else 0.0

            volatility_lag_1 = prev_volatility
            rsi_lag_1 = prev_rsi

            macd_momentum_interaction = macd * momentum_10
            volume_volatility_interaction = vol_ratio * rolling_vol

            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": candle.timestamp.isoformat(),
                    "close": float(close),
                    "return_1": float(ret),
                    "log_return_1": float(log_ret),
                    "rolling_volatility_20": float(rolling_vol),
                    "momentum_10": float(momentum_10),
                    "rsi_14": float(rsi),
                    "macd": float(macd),
                    "macd_signal": float(signal),
                    "atr_14": float(atr),
                    "zscore_20": float(zscore),
                    "bb_position": float(bb_position),
                    "mean_reversion_signal": float(mean_reversion_signal),
                    "mean_reversion_direction": float(mean_reversion_direction),
                    "volume": float(volumes[i]),
                    "volume_ratio_20": float(vol_ratio),
                    "volume_zscore_20": float(vol_z),
                    "trend_regime": float(trend_regime),
                    "volatility_regime": float(volatility_regime),
                    "return_lag_1": float(return_lag_1),
                    "return_lag_2": float(return_lag_2),
                    "return_lag_5": float(return_lag_5),
                    "volatility_lag_1": float(volatility_lag_1),
                    "rsi_lag_1": float(rsi_lag_1),
                    "macd_momentum_interaction": float(macd_momentum_interaction),
                    "volume_volatility_interaction": float(volume_volatility_interaction),
                }
            )

            prev_return_1 = ret
            prev_volatility = rolling_vol
            prev_rsi = rsi

    rows.sort(key=lambda x: (x["ticker"], x["timestamp"]))
    return rows


def parse_market_candle_rows(rows: list[dict]) -> list[CandleRecord]:
    return [CandleRecord.model_validate(row) for row in rows]
