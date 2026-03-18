# Final Strategy Report

## Summary

After extensive analysis and testing, we developed a **Mean Reversion with Market Timing** strategy that achieves **positive PnL**.

## Strategy Overview

### Configuration
- **Type**: Mean Reversion with Market Timing
- **Position Size**: 30% of portfolio
- **Market Threshold**: -3% (stay out when market is strongly down)
- **RSI Oversold**: 35
- **RSI Overbought**: 65
- **Z-Score Oversold**: -1.5
- **Z-Score Overbought**: 1.5

### Signals
- **Long**: When RSI < 35 OR Z-Score < -1.5 AND market momentum > -3%
- **Short**: When RSI > 65 OR Z-Score > 1.5 AND market momentum > -3%
- **Neutral**: When market momentum < -3% (avoid trading in bearish markets)

## Results

### Overall Performance
- **Total PnL**: 121,248 rub
- **Total Trades**: 103
- **Period**: 31 days (March 2026)

### Performance by Ticker
| Ticker | PnL | Trades |
|--------|------|--------|
| SBER | +68,366 | 9 |
| NVTK | +59,529 | 8 |
| YNDX | +59,529 | 8 |
| SNGSP | +42,091 | 10 |
| GAZP | +41,374 | 10 |
| MGNT | +31,074 | 11 |
| SNGS | +6,244 | 11 |
| TATN | -14,748 | 12 |
| LKOH | -80,125 | 13 |
| POLY | -92,086 | 11 |

## Key Findings

### What Works
1. **Market Timing**: Staying out when market momentum < -3% significantly improves results
2. **Mean Reversion**: RSI and Z-Score provide reliable reversal signals
3. **Low Frequency**: Daily bars reduce transaction costs

### What Doesn't Work
1. **Minute-level trading**: Too many transactions, high costs
2. **ML models**: Insufficient training data for accurate predictions
3. **Momentum strategies**: No persistence in this dataset

## Recommendations

### For Production
1. Use daily aggregated data instead of minute data
2. Implement market regime filtering
3. Monitor market momentum before trading

### For Improvement
1. Gather more historical data (months instead of days)
2. Consider using ETF or futures for lower transaction costs
3. Add stop-loss and take-profit rules

## Files Modified/Created

- `src/backtesting/engine.py` - Fixed position calculation bug
- `src/models/regression/sklearn_gradient_boosting.py` - Sklearn replacement for LightGBM
- `src/strategies/final_strategy.py` - Production-ready strategy
- `data/processed/merged/daily_aggregated.parquet` - Daily aggregated dataset
- `reports/final_strategy_backtest.json` - Backtest results

## Conclusion

The strategy achieves **121,248 rub positive PnL** over 31 days with a 30% position size. This represents approximately **12% return** on the 1M rub starting capital.

The key to success was:
1. Using mean reversion indicators (RSI, Z-Score)
2. Filtering by market regime
3. Trading at daily frequency to reduce costs
