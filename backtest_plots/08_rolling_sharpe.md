# Chart 08: Rolling Sharpe Ratio Analysis

## Overview
This chart displays the rolling 20-day Sharpe ratio for the strategy, measuring risk-adjusted returns over time.

## What is Sharpe Ratio?
The Sharpe ratio measures excess return per unit of risk (volatility):
- **> 1.0**: Good risk-adjusted returns
- **> 2.0**: Very good
- **< 1.0**: Poor risk-adjusted returns
- **< 0**: Strategy is losing money on a risk-adjusted basis

## Key Findings

### Summary Statistics
- **Average Rolling Sharpe**: ~0.8 (varies by period)
- **Maximum**: ~2.5-3.0 during strong trending periods
- **Minimum**: -1.5 to -2.0 during drawdowns
- **Volatility of Sharpe**: High variability indicates changing market conditions

## Interpretation by Period

### High Sharpe Periods (Sharpe > 1.5)
- Occur during trending markets with clear mean reversion opportunities
- Tickers like SBER, GAZP, NVTK contribute strongly
- Low volatility combined with consistent gains

### Low/Negative Sharpe Periods (Sharpe < 0)
- Occur during:
  - High volatility without directional moves
  - Whipsaw markets (false signals)
  - Large single-day losses (e.g., POLY, LKOH drawdowns)

### Sharpe Variability Analysis
The high variability in rolling Sharpe suggests:
1. **Market regime changes**: Strategy performs differently in trending vs ranging markets
2. **Ticker-specific performance**: Some tickers (POLY, LKOH) drag down overall Sharpe
3. **Timing sensitivity**: Entry/exit timing significantly affects risk-adjusted returns

## Comparison with Static Sharpe
- If we calculate a single Sharpe for the entire period:
  - Strategy: ~0.6-0.8 (annualized)
  - Buy & Hold: ~0.1-0.3 (annualized)
- Rolling Sharpe shows that performance is NOT consistent - there are periods of outperformance and underperformance

## Implications for Strategy Improvement

### To Stabilize Sharpe Ratio:
1. **Reduce exposure to POLY, LKOH, TATN**: These tickers introduce high variance
2. **Add stop-loss rules**: Limit maximum drawdown per trade
3. **Dynamic position sizing**: Reduce size during high-volatility periods
4. **Regime detection**: Reduce trading during low-Sharpe market regimes

### Expected Impact
- Stabilizing Sharpe around 1.0-1.5 would significantly improve the strategy
- Reducing the worst drawdowns (Sharpe < -1.5) would improve overall performance

## Conclusion
The rolling Sharpe analysis reveals that while our strategy outperforms Buy & Hold in total returns, the risk-adjusted returns are variable. Focus on reducing the variance of returns (especially during drawdowns) would make the strategy more robust for live trading.
