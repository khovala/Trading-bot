"""
Final Production Strategy: Mean Reversion with Market Timing
Combines RSI/Z-score mean reversion with market regime filtering.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import json

from src.backtesting.engine import BacktestConfig, run_backtest


@dataclass
class StrategyConfig:
    position_size_pct: float = 0.3
    market_threshold: float = -0.03
    use_market_timing: bool = True
    rsi_oversold: float = 35
    rsi_overbought: float = 65
    zscore_oversold: float = -1.5
    zscore_overbought: float = 1.5
    commission_bps: float = 5.0
    slippage_bps: float = 5.0


@dataclass
class BacktestResult:
    total_pnl: float
    trades: int
    sharpe: float
    max_drawdown: float
    win_rate: float
    tickers: dict


class MeanReversionMarketTimingStrategy:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        df = df.copy()
        
        # Calculate market momentum
        market_mom = df.groupby('date')['daily_return'].mean().to_dict()
        df['market_momentum'] = df['date'].map(market_mom)
        
        cfg = BacktestConfig(
            initial_cash=1_000_000,
            commission_bps=self.config.commission_bps,
            slippage_bps=self.config.slippage_bps,
            lot_size=1,
            execution_delay_bars=1,
            position_size_pct=self.config.position_size_pct,
            target_position_column='policy_target_position',
        )
        
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        results_by_ticker = {}
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            rows = ticker_df.to_dict('records')
            
            positions = []
            for row in rows:
                market_mom = row.get('market_momentum', 0)
                rsi = row.get('rsi_14_last', 50)
                zscore = row.get('zscore_20_last', 0)
                
                # Market timing filter
                if self.config.use_market_timing and market_mom < self.config.market_threshold:
                    positions.append(0.0)
                    continue
                
                # Mean reversion signals
                if rsi < self.config.rsi_oversold or zscore < self.config.zscore_oversold:
                    positions.append(1.0)  # Long (oversold)
                elif rsi > self.config.rsi_overbought or zscore > self.config.zscore_overbought:
                    positions.append(-1.0)  # Short (overbought)
                else:
                    positions.append(0.0)  # Neutral
            
            for r, pos in zip(rows, positions):
                r['policy_target_position'] = pos
            
            result = run_backtest(rows, cfg)
            pnl = result['summary']['pnl']
            trades = len(result['trade_log'])
            
            total_pnl += pnl
            total_trades += trades
            
            # Calculate wins/losses from trade log
            wins = sum(1 for t in result['trade_log'] if t.get('delta_qty', 0) > 0)
            total_wins += wins
            
            results_by_ticker[ticker] = {
                'pnl': pnl,
                'trades': trades,
                'wins': wins,
                'losses': trades - wins,
            }
        
        win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0
        
        return BacktestResult(
            total_pnl=total_pnl,
            trades=total_trades,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=win_rate,
            tickers=results_by_ticker,
        )
    
    def optimize(self, df: pd.DataFrame) -> dict:
        best_pnl = float('-inf')
        best_config = None
        results = []
        
        for market_thresh in [-0.01, -0.02, -0.03, -0.05]:
            for rsi_oversold in [30, 35, 40]:
                for rsi_overbought in [60, 65, 70]:
                    for zscore_thresh in [1.0, 1.5, 2.0]:
                        cfg = StrategyConfig(
                            market_threshold=market_thresh,
                            rsi_oversold=rsi_oversold,
                            rsi_overbought=rsi_overbought,
                            zscore_oversold=-zscore_thresh,
                            zscore_overbought=zscore_thresh,
                        )
                        self.config = cfg
                        
                        result = self.run_backtest(df)
                        
                        results.append({
                            'market_threshold': market_thresh,
                            'rsi_oversold': rsi_oversold,
                            'rsi_overbought': rsi_overbought,
                            'zscore_threshold': zscore_thresh,
                            'pnl': result.total_pnl,
                            'trades': result.trades,
                        })
                        
                        if result.total_pnl > best_pnl:
                            best_pnl = result.total_pnl
                            best_config = cfg
        
        return {
            'best_config': best_config,
            'best_pnl': best_pnl,
            'all_results': results,
        }


def main():
    df = pd.read_parquet('data/processed/merged/daily_aggregated.parquet')
    print(f'Data: {len(df)} rows, {df[\"date\"].nunique()} days')
    
    strategy = MeanReversionMarketTimingStrategy()
    
    print('\\n=== Final Strategy Backtest ===')
    result = strategy.run_backtest(df)
    
    print(f'Total PnL: {result.total_pnl:,.0f} rub')
    print(f'Total Trades: {result.trades}')
    print(f'Win Rate: {result.win_rate:.1%}')
    
    print('\\nBy Ticker:')
    for ticker, res in sorted(result.tickers.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f'  {ticker}: PnL={res[\"pnl\"]:,.0f}, Trades={res[\"trades\"]}')
    
    # Save result
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'strategy': 'MeanReversionMarketTiming',
        'config': {
            'position_size_pct': strategy.config.position_size_pct,
            'market_threshold': strategy.config.market_threshold,
            'rsi_oversold': strategy.config.rsi_oversold,
            'rsi_overbought': strategy.config.rsi_overbought,
            'zscore_oversold': strategy.config.zscore_oversold,
            'zscore_overbought': strategy.config.zscore_overbought,
        },
        'results': {
            'total_pnl': result.total_pnl,
            'trades': result.trades,
            'win_rate': result.win_rate,
            'sharpe': result.sharpe,
        },
    }
    
    Path('reports').mkdir(exist_ok=True)
    with open('reports/final_strategy_backtest.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print('\\nResults saved to reports/final_strategy_backtest.json')


if __name__ == '__main__':
    main()
