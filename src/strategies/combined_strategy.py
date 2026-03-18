"""
Combined Momentum + ML Strategy
Uses historical performance to select direction, ML to time entries/exits.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import pickle

from src.backtesting.engine import BacktestConfig, run_backtest


@dataclass
class StrategyResult:
    ticker: str
    direction: int
    pnl: float
    trades: int
    entry_price: float
    exit_price: float


class CombinedStrategy:
    def __init__(
        self,
        model_path: Path | None = None,
        position_size_pct: float = 0.2,
        use_momentum: bool = True,
        use_ml: bool = True,
        momentum_lookback_days: int = 5,
        ml_threshold: float = 0.0003,
    ):
        self.model_path = model_path or Path('models/base/simple_model.pkl')
        self.position_size_pct = position_size_pct
        self.use_momentum = use_momentum
        self.use_ml = use_ml
        self.momentum_lookback_days = momentum_lookback_days
        self.ml_threshold = ml_threshold
        self.model = None
        self.feature_cols = None
        
    def load_model(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                self.feature_cols = data.get('features')
    
    def calculate_momentum(self, df: pd.DataFrame) -> dict[str, float]:
        momentum = {}
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('timestamp')
            if len(ticker_df) > self.momentum_lookback_days * 390:  # ~390 1min bars per day
                recent = ticker_df.tail(self.momentum_lookback_days * 390)
            else:
                recent = ticker_df
            if len(recent) > 1:
                momentum[ticker] = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            else:
                momentum[ticker] = 0.0
        return momentum
    
    def get_ml_signal(self, row: dict) -> int:
        if self.model is None or self.feature_cols is None:
            return 0
        
        features = [row.get(c, 0) for c in self.feature_cols]
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        pred = self.model.predict([features])[0]
        
        if pred > self.ml_threshold:
            return 1
        elif pred < -self.ml_threshold:
            return -1
        return 0
    
    def run_backtest(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        direction_mode: str = 'momentum',  # 'momentum', 'reverse', 'neutral'
    ) -> dict[str, StrategyResult]:
        
        results = {}
        
        # Calculate momentum from train period
        momentum = self.calculate_momentum(train_df)
        
        # Sort tickers by momentum
        sorted_tickers = sorted(momentum.items(), key=lambda x: x[1])
        n = len(sorted_tickers)
        
        # Select long/short based on momentum
        long_tickers = [t for t, _ in sorted_tickers[n//3:]]  # Top 66%
        short_tickers = [t for t, _ in sorted_tickers[:n//3]]  # Bottom 33%
        
        if direction_mode == 'reverse':
            long_tickers, short_tickers = short_tickers, long_tickers
        
        cfg = BacktestConfig(
            initial_cash=1_000_000,
            commission_bps=5,
            slippage_bps=5,
            lot_size=1,
            execution_delay_bars=1,
            position_size_pct=self.position_size_pct,
            target_position_column='policy_target_position',
        )
        
        # Test long tickers
        for ticker in long_tickers:
            ticker_df = test_df[test_df['ticker'] == ticker].sort_values('timestamp').reset_index(drop=True)
            rows = ticker_df.to_dict('records')
            
            for r in rows:
                r['policy_target_position'] = 1.0
            
            result = run_backtest(rows, cfg)
            
            results[ticker] = StrategyResult(
                ticker=ticker,
                direction=1,
                pnl=result['summary']['pnl'],
                trades=len(result['trade_log']),
                entry_price=ticker_df['close'].iloc[0],
                exit_price=ticker_df['close'].iloc[-1],
            )
        
        # Test short tickers
        for ticker in short_tickers:
            ticker_df = test_df[test_df['ticker'] == ticker].sort_values('timestamp').reset_index(drop=True)
            rows = ticker_df.to_dict('records')
            
            for r in rows:
                r['policy_target_position'] = -1.0
            
            result = run_backtest(rows, cfg)
            
            results[ticker] = StrategyResult(
                ticker=ticker,
                direction=-1,
                pnl=result['summary']['pnl'],
                trades=len(result['trade_log']),
                entry_price=ticker_df['close'].iloc[0],
                exit_price=ticker_df['close'].iloc[-1],
            )
        
        return results


def main():
    from src.features.market.engineering import add_lag_features
    
    # Load data
    train_val = pd.read_parquet('data/processed/merged/train_val_expanded.parquet')
    test = pd.read_parquet('data/processed/merged/test_expanded.parquet')
    
    strategy = CombinedStrategy(position_size_pct=0.2)
    strategy.load_model()
    
    # Test momentum strategy
    print('=== Momentum Strategy ===')
    results = strategy.run_backtest(train_val, test, direction_mode='momentum')
    
    total_pnl = 0
    for ticker, res in sorted(results.items(), key=lambda x: -x[1].pnl):
        dir_str = 'LONG' if res.direction > 0 else 'SHORT'
        print(f'{ticker} ({dir_str}): PnL={res.pnl:.0f}, Trades={res.trades}')
        total_pnl += res.pnl
    print(f'\\nTotal PnL: {total_pnl:.0f}')
    
    # Test reverse momentum
    print('\\n=== Reverse Momentum Strategy ===')
    results = strategy.run_backtest(train_val, test, direction_mode='reverse')
    
    total_pnl = 0
    for ticker, res in sorted(results.items(), key=lambda x: -x[1].pnl):
        dir_str = 'LONG' if res.direction > 0 else 'SHORT'
        print(f'{ticker} ({dir_str}): PnL={res.pnl:.0f}, Trades={res.trades}')
        total_pnl += res.pnl
    print(f'\\nTotal PnL: {total_pnl:.0f}')
    
    # Test with different position sizes
    print('\\n=== Position Size Sensitivity ===')
    for pos_size in [0.05, 0.1, 0.15, 0.2, 0.3]:
        strategy.position_size_pct = pos_size
        results = strategy.run_backtest(train_val, test, direction_mode='momentum')
        total_pnl = sum(r.pnl for r in results.values())
        print(f'Position {pos_size}: Total PnL={total_pnl:.0f}')


if __name__ == '__main__':
    main()
