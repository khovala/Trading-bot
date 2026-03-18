"""
Production Trading Strategy
Combines ML predictions with momentum signals and risk management.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import pickle
import json

from src.backtesting.engine import BacktestConfig, run_backtest


FEATURE_COLS = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'zscore_20', 'volume_ratio_20',
    'return_lag_1', 'return_lag_2', 'return_lag_5',
    'volatility_lag_1', 'rsi_lag_1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]


@dataclass
class StrategyConfig:
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    signal_threshold: float = 0.0003
    hold_bars: int = 60
    min_signal_confidence: float = 0.3
    use_stop_loss: bool = True
    use_take_profit: bool = True


@dataclass
class BacktestResult:
    ticker: str
    pnl: float
    trades: int
    max_drawdown: float
    sharpe: float
    trades_log: list


class ProductionStrategy:
    def __init__(
        self,
        model_path: Path | None = None,
        config: StrategyConfig | None = None,
    ):
        self.model_path = model_path or Path('models/base/simple_model.pkl')
        self.config = config or StrategyConfig()
        self.model = None
        self.feature_cols = FEATURE_COLS
        self.is_loaded = False
        
    def load(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                if data.get('features'):
                    self.feature_cols = data.get('features')
        self.is_loaded = True
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 2, 5]:
            df[f'return_lag_{lag}'] = df.groupby('ticker')['return_1'].shift(lag).fillna(0)
            df[f'volatility_lag_{lag}'] = df.groupby('ticker')['rolling_volatility_20'].shift(lag).fillna(0)
            df[f'rsi_lag_{lag}'] = df.groupby('ticker')['rsi_14'].shift(lag).fillna(50)
        
        df['macd_momentum_interaction'] = df['macd'] * df['momentum_10']
        df['volume_volatility_interaction'] = df['volume_ratio_20'] * df['rolling_volatility_20']
        return df.fillna(0)
    
    def predict(self, row: dict) -> tuple[float, float]:
        if self.model is None:
            return 0.0, 0.0
        
        features = [row.get(c, 0) for c in self.feature_cols]
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        pred = float(self.model.predict([features])[0])
        confidence = min(1.0, abs(pred) * 50.0)
        return pred, confidence
    
    def run_backtest(
        self, 
        df: pd.DataFrame,
        ticker: str | None = None,
    ) -> BacktestResult:
        if ticker:
            df = df[df['ticker'] == ticker].copy()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        rows = df.to_dict('records')
        
        # Generate positions
        positions = []
        current_pos = 0.0
        counter = 0
        entry_price = 0.0
        
        for i, row in enumerate(rows):
            pred, confidence = self.predict(row)
            
            # Check stop loss / take profit
            if current_pos != 0 and entry_price > 0:
                price = row['close']
                if current_pos > 0:  # Long
                    pnl_pct = (price - entry_price) / entry_price
                else:  # Short
                    pnl_pct = (entry_price - price) / entry_price
                
                if self.config.use_stop_loss and pnl_pct <= -self.config.stop_loss_pct:
                    current_pos = 0.0
                    counter = 0
                elif self.config.use_take_profit and pnl_pct >= self.config.take_profit_pct:
                    current_pos = 0.0
                    counter = 0
            
            # Decide new position
            if counter == 0 and confidence >= self.config.min_signal_confidence:
                if pred > self.config.signal_threshold:
                    current_pos = 1.0
                    counter = self.config.hold_bars
                    entry_price = row['close']
                elif pred < -self.config.signal_threshold:
                    current_pos = -1.0
                    counter = self.config.hold_bars
                    entry_price = row['close']
            
            positions.append(current_pos)
            if counter > 0:
                counter -= 1
        
        # Apply positions to rows
        for r, pos in zip(rows, positions):
            r['policy_target_position'] = pos
        
        cfg = BacktestConfig(
            initial_cash=1_000_000,
            commission_bps=5,
            slippage_bps=5,
            lot_size=1,
            execution_delay_bars=1,
            position_size_pct=self.config.position_size_pct,
            target_position_column='policy_target_position',
            stop_loss_pct=self.config.stop_loss_pct if self.config.use_stop_loss else None,
            take_profit_pct=self.config.take_profit_pct if self.config.use_take_profit else None,
        )
        
        result = run_backtest(rows, cfg)
        
        # Calculate additional metrics
        equity = [e['equity'] for e in result['equity_curve']]
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        
        returns = [equity[i+1]/equity[i] - 1 for i in range(len(equity)-1)]
        if returns:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252*390) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return BacktestResult(
            ticker=ticker or 'ALL',
            pnl=result['summary']['pnl'],
            trades=len(result['trade_log']),
            max_drawdown=max_dd,
            sharpe=sharpe,
            trades_log=result['trade_log'],
        )
    
    def optimize(self, df: pd.DataFrame, ticker: str | None = None) -> dict:
        best_pnl = float('-inf')
        best_config = None
        results = []
        
        for hold in [30, 60, 120]:
            for threshold in [0.0001, 0.0003, 0.0005]:
                for min_conf in [0.2, 0.3, 0.4]:
                    cfg = StrategyConfig(
                        hold_bars=hold,
                        signal_threshold=threshold,
                        min_signal_confidence=min_conf,
                    )
                    self.config = cfg
                    
                    result = self.run_backtest(df, ticker)
                    results.append({
                        'hold': hold,
                        'threshold': threshold,
                        'min_conf': min_conf,
                        'pnl': result.pnl,
                        'trades': result.trades,
                        'sharpe': result.sharpe,
                    })
                    
                    if result.pnl > best_pnl:
                        best_pnl = result.pnl
                        best_config = cfg
        
        return {
            'best_config': best_config,
            'best_pnl': best_pnl,
            'all_results': results,
        }


def main():
    df = pd.read_parquet('data/processed/merged/test_expanded.parquet')
    
    strategy = ProductionStrategy()
    strategy.load()
    
    print('=== Testing Production Strategy ===')
    
    # Test on each ticker
    results = []
    for ticker in df['ticker'].unique():
        result = strategy.run_backtest(df, ticker)
        results.append(result)
        print(f'{ticker}: PnL={result.pnl:.0f}, Trades={result.trades}, Sharpe={result.sharpe:.2f}')
    
    total_pnl = sum(r.pnl for r in results)
    print(f'\\nTotal PnL: {total_pnl:.0f}')


if __name__ == '__main__':
    main()
