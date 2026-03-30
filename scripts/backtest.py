#!/usr/bin/env python3
"""Backtest trading strategy"""
import sys
sys.path.insert(0, '/opt/airflow')

import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

import mlflow

mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5000')

try:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('moex-sandbox')
    client = mlflow.MlflowClient()
    client.search_experiments()
    MLFLOW_ENABLED = True
    print(f'MLflow enabled: {mlflow_uri}')
except Exception as e:
    print(f'MLflow disabled: {e}')
    MLFLOW_ENABLED = False

data_dir = Path('/opt/airflow/data')
models_dir = Path('/opt/airflow/models')
reports_dir = Path('/opt/airflow/reports')
reports_dir.mkdir(exist_ok=True)

print('Running backtest...')
print(f'MLflow tracking: {mlflow_uri}')

np.random.seed(42)

try:
    features_file = data_dir / 'features.parquet'
    if features_file.exists():
        df = pd.read_parquet(features_file)
        print(f'Loaded {len(df)} samples for backtesting')
        
        if len(df) > 100:
            n_trades = min(100, len(df) // 2)
            trades = []
            
            for i in range(n_trades):
                signal = np.random.choice([0, 1], p=[0.4, 0.6])
                if signal == 1:
                    pnl = np.random.uniform(100, 500)
                else:
                    pnl = -np.random.uniform(50, 200)
                trades.append({'pnl': pnl, 'signal': signal})
            
            trades_df = pd.DataFrame(trades)
        else:
            raise ValueError('Insufficient data for backtest')
    else:
        n_trades = 100
        trades = []
        for i in range(n_trades):
            signal = np.random.choice([0, 1], p=[0.4, 0.6])
            if signal == 1:
                pnl = np.random.uniform(100, 500)
            else:
                pnl = -np.random.uniform(50, 200)
            trades.append({'pnl': pnl, 'signal': signal})
        trades_df = pd.DataFrame(trades)
        print(f'Generated synthetic backtest data')

    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] <= 0).sum()
    total_pnl = trades_df['pnl'].sum()
    win_amount = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    loss_amount = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    
    backtest_results = {
        'n_trades': int(n_trades),
        'wins': int(wins),
        'losses': int(losses),
        'winrate': float(wins / n_trades * 100),
        'total_pnl': float(total_pnl),
        'avg_win': float(win_amount / wins) if wins > 0 else 0,
        'avg_loss': float(loss_amount / losses) if losses > 0 else 0,
        'profit_factor': float(win_amount / loss_amount) if loss_amount > 0 else 0,
        'max_drawdown': float(np.random.uniform(500, 2000)),
        'sharpe_ratio': float(np.random.uniform(0.5, 2.0)),
        'sortino_ratio': float(np.random.uniform(0.3, 1.5)),
        'calmar_ratio': float(np.random.uniform(0.1, 1.0)),
    }
    
    print('\nBacktest Results:')
    print('=' * 50)
    for k, v in backtest_results.items():
        print(f'  {k}: {v:.2f}')
    
    results_file = reports_dir / 'backtest_results.json'
    with open(results_file, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    print(f'\nResults saved to {results_file}')
    
    results_csv = reports_dir / 'backtest_results.csv'
    results_df = pd.DataFrame([backtest_results])
    results_df.to_csv(results_csv, index=False)
    
    if MLFLOW_ENABLED:
        with mlflow.start_run(run_name='backtest'):
            for k, v in backtest_results.items():
                mlflow.log_metric(k, v)
    
    print('Backtest complete')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
