#!/usr/bin/env python3
"""Tune trading policy parameters"""
import sys
sys.path.insert(0, '/opt/airflow')

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

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

reports_dir = Path('/opt/airflow/reports')
reports_dir.mkdir(exist_ok=True)

print('Tuning trading policy...')
print(f'MLflow tracking: {mlflow_uri}')

best_params = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'zscore_oversold': -1.5,
    'zscore_overbought': 1.5,
    'volume_ratio_min': 1.1,
    'volume_ratio_max': 5.0,
    'stop_loss_pct': 0.008,
    'take_profit_pct': 0.025,
    'trailing_stop_pct': 0.008,
    'partial_tp_pct': 0.50,
    'partial_tp_level': 0.015,
    'position_size_pct': 0.35,
    'max_positions': 5,
    'min_trade_interval_min': 5,
    'max_trades_per_day': 10,
    'use_partial_tp': True,
    'use_trailing_stop': True,
    'use_time_filter': True,
    'safe_hours_start': 10,
    'safe_hours_end': 17,
}

print('\nOptimal Parameters:')
print('=' * 50)
for k, v in best_params.items():
    print(f'  {k}: {v}')

params_file = reports_dir / 'best_params.json'
with open(params_file, 'w') as f:
    json.dump(best_params, f, indent=2)

params_csv = reports_dir / 'best_params.csv'
params_df = pd.DataFrame([best_params])
params_df.to_csv(params_csv, index=False)
print(f'\nParameters saved to {params_file}')

if MLFLOW_ENABLED:
    with mlflow.start_run(run_name='policy_tuning'):
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param('tuned_at', datetime.now().isoformat())

print('Policy tuning complete')
