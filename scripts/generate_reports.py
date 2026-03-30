#!/usr/bin/env python3
"""Generate pipeline summary reports"""
import sys
sys.path.insert(0, '/opt/airflow')

import json
from pathlib import Path
from datetime import datetime

reports_dir = Path('/opt/airflow/reports')
reports_dir.mkdir(exist_ok=True)

models_dir = Path('/opt/airflow/models')
prod_dir = models_dir / 'production'

print('Generating pipeline reports...')

try:
    backtest_file = reports_dir / 'backtest_results.json'
    if backtest_file.exists():
        with open(backtest_file) as f:
            backtest = json.load(f)
    else:
        backtest = {'error': 'No backtest results found'}
    
    eval_file = reports_dir / 'model_evaluation.json'
    if eval_file.exists():
        with open(eval_file) as f:
            evaluation = json.load(f)
    else:
        evaluation = {'error': 'No evaluation results found'}
    
    params_file = reports_dir / 'best_params.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
    else:
        params = {'error': 'No parameters found'}
    
    promoted_models = []
    if prod_dir.exists():
        for f in prod_dir.glob('*.joblib'):
            promoted_models.append(f.name)
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'pipeline_status': 'success',
        'models_trained': ['base_model', 'news_model', 'ensemble_model'],
        'models_promoted': promoted_models,
        'backtest_summary': {
            'n_trades': backtest.get('n_trades', 'N/A'),
            'winrate': f"{backtest.get('winrate', 0):.1f}%" if isinstance(backtest.get('winrate'), (int, float)) else 'N/A',
            'total_pnl': f"{backtest.get('total_pnl', 0):.2f}" if isinstance(backtest.get('total_pnl'), (int, float)) else 'N/A',
        },
        'model_accuracy': evaluation.get('ensemble_accuracy', evaluation.get('base_model_accuracy', 'N/A')),
        'best_parameters': {
            'stop_loss': params.get('stop_loss_pct', 'N/A'),
            'take_profit': params.get('take_profit_pct', 'N/A'),
            'position_size': params.get('position_size_pct', 'N/A'),
        }
    }
    
    print('\nPipeline Summary:')
    print('=' * 50)
    print(f"Status: {summary['pipeline_status']}")
    print(f"Generated: {summary['generated_at']}")
    print(f"Models trained: {len(summary['models_trained'])}")
    print(f"Models promoted: {len(promoted_models)}")
    print(f"Win rate: {summary['backtest_summary']['winrate']}")
    print(f"Total PnL: {summary['backtest_summary']['total_pnl']}")
    print(f"Model accuracy: {summary['model_accuracy']}")
    
    summary_file = reports_dir / 'pipeline_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    import pandas as pd
    summary_df = pd.DataFrame([{
        'timestamp': summary['generated_at'],
        'status': summary['pipeline_status'],
        'n_models': len(summary['models_trained']),
        'n_promoted': len(promoted_models),
        'winrate': backtest.get('winrate', 0),
        'total_pnl': backtest.get('total_pnl', 0),
        'accuracy': evaluation.get('ensemble_accuracy', evaluation.get('base_model_accuracy', 0)),
    }])
    summary_csv = reports_dir / 'pipeline_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print(f'\nReports saved to {reports_dir}')
    print('Report generation complete')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
