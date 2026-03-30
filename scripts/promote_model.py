#!/usr/bin/env python3
"""Promote model to production"""
import sys
sys.path.insert(0, '/opt/airflow')

import os
import shutil
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

models_dir = Path('/opt/airflow/models')
prod_dir = models_dir / 'production'
prod_dir.mkdir(exist_ok=True)

print('Promoting model to production...')
print(f'MLflow tracking: {mlflow_uri}')

try:
    ensemble_path = models_dir / 'ensemble_model.joblib'
    base_model_path = models_dir / 'base_model.joblib'
    
    promoted_files = []
    
    if ensemble_path.exists():
        prod_path = prod_dir / 'ensemble_model.joblib'
        shutil.copy(ensemble_path, prod_path)
        print(f'Promoted: ensemble_model -> {prod_path}')
        promoted_files.append('ensemble_model')
    else:
        print('Ensemble model not found')
    
    if base_model_path.exists():
        prod_path = prod_dir / 'base_model.joblib'
        shutil.copy(base_model_path, prod_path)
        print(f'Promoted: base_model -> {prod_path}')
        promoted_files.append('base_model')
    
    promotion_info = {
        'promoted_at': datetime.now().isoformat(),
        'promoted_models': promoted_files,
        'status': 'success'
    }
    
    if MLFLOW_ENABLED:
        with mlflow.start_run(run_name='model_promotion'):
            mlflow.log_param('promoted_at', promotion_info['promoted_at'])
            mlflow.log_param('promoted_models', str(promoted_files))
            mlflow.log_param('status', 'success')
    
    info_file = prod_dir / 'promotion_info.json'
    import json
    with open(info_file, 'w') as f:
        json.dump(promotion_info, f, indent=2)
    
    print(f'\nPromotion info saved to {info_file}')
    print(f'Production directory: {prod_dir}')
    print('Model promotion complete')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
