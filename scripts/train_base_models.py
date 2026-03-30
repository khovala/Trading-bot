#!/usr/bin/env python3
"""Train base ML models"""
import sys
sys.path.insert(0, '/opt/airflow')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

import mlflow
from mlflow.sklearn import log_model

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
models_dir.mkdir(exist_ok=True)

print('Training base models...')
print(f'MLflow tracking: {mlflow_uri}')

try:
    features_file = data_dir / 'features.parquet'
    if features_file.exists():
        df = pd.read_parquet(features_file)
        print(f'Loaded {len(df)} features')
        
        if len(df) > 50 and 'returns' in df.columns:
            df = df.dropna()
            
            feature_cols = ['returns', 'log_returns', 'rsi_14', 'zscore', 'volume_ratio']
            available_cols = [c for c in feature_cols if c in df.columns]
            
            X = df[available_cols].fillna(0)
            y = (df['returns'] > 0).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            print(f'Training on {len(X_train)} samples...')
            
            model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f'Model accuracy: {accuracy:.4f}')
            
            model_path = models_dir / 'base_model.joblib'
            joblib.dump(model, model_path)
            print(f'Model saved to {model_path}')
            
            if MLFLOW_ENABLED:
                with mlflow.start_run(run_name='base_model_training'):
                    mlflow.log_param('n_samples', len(X_train))
                    mlflow.log_param('n_features', len(available_cols))
                    mlflow.log_param('accuracy', accuracy)
                    mlflow.sklearn.log_model(model, 'base_model')
            
            print('Base model training complete')
        else:
            raise ValueError('Insufficient data or missing returns column')
    else:
        raise FileNotFoundError('Features file not found')

except Exception as e:
    print(f'Using synthetic data due to: {e}')
    
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.random.randint(0, 2, 500)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Synthetic model accuracy: {accuracy:.4f}')
    
    model_path = models_dir / 'base_model.joblib'
    joblib.dump(model, model_path)
    print(f'Synthetic model saved to {model_path}')
    
    if MLFLOW_ENABLED:
        with mlflow.start_run(run_name='base_model_synthetic'):
            mlflow.log_param('type', 'synthetic')
            mlflow.log_param('accuracy', accuracy)
            mlflow.sklearn.log_model(model, 'base_model')
    
    print('Base model training complete (synthetic data)')
