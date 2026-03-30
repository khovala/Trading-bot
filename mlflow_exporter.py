#!/usr/bin/env python3
"""MLflow Prometheus metrics exporter - collects metrics from MLflow API"""
import os
import time
import requests
from prometheus_client import start_http_server, Gauge

MLFLOW_HOST = os.environ.get('MLFLOW_HOST', 'mlflow')
MLFLOW_PORT = os.environ.get('MLFLOW_PORT', '5000')
METRICS_PORT = int(os.environ.get('METRICS_PORT', '9092'))

experiments_count = Gauge('mlflow_experiments_count', 'Total number of experiments')
runs_active = Gauge('mlflow_runs_active', 'Number of active runs')
runs_completed = Gauge('mlflow_runs_completed', 'Number of completed runs')
runs_failed = Gauge('mlflow_runs_failed', 'Number of failed runs')
latest_run_accuracy = Gauge('mlflow_latest_run_accuracy', 'Accuracy of the latest run')
latest_run_loss = Gauge('mlflow_latest_run_loss', 'Loss of the latest run')
latest_run_auc = Gauge('mlflow_latest_run_auc', 'AUC of the latest run')

def collect_metrics():
    base_url = f'http://{MLFLOW_HOST}:{MLFLOW_PORT}'
    
    try:
        r = requests.get(f'{base_url}/api/2.0/mlflow/experiments/list', timeout=5)
        if r.status_code == 200:
            experiments = r.json().get('experiments', [])
            experiments_count.set(len(experiments))
    except Exception as e:
        print(f"Error collecting experiments: {e}")
    
    try:
        runs_active.set(0)
        runs_completed.set(0)
        runs_failed.set(0)
        
        r = requests.get(f'{base_url}/api/2.0/mlflow/experiments/list', timeout=5)
        if r.status_code == 200:
            experiments = r.json().get('experiments', [])
            for exp in experiments:
                exp_id = exp.get('experiment_id')
                try:
                    r2 = requests.get(
                        f'{base_url}/api/2.0/mlflow/runs/search',
                        params={'experiment_ids': [exp_id]},
                        timeout=5
                    )
                    if r2.status_code == 200:
                        runs = r2.json().get('runs', [])
                        for run in runs:
                            status = run.get('info', {}).get('status', '')
                            if status == 'RUNNING':
                                runs_active.inc()
                            elif status == 'FINISHED':
                                runs_completed.inc()
                            elif status == 'FAILED':
                                runs_failed.inc()
                                
                            metrics = run.get('data', {}).get('metrics', {})
                            if metrics:
                                accuracy = metrics.get('accuracy')
                                loss = metrics.get('loss')
                                auc = metrics.get('auc')
                                
                                if accuracy is not None:
                                    latest_run_accuracy.set(accuracy)
                                if loss is not None:
                                    latest_run_loss.set(loss)
                                if auc is not None:
                                    latest_run_auc.set(auc)
                except Exception as exp_e:
                    print(f"Error processing experiment {exp_id}: {exp_e}")
    except Exception as e:
        print(f"Error collecting runs: {e}")

if __name__ == '__main__':
    print(f'Starting MLflow exporter on port {METRICS_PORT}')
    print(f'Connecting to MLflow at {MLFLOW_HOST}:{MLFLOW_PORT}')
    start_http_server(METRICS_PORT)
    while True:
        collect_metrics()
        time.sleep(30)
