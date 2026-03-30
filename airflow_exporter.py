#!/usr/bin/env python3
"""Airflow Prometheus metrics exporter"""
import os
import time
import requests
from prometheus_client import start_http_server, Gauge

AIRFLOW_HOST = os.environ.get('AIRFLOW_HOST', 'airflow-webserver')
AIRFLOW_PORT = os.environ.get('AIRFLOW_PORT', '8080')
METRICS_PORT = int(os.environ.get('METRICS_PORT', '9091'))

dagrun_count = Gauge('airflow_dagrun_count', 'Total DAG runs')
dagrun_running = Gauge('airflow_dagrun_running', 'Running DAG runs')
dagrun_success = Gauge('airflow_dagrun_success', 'Successful DAG runs')
dagrun_failed = Gauge('airflow_dagrun_failed', 'Failed DAG runs')
task_count = Gauge('airflow_task_count', 'Total tasks')
task_running = Gauge('airflow_task_running', 'Running tasks')
task_success = Gauge('airflow_task_success', 'Successful tasks')
task_failed = Gauge('airflow_task_failed', 'Failed tasks')
scheduler_heartbeat = Gauge('airflow_scheduler_heartbeat_unix', 'Scheduler heartbeat timestamp')

def collect_metrics():
    try:
        r = requests.get(f'http://{AIRFLOW_HOST}:{AIRFLOW_PORT}/api/v1/dags', 
                        auth=('airflow', 'airflow'), timeout=5)
        if r.status_code == 200:
            dags = r.json().get('dags', [])
            dagrun_count.set(len(dags))
    except Exception as e:
        print(f"Error collecting DAGs: {e}")

    try:
        r = requests.get(f'http://{AIRFLOW_HOST}:{AIRFLOW_PORT}/api/v1/dagRuns',
                        auth=('airflow', 'airflow'), timeout=5)
        if r.status_code == 200:
            runs = r.json().get('dag_runs', [])
            running = sum(1 for r in runs if r.get('state') == 'running')
            success = sum(1 for r in runs if r.get('state') == 'success')
            failed = sum(1 for r in runs if r.get('state') == 'failed')
            dagrun_running.set(running)
            dagrun_success.set(success)
            dagrun_failed.set(failed)
    except Exception as e:
        print(f"Error collecting DAG runs: {e}")

    try:
        r = requests.get(f'http://{AIRFLOW_HOST}:{AIRFLOW_PORT}/health',
                        timeout=5)
        if r.status_code == 200:
            health = r.json()
            scheduler = health.get('scheduler', {})
            hb = scheduler.get('latest_scheduler_heartbeat')
            if hb:
                from datetime import datetime
                dt = datetime.fromisoformat(hb.replace('Z', '+00:00'))
                scheduler_heartbeat.set(dt.timestamp())
    except Exception as e:
        print(f"Error collecting scheduler health: {e}")

if __name__ == '__main__':
    print(f'Starting Airflow exporter on port {METRICS_PORT}')
    start_http_server(METRICS_PORT)
    while True:
        collect_metrics()
        time.sleep(15)
