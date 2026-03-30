#!/usr/bin/env python3
"""Simple trainer service with Prometheus metrics"""
from prometheus_client import start_http_server, Gauge
import time
import os

trainer_model_accuracy = Gauge('trainer_model_accuracy', 'Current model accuracy')
trainer_model_loss = Gauge('trainer_model_loss', 'Current model loss')
trainer_epochs_total = Gauge('trainer_epochs_total', 'Total training epochs')
trainer_samples_total = Gauge('trainer_samples_total', 'Total training samples')

trainer_model_accuracy.set(0.75)
trainer_model_loss.set(0.25)
trainer_epochs_total.set(100)
trainer_samples_total.set(10000)

if __name__ == '__main__':
    port = int(os.environ.get('METRICS_PORT', 8002))
    print(f'Starting trainer metrics server on port {port}')
    start_http_server(port)
    while True:
        time.sleep(60)
