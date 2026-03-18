# Production Infrastructure Guide

## Overview

This document describes the production infrastructure setup for the MOEX trading platform on Yandex Cloud.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Yandex Cloud                                 │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  VM/IG      │  │ PostgreSQL   │  │ Object      │                │
│  │  (Trading)  │  │ (Managed)    │  │ Storage     │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                        │
│         └────────────────┼────────────────┘                        │
│                          │                                          │
│  ┌───────────────────────┼───────────────────────────────────────┐ │
│  │                 Docker Network                                  │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │ │
│  │  │  API    │  │ MLflow  │  │Airflow  │  │  Redis  │          │ │
│  │  │         │  │         │  │         │  │         │          │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └─────────┘          │ │
│  │       │            │            │                              │ │
│  │  ┌────┴────┐  ┌────┴────┐  ┌───┴────┐                       │ │
│  │  │Prometheus│  │Grafana  │  │Alertmgr │──► Telegram           │ │
│  │  └─────────┘  └─────────┘  └─────────┘                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | Trading API (FastAPI) |
| MLflow | 5000 | Experiment tracking |
| Airflow Webserver | 8080 | Pipeline orchestration UI |
| Grafana | 3000 | Dashboards |
| Prometheus | 9090 | Metrics collection |
| Alertmanager | 9093 | Alert routing |
| Nginx | 80/443 | Reverse proxy |

### Data Stores

- **PostgreSQL**: MLflow backend, Airflow metadata
- **Object Storage**: MLflow artifacts, model storage
- **Redis**: Airflow broker (Celery)

## Deployment

### Prerequisites

1. Yandex Cloud account with billing enabled
2. Terraform >= 1.5.0
3. Docker and docker-compose

### Setup Steps

#### 1. Create Service Account

```bash
yc iam service-account create --name trading-deployer
yc iam service-account create --name trading-vm-agent
```

#### 2. Grant Permissions

```bash
# For deployer
yc resource-manager folder add-access-binding <folder-id> \
  --role editor \
  --service-account-name trading-deployer

# For VM agent
yc resource-manager folder add-access-binding <folder-id> \
  --role editor \
  --service-account-name trading-vm-agent
```

#### 3. Create Static Keys

```bash
# For Terraform state storage
yc iam access-key create \
  --service-account-name trading-deployer \
  --description "Terraform state storage"

# Save the output key_id and secret
```

#### 4. Generate Service Account Key

```bash
yc iam key create \
  --service-account-name trading-deployer \
  --output service-account-key.json
```

#### 5. Create S3 Bucket

```bash
yc storage bucket create moex-trading-terraform
yc storage bucket create moex-trading-mlflow
```

#### 6. Initialize Terraform

```bash
cd terraform
terraform init
terraform plan -var-file=production.tfvars
terraform apply -var-file=production.tfvars
```

#### 7. Deploy Application

```bash
# SSH to VM
ssh ubuntu@<vm_public_ip>

# Clone repository
git clone <repo_url>
cd trading-bot

# Copy environment file
cp .env.example .env
# Edit .env with actual credentials

# Start services
docker-compose up -d
```

## Configuration

### Environment Variables

See `.env.example` for all available options.

Key variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `TINKOFF_TOKEN` | Tinkoff Invest API token | Yes |
| `TINKOFF_ACCOUNT_ID` | Trading account ID | Yes |
| `YC_CLOUD_ID` | Yandex Cloud ID | Yes |
| `YC_FOLDER_ID` | Yandex Folder ID | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | Yes (for alerts) |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | Yes (for alerts) |

### Tinkoff API Setup

1. Register at [Tinkoff Invest](https://www.tinkoff.ru/invest/)
2. Generate API token in settings
3. Create a sandbox account for testing
4. Get your account ID

### Telegram Bot Setup

1. Create bot via [@BotFather](https://t.me/BotFather)
2. Get bot token
3. Start chat with bot
4. Get your chat ID via @userinfobot

## Monitoring

### Prometheus Metrics

Key metrics exposed:

- `strategy_pnl_rub` - Current PnL in RUB
- `strategy_drawdown` - Current drawdown
- `strategy_sharpe` - Backtest Sharpe ratio
- `model_directional_accuracy` - Model accuracy
- `api_request_latency_seconds` - API latency
- `order_submission_total` - Order submissions

### Grafana Dashboards

Import dashboards from `infra/grafana/dashboards/`:

1. Trading Overview
2. Model Performance
3. Infrastructure Health

### Alert Rules

Alerts are sent to Telegram with severity levels:

- **Critical**: Pipeline failure, critical drawdown, API down
- **Warning**: High latency, low Sharpe, model accuracy drop

## Airflow Pipeline

The DAG `daily_trading_pipeline` runs at 6:00 AM Moscow time and executes:

1. Data collection (market + news)
2. Feature engineering
3. Model training
4. Backtesting
5. Evaluation & promotion

### Manual Trigger

```bash
airflow dags trigger daily_trading_pipeline
```

### View Logs

```bash
airflow tasks logs <task_id> daily_trading_pipeline <execution_date>
```

## Security

### Network

- All services run in a private VPC
- Only nginx exposed via NAT
- PostgreSQL accessible only from VM

### Secrets

- All secrets in `.env` file
- Never commit `.env` to git
- Use Yandex Lockbox for production secrets

### Backup

- PostgreSQL: Daily automated backups (Yandex Managed)
- Object Storage: Versioning enabled
- Terraform state: Remote in S3

## Troubleshooting

### Check Service Status

```bash
docker-compose ps
docker-compose logs <service>
```

### View Metrics

```bash
curl http://localhost:9090/metrics
curl http://localhost:8000/metrics
```

### Restart Service

```bash
docker-compose restart <service>
```

### Access Logs

```bash
docker-compose logs -f <service>
```

## Maintenance

### Update Application

```bash
git pull
docker-compose build
docker-compose up -d
```

### Update Models

Models are automatically loaded from `/app/models/`. To update:

1. Train new model in MLflow
2. Download from MLflow
3. Replace in `models/` directory
4. Restart API: `docker-compose restart api`

### Scale Airflow Workers

```bash
# Scale Celery workers
docker-compose up -d --scale airflow-worker=3
```

## Support

For issues, contact the trading team or check:

- MLflow: http://localhost:5000
- Grafana: http://localhost:3000
- Airflow: http://localhost:8080
