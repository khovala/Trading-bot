
# MOEX Sandbox Trading Platform (Phase 1 Skeleton)

Modular, sandbox-first architecture for MOEX equity trading with T-Invest API, staged retraining, MLflow tracking, and observability.

## Architecture Overview

- **Apps layer**: API, trader, trainer, backtester, notifier entrypoints.
- **Domain layer**: strict shared schemas and enums (normalized model output and auditable decisions).
- **Data/NLP/Features layer**: isolated ingestion and preprocessing for market/news pipelines.
- **Training layer**: stage abstraction + manifest-driven retraining (`pipeline.yaml`) + MLflow wrapper.
- **Execution boundary**: broker logic isolated from models/orchestrator.
- **Ops layer**: Docker Compose + Prometheus/Grafana/Alertmanager placeholders.

## Repository Tree (Target)

```text
apps/{api,trader,trainer,backtester,notifier}
src/
  config domain data features nlp models training orchestration risk execution backtesting monitoring utils
infra/{compose,prometheus,grafana,alertmanager,nginx}
data/{raw,interim,processed}
reports models artifacts tests scripts notebooks
params.yaml pipeline.yaml pyproject.toml .env.example README.md
```

## ADR-Style Decisions (Phase 1)

1. **Sandbox-first safety**: default is sandbox + dry-run; live trading requires explicit flag.
2. **Schema contract first**: all models must emit normalized prediction schema before orchestration.
3. **Manifest-driven pipeline**: all retraining/backtest steps declared in `pipeline.yaml` for reproducibility.
4. **MLflow mandatory**: every train/eval/backtest/promote stage logs params/metrics/artifacts.
5. **News as first-class subsystem**: dedicated ingestion adapters and dedicated news model path.
6. **Separation of concerns**: broker, orchestration, risk, and model code remain independent.
7. **Incremental delivery**: interfaces/stubs first, business logic in later phases.

## Pipeline and Params Design

- `pipeline.yaml` contains all 18 required stages with:
  - purpose
  - deps
  - outs
  - reports
  - mlflow logging behavior
- `params.yaml` centralizes:
  - environment/trading safety flags
  - market/news feature parameters
  - ensemble/risk settings
  - backtest assumptions
  - MLflow settings

## MLflow Integration Design

- `MLflowTracker` wrapper provides:
  - run context manager
  - `log_params`, `log_metrics`, `log_artifacts`
- Future phases add:
  - dataset + schema version tags
  - candidate/champion comparison tags
  - model registry integration

## MVP News Source Adapter Design

- Base interface: `NewsSourceAdapter.fetch(since_iso, limit) -> list[RawNewsItem]`.
- Source adapters are isolated per source (`src/nlp/sources/*` future files).
- MVP source priority:
  1. public RSS business/financial feeds
  2. issuer/corporate official feeds when available
  3. Telegram public channels in later phase

## Ticker/Entity Mapping Strategy (RU Issuers)

- Multi-step mapping pipeline (future implementation):
  1. alias dictionary (issuer legal names, short names, brand names)
  2. ticker lexicon (MOEX ticker, issuer, FIGI/ISIN where available)
  3. rule-based NER + fuzzy matching
  4. confidence score + fallback `unmapped`
- Base interface: `EntityMapper.map_news_item(item) -> TickerMappingResult`.

## First Commit File List (Phase 1)

- Root: `pyproject.toml`, `.env.example`, `params.yaml`, `pipeline.yaml`, `README.md`, `docker-compose.yml`
- Apps: `apps/api/main.py`, `apps/trainer/run_stage.py`
- Core src:
  - `src/config/settings.py`
  - `src/domain/enums.py`
  - `src/domain/schemas.py`
  - `src/training/pipeline/base.py`
  - `src/training/tracking/mlflow_utils.py`
  - `src/nlp/sources/base.py`
  - `src/nlp/entity_mapping/base.py`

## Quick Start (Local)

1. Copy env:
   - `cp .env.example .env`
2. Run API:
   - `docker compose up api`
3. Check:
   - `GET /health`
   - `GET /ready`
   - `GET /metrics`

## Phase 1 Scope Guardrails

- No model training implementation
- No broker integration implementation
- No backtesting implementation
- Interfaces and stubs only

# Trading-bot
ML for finances (OTUS course)
88676d9b4169d2e96049d6d99c24ff8a96c63bd4
