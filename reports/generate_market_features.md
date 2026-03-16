# Пояснение к отчету generate_market_features

- Этап генерирует рыночные признаки из свечей.
- `run_id` — идентификатор запуска.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `metrics.ticker_count` — число тикеров, для которых посчитаны фичи.
- `metrics.market_feature_row_count` — число строк в таблице рыночных признаков.
- `artifacts` — файл с рыночными фичами:
  - `data/processed/market/features/market_features.parquet`
