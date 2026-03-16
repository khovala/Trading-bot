# Пояснение к отчету preprocess_market_data

- Этап валидирует и нормализует рыночные свечи.
- `run_id` — идентификатор запуска стадии.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `metrics.instrument_count` — число инструментов после валидации.
- `metrics.ticker_count` — число обработанных тикеров.
- `metrics.duplicate_bar_count` — количество найденных дублей свечей.
- `metrics.missing_bar_count` — количество пропусков временных баров.
- `artifacts` — выходные данные interim-слоя:
  - `data/interim/market/instruments.json`
  - `data/interim/market/candles`
