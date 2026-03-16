# Пояснение к отчету download_market_data

- Этап загружает сырые рыночные данные (инструменты и свечи) в raw-слой.
- `run_id` — уникальный идентификатор прогона этой стадии.
- `stage_name` — название этапа пайплайна.
- `success` — успешность выполнения стадии.
- `metrics.instrument_count` — сколько инструментов выгружено.
- `metrics.ticker_count` — сколько тикеров было запрошено.
- `metrics.candle_count` — общее число загруженных свечей.
- `artifacts` — пути, где лежит результат:
  - `data/raw/market/instruments.json`
  - `data/raw/market/candles`
