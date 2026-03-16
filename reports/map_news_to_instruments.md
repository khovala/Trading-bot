# Пояснение к отчету map_news_to_instruments

- Этап сопоставляет новости с тикерами/эмитентами.
- `run_id` — идентификатор прогона.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `metrics.input_item_count` — сколько новостей подано на маппинг.
- `metrics.mapped_item_count` — сколько новостей обработано маппером.
- `metrics.unmapped_item_count` — сколько новостей не удалось привязать к тикеру.
- `metrics.mapping_success_ratio` — доля успешного маппинга в диапазоне 0..1.
- `artifacts` — файл с результатом маппинга:
  - `data/processed/news/mapped_news.parquet`
