# Пояснение к отчету preprocess_news_data

- Этап очищает и нормализует новости перед маппингом на инструменты.
- `run_id` — идентификатор запуска.
- `stage_name` — имя стадии.
- `success` — статус выполнения этапа.
- `metrics.input_item_count` — сколько записей пришло на вход.
- `metrics.deduped_item_count` — сколько осталось после дедупликации.
- `metrics.processed_item_count` — сколько записей прошло полную обработку.
- `artifacts` — выходной файл:
  - `data/interim/news/items.jsonl`
