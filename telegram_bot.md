# Telegram Bot для торговой платформы MOEX

## Обзор

Telegram бот отправляет **отчеты с бизнес-метриками каждые 5 минут** во время торговой сессии. Отчеты включают:
- 💰 Баланс счета
- 📈 P&L (реализованный, нереализованный, общий)
- 🎯 Winrate (процент успешных сделок)
- 📊 Открытые позиции с P&L по каждой акции
- 🔄 Количество сделок BUY/SELL
- 🛑 Закрыто по Stop Loss
- 💎 Закрыто по Take Profit
- 💳 Накопленная комиссия
- 〰️ Накопленное проскальзывание

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading Session Monitor                       │
│                  (trading_session_monitor.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   T-Invest   │    │  Prometheus  │    │   Telegram   │   │
│  │     API      │    │   :8003      │    │     Bot      │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                   │                    │              │
│    Получает           Экспортирует          Отправляет         │
│    данные             метрики               отчеты            │
│    о позициях                                каждые           │
│    и балансе                                5 минут          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Компоненты

1. **trading_session_monitor.py** — основной скрипт мониторинга
2. **run_paper_trading.py** — торговый робот (собирает метрики)
3. **Docker контейнер** — trading-monitor (запускает мониторинг)

---

## Как работает мониторинг

### 1. Сбор данных из T-Invest API

При каждом запуске цикла (каждые 5 минут) мониторинг:

```python
# Подключается к API T-Invest
client = TInvestClient(TINKOFF_TOKEN)

# Получает список счетов
accounts = client.get_accounts()

# Получает портфель
portfolio = client.get_portfolio(account_id)

# Получает открытые позиции
positions = client.get_positions(account_id)
```

### 2. Чтение статистики из файла

Статистика торговой сессии хранится в файле `/tmp/paper_trading_session.json`:

```json
{
    "total_trades": 10,
    "won_trades": 7,
    "lost_trades": 3,
    "commission": 125.50,
    "slippage": 45.20,
    "cumulative_pnl": 3250.00,
    "buy_deals": 5,
    "sell_deals": 5,
    "sl_closed": 2,
    "tp_closed": 3
}
```

Этот файл обновляется торговым роботом при каждой сделке.

### 3. Расчет метрик

```python
# Нереализованный P&L (текущий)
unrealized_pnl = sum(position.pnl for position in positions)

# Общий P&L
total_pnl = cumulative_pnl + unrealized_pnl

# Winrate
winrate = (won_trades / total_trades) * 100
```

### 4. Экспорт метрик в Prometheus

Мониторинг запускает HTTP сервер на порту 8003 и экспортирует метрики:

```python
from prometheus_client import start_http_server, Gauge, Counter

MONITOR_BALANCE = Gauge("monitor_account_balance_rub", "Баланс счета")
MONITOR_TOTAL_PNL = Gauge("monitor_total_pnl_rub", "Общий P&L")
MONITOR_WINRATE = Gauge("monitor_winrate_percent", "Winrate %")
MONITOR_COMMISSION = Counter("monitor_commission_rub", "Комиссия")
```

Prometheus периодически опрашивает этот endpoint и сохраняет метрики.

### 5. Отправка отчета в Telegram

```python
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)
```

Формируется красивое сообщение с эмодзи и отправляется в чат.

---

## Настройка Telegram бота

### Шаг 1: Создание бота

1. Откройте Telegram
2. Найдите **@BotFather** (официальный бот для создания ботов)
3. Отправьте команду `/newbot`
4. Введите имя бота (например: `MOEX Trading Monitor`)
5. Введите username бота (должен оканчиваться на `bot`):
   - Пример: `moex_trading_bot`
   - Важно: username должен быть уникальным
6. **Сохраните токен** — он будет в формате:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

### Шаг 2: Получение Chat ID

1. Найдите созданного бота по username
2. Нажмите **START** (или отправьте `/start`)
3. Откройте **@userinfobot**
4. Нажмите **START**
5. Бот покажет ваш ID — это и есть `chat_id`
   - Пример: `123456789`

### Шаг 3: Настройка переменных окружения

Добавьте в файл `.env`:

```bash
# Токен бота (полученный от BotFather)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Ваш Chat ID (полученный от userinfobot)
TELEGRAM_CHAT_ID=123456789

# Токен T-Invest (для доступа к API)
TINKOFF_TOKEN=ваш_токен_tinvest
```

---

## Запуск мониторинга

### Вариант 1: Docker Compose (рекомендуется)

```bash
# Запуск мониторинга
docker compose up -d trading-monitor

# Просмотр логов
docker logs -f trading-monitor

# Остановка
docker compose stop trading-monitor
```

### Вариант 2: Напрямую (без Docker)

```bash
# Установка зависимостей
pip install requests urllib3 prometheus-client

# Запуск мониторинга
python trading_session_monitor.py
```

---

## Запуск торговли

Для работы мониторинга нужно запустить торгового робота:

```bash
# Однократный запуск
python run_paper_trading.py

# Или непрерывный режим (каждые 15 минут)
python run_paper_trading.py --continuous --interval 15
```

**Важно**: Торговый робот и мониторинг должны работать одновременно!

---

## Пример отчета в Telegram

```
📊 MOEX Trading Session Report
🕐 Время: 14:30:00

💰 Баланс счета: 1 250 000.00 RUB

📈 P&L:
   • Реализованный: 3 250.00 RUB
   • Нереализованный: -520.00 RUB
   • Общий: 2 730.00 RUB

🎯 Winrate: 7/10 = 70.0%

🔄 Сделки:
   • BUY: 5
   • SELL: 5
   • Закрыто по SL: 2
   • Закрыто по TP: 3

💳 Комиссия: 125.50 RUB
〰️ Проскальзывание: 45.20 RUB

📊 Открытые позиции:
   🟢 SBER: 100 шт @ 265.50 → 268.20 (+1.02%)
   🔴 NVTK: 50 шт @ 1420.00 → 1395.00 (-1.76%)
   🟢 LKOH: 20 шт @ 7450.00 → 7520.00 (+0.94%)
```

---

## Метрики Prometheus

Мониторинг экспортирует следующие метрики:

| Метрика | Тип | Описание |
|---------|-----|----------|
| `monitor_account_balance_rub` | Gauge | Баланс счета в рублях |
| `monitor_unrealized_pnl_rub` | Gauge | Нереализованный P&L |
| `monitor_cumulative_pnl_rub` | Gauge | Реализованный P&L |
| `monitor_total_pnl_rub` | Gauge | Общий P&L |
| `monitor_winrate_percent` | Gauge | Winrate в процентах |
| `monitor_total_trades` | Gauge | Всего закрытых сделок |
| `monitor_buy_deals_total` | Counter | Сделок BUY |
| `monitor_sell_deals_total` | Counter | Сделок SELL |
| `monitor_sl_closed_total` | Counter | Закрыто по SL |
| `monitor_tp_closed_total` | Counter | Закрыто по TP |
| `monitor_commission_rub` | Counter | Накопленная комиссия |
| `monitor_slippage_rub` | Counter | Накопленное проскальзывание |

---

## Устранение проблем

### Бот не отвечает

1. Проверьте токен бота:
   ```bash
   curl https://api.telegram.org/bot<TOKEN>/getMe
   ```

2. Проверьте chat_id:
   ```bash
   curl https://api.telegram.org/bot<TOKEN>/getUpdates
   ```

### Нет данных о позициях

1. Убедитесь что запущен `run_paper_trading.py`
2. Проверьте токен T-Invest
3. Проверьте логи: `docker logs trading-monitor`

### Ошибка "No accounts found"

1. Сначала запустите `run_paper_trading.py` для создания счета в песочнице
2. Дождитесь пока робот создаст позиции

---

## Файлы проекта

- `trading_session_monitor.py` — основной скрипт мониторинга
- `run_paper_trading.py` — торговый робот
- `Dockerfile.monitor` — Docker образ для мониторинга
- `docker-compose.yml` — конфигурация сервисов

---

## Расписание

- Мониторинг отправляет отчеты **каждые 5 минут**
- Интервал настраивается в `trading_session_monitor.py`:
  ```python
  CHECK_INTERVAL = 300  # секунд (5 минут)
  ```
