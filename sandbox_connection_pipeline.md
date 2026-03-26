# Подключение к T-Invest Sandbox через API

## Введение

Данный документ описывает процесс подключения к песочнице (sandbox) T-Invest API для тестирования торговой стратегии без использования реальных денег.

**Песочница (Sandbox)** — это тестовый контур T-Invest API, который позволяет:
- Тестировать торговых роботов без риска потери реальных средств
- Создавать виртуальные счета с виртуальными деньгами
- Отрабатывать алгоритмы в условиях, приближенных к реальным

---

## Шаг 1: Получение API токена

### Что такое API токен?

API токен — это уникальный ключ авторизации, который позволяет вашему приложению получить доступ к вашему аккаунту T-Invest. Токен генерируется в личном кабинете и является секретным — **его никогда нельзя публиковать**.

### Как получить токен:

1. Зайдите на сайт [https://www.tinkoff.ru/invest/](https://www.tinkoff.ru/invest/)
2. Авторизуйтесь в своём аккаунте
3. Перейдите в раздел **Настройки** → **Настройки безопасности**
4. Найдите раздел **Токен T-Invest API**
5. Нажмите **Выпустить токен** (или **Создать токен**)
6. **Важно**: Токен отображается только один раз! Сразу сохраните его в безопасное место
7. При необходимости выберите доступы (чтение данных, торговля и т.д.)
8. Подтвердите создание токена

### В нашем проекте:

```python
TOKEN = 't.9kPmBnJM7bBln56Nhtj_a3iu-aajTyoArtYKam7J_bmob_7jQXbQVzo4N2X9hyhwN-HyMyyzjQhS2YPPoZ4Owg'
```

**Внимание**: Это учебный токен. В реальном проекте храните токен в переменных окружения или файле `.env`:

```python
import os
TOKEN = os.environ.get('TINKOFF_TOKEN')
```

---

## Шаг 2: Выбор правильного API endpoints

### Важно: API T-Invest перешёл на новый формат

Исторически T-Invest использовал REST API по адресу `api-invest.tinkoff.ru`. Однако в 2024-2025 году произошёл переход на новую архитектуру:

| Тип | Старый API | Новый API |
|-----|------------|-----------|
| **Production** | `https://api-invest.tinkoff.ru` | `https://invest-public-api.tbank.ru` |
| **Sandbox** | `https://sandbox-api-invest.tinkoff.ru` | `https://sandbox-invest-public-api.tbank.ru` |

### Формат запросов

Новый API использует:
- **gRPC** для высокопроизводительного взаимодействия
- **REST-обёртку** над gRPC сервисами
- **Path**: `/rest/tinkoff.public.invest.api.contract.v1.{ServiceName}/{MethodName}`

Пример:
```
POST https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxAccounts
```

### Заголовки (Headers)

Каждый запрос должен содержать:
```python
HEADERS = {
    'Authorization': 'Bearer {YOUR_TOKEN}',  # Токен авторизации
    'Content-Type': 'application/json'        # Формат данных
}
```

---

## Шаг 3: Настройка SSL/TLS

### Проблема с самоподписанными сертификатами

Сервер песочницы использует самоподписанный SSL-сертификат. По умолчанию Python (urllib3/requests) отклоняет такие соединения для безопасности.

### Решение: отключение проверки SSL

Для тестирования можно отключить проверку сертификатов:

```python
import urllib3
import requests

# Отключаем предупреждения о небезопасном соединении
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Создаём сессию с отключённой проверкой
session = requests.Session()
session.verify = False  # Не проверять SSL сертификаты

# Теперь можно делать запросы
response = session.post(url, headers=headers, json=data, verify=False)
```

**Внимание**: В продакшене (с реальными деньгами) так делать нельзя! Нужно использовать правильные сертификаты или прокси.

---

## Шаг 4: Создание sandbox аккаунта

### Если аккаунт уже существует

Если вы уже создавали sandbox ранее, получите его ID:

```python
import requests

def get_sandbox_accounts(token):
    """Получение списка sandbox аккаунтов"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxAccounts"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, json={}, verify=False)
    
    if response.status_code == 200:
        data = response.json()
        accounts = data.get('accounts', [])
        return accounts
    
    return []

# Использование
accounts = get_sandbox_accounts(TOKEN)
for account in accounts:
    print(f"ID: {account['id']}")
    print(f"Тип: {account['type']}")
    print(f"Статус: {account['status']}")
```

**Результат:**
```json
{
  "accounts": [{
    "id": "8a7bddc0-2f49-4556-8af7-050c977d68d6",
    "type": "ACCOUNT_TYPE_TINKOFF",
    "status": "ACCOUNT_STATUS_OPEN"
  }]
}
```

### Если аккаунта нет — создаём новый

```python
def open_sandbox_account(token, name="Trading Bot"):
    """Создание нового sandbox аккаунта"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/OpenSandboxAccount"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"name": name}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('accountId')
    
    return None

# Создание аккаунта
account_id = open_sandbox_account(TOKEN, "My Trading Bot")
print(f"Создан аккаунт: {account_id}")
```

---

## Шаг 5: Пополнение sandbox счёта

По умолчанию sandbox счёт пустой. Добавим виртуальных денег:

```python
def sandbox_pay_in(token, account_id, amount_rub=100000):
    """Пополнение sandbox счёта"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/SandboxPayIn"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "accountId": account_id,
        "amount": {
            "currency": "RUB",
            "units": str(amount_rub),
            "nano": 0
        }
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None

# Пополняем счёт на 1 000 000 рублей
result = sandbox_pay_in(TOKEN, account_id, 1000000)
print(f"Счёт пополнен: {result}")
```

---

## Шаг 6: Получение информации об инструментах

### Поиск тикера (акции)

Для работы с акциями нам нужен FIGI — уникальный идентификатор инструмента:

```python
def find_instrument(token, query):
    """Поиск инструмента по тикеру или названию"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"query": query}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        result = response.json()
        instruments = result.get('instruments', [])
        if instruments:
            inst = instruments[0]
            return {
                'figi': inst['figi'],
                'ticker': inst['ticker'],
                'name': inst['name'],
                'lot': inst.get('lot', 1),
                'currency': inst.get('currency', 'RUB')
            }
    
    return None

# Пример поиска
nvtk = find_instrument(TOKEN, 'NVTK')
print(f"NVTK: FIGI={nvtk['figi']}, Lot={nvtk['lot']}")
```

**Результат:**
```
NVTK: FIGI=BBG000QY6FL3, Lot=1
```

### Список поддерживаемых тикеров

В нашей стратегии используются:

| Тикер | FIGI | Название |
|-------|------|----------|
| NVTK | BBG000QY6FL3 | Новатэк |
| YNDX | TCS109805522 | Яндекс |
| TATN | BBG004RVFFC0 | Татнефть |
| MGNT | BBG004RVFCY3 | Магнит |
| POLY | FUTPOLY03220 | Полюс |

---

## Шаг 7: Получение рыночных данных

### Получение текущей цены (Last Price)

```python
def get_last_price(token, figi):
    """Получение последней известной цены"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"figi": [figi]}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        result = response.json()
        prices = result.get('lastPrices', [])
        for price in prices:
            if price['figi'] == figi:
                units = price['price']['units']
                nano = price['price']['nano']
                return float(f"{units}.{nano:09d}")
    
    return None

# Получение цены
price = get_last_price(TOKEN, 'BBG000QY6FL3')
print(f"NVTK: {price} руб.")
```

### Получение свечей (исторические данные)

```python
from datetime import datetime, timezone, timedelta

def get_candles(token, figi, from_time, to_time, interval='CANDLE_INTERVAL_5_MIN'):
    """Получение свечей за период"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "figi": figi,
        "from": from_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        "to": to_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        "interval": interval
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('candles', [])
    
    return []

# Пример: получить свечи за последний час
now = datetime.now(timezone.utc)
from_time = now - timedelta(hours=1)

candles = get_candles(TOKEN, 'BBG000QY6FL3', from_time, now, 'CANDLE_INTERVAL_5_MIN')
print(f"Получено {len(candles)} свечей")
```

**Доступные интервалы свечей:**
- `CANDLE_INTERVAL_1_MIN` — 1 минута (максимум 1 день)
- `CANDLE_INTERVAL_5_MIN` — 5 минут (максимум 1 день)
- `CANDLE_INTERVAL_15_MIN` — 15 минут (максимум 1 день)
- `CANDLE_INTERVAL_HOUR` — 1 час (максимум 7 дней)
- `CANDLE_INTERVAL_DAY` — 1 день (максимум 1 год)

---

## Шаг 8: Получение портфеля и позиций

### Просмотр портфеля

```python
def get_portfolio(token, account_id):
    """Получение портфеля"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxPortfolio"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"accountId": account_id}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None

# Получение портфеля
portfolio = get_portfolio(TOKEN, account_id)
print(f"Позиции: {len(portfolio.get('positions', []))}")
```

### Просмотр открытых позиций

```python
def get_positions(token, account_id):
    """Получение открытых позиций"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxPositions"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"accountId": account_id}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None
```

---

## Шаг 9: Размещение ордера

### Покупка акций

```python
def post_order(token, account_id, figi, quantity, direction='ORDER_DIRECTION_BUY'):
    """Размещение ордера"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/PostSandboxOrder"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "accountId": account_id,
        "figi": figi,
        "direction": direction,           # ORDER_DIRECTION_BUY или ORDER_DIRECTION_SELL
        "quantity": str(quantity),        # Количество лотов (строка!)
        "orderType": "ORDER_TYPE_MARKET"  # Рыночный ордер (исполнить по текущей цене)
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None

# Пример: купить 100 акций Новатэка
result = post_order(TOKEN, account_id, 'BBG000QY6FL3', 100, 'ORDER_DIRECTION_BUY')
print(f"Ордер размещён: {result.get('orderId')}")
```

### Типы ордеров

| Тип | Описание |
|-----|----------|
| `ORDER_TYPE_MARKET` | Рыночный — исполняется по текущей рыночной цене |
| `ORDER_TYPE_LIMIT` | Лимитный — исполняется по указанной цене или лучше |

### Направление

| Направление | Описание |
|-------------|----------|
| `ORDER_DIRECTION_BUY` | Покупка |
| `ORDER_DIRECTION_SELL` | Продажа |

---

## Шаг 10: Управление ордерами

### Получение активных ордеров

```python
def get_orders(token, account_id):
    """Получение списка активных ордеров"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxOrders"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {"accountId": account_id}
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None

orders = get_orders(TOKEN, account_id)
print(f"Активных ордеров: {len(orders.get('orders', []))}")
```

### Отмена ордера

```python
def cancel_order(token, account_id, order_id):
    """Отмена ордера"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/CancelSandboxOrder"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "accountId": account_id,
        "orderId": order_id
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    return response.status_code == 200

# Отмена ордера
success = cancel_order(TOKEN, account_id, 'order-id-here')
print(f"Ордер отменён: {success}")
```

---

## Шаг 11: Получение истории операций

```python
def get_operations(token, account_id, from_time, to_time):
    """Получение истории операций"""
    url = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxOperations"
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "accountId": account_id,
        "from": from_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        "to": to_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code == 200:
        return response.json()
    
    return None

# Получить операции за последнюю неделю
from_time = datetime.now(timezone.utc) - timedelta(days=7)
to_time = datetime.now(timezone.utc)

operations = get_operations(TOKEN, account_id, from_time, to_time)
print(f"Операций: {len(operations.get('operations', []))}")
```

---

## Полный пример: клиент для работы с API

Ниже представлен готовый класс `TInvestClient`, который объединяет все операции:

```python
import requests
import urllib3
from datetime import datetime, timezone, timedelta
from typing import Optional

class TInvestClient:
    """Клиент для работы с T-Invest Sandbox API"""
    
    def __init__(self, token: str, use_sandbox: bool = True):
        self.token = token
        self.api_base = 'https://sandbox-invest-public-api.tbank.ru/rest' if use_sandbox else 'https://invest-public-api.tbank.ru/rest'
        
        # Настройка сессии с отключённой проверкой SSL (для sandbox)
        self.session = requests.Session()
        self.session.verify = False
        urllib3.disable_warnings()
        
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def _call(self, endpoint: str, data: dict = None) -> Optional[dict]:
        """Универсальный метод для API вызовов"""
        url = f"{self.api_base}/tinkoff.public.invest.api.contract.v1.{endpoint}"
        
        try:
            response = self.session.post(url, headers=self.headers, json=data or {}, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    # === Аккаунты ===
    
    def get_accounts(self):
        """Получение списка sandbox аккаунтов"""
        result = self._call('SandboxService/GetSandboxAccounts', {})
        return result.get('accounts', []) if result else []
    
    def open_account(self, name: str = "Trading Bot"):
        """Создание нового sandbox аккаунта"""
        result = self._call('SandboxService/OpenSandboxAccount', {"name": name})
        return result.get('accountId') if result else None
    
    # === Портфель и позиции ===
    
    def get_portfolio(self, account_id: str):
        """Получение портфеля"""
        return self._call('SandboxService/GetSandboxPortfolio', {"accountId": account_id})
    
    def get_positions(self, account_id: str):
        """Получение позиций"""
        return self._call('SandboxService/GetSandboxPositions', {"accountId": account_id})
    
    # === Инструменты ===
    
    def find_instrument(self, query: str):
        """Поиск инструмента по тикеру"""
        result = self._call('InstrumentsService/FindInstrument', {"query": query})
        instruments = result.get('instruments', []) if result else []
        if instruments:
            inst = instruments[0]
            return {
                'figi': inst['figi'],
                'ticker': inst['ticker'],
                'name': inst['name'],
                'lot': inst.get('lot', 1)
            }
        return None
    
    # === Рыночные данные ===
    
    def get_last_price(self, figi: str) -> Optional[float]:
        """Получение последней цены"""
        result = self._call('MarketDataService/GetLastPrices', {"figi": [figi]})
        if result:
            for price in result.get('lastPrices', []):
                if price['figi'] == figi:
                    units = price['price']['units']
                    nano = price['price']['nano']
                    return float(f"{units}.{nano:09d}")
        return None
    
    def get_candles(self, figi: str, from_time: datetime, to_time: datetime, 
                   interval: str = 'CANDLE_INTERVAL_5_MIN'):
        """Получение свечей"""
        result = self._call('MarketDataService/GetCandles', {
            "figi": figi,
            "from": from_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            "to": to_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            "interval": interval
        })
        return result.get('candles', []) if result else []
    
    # === Ордера ===
    
    def post_order(self, account_id: str, figi: str, direction: str, 
                  quantity: int, order_type: str = 'ORDER_TYPE_MARKET') -> Optional[dict]:
        """Размещение ордера"""
        return self._call('SandboxService/PostSandboxOrder', {
            "accountId": account_id,
            "figi": figi,
            "direction": direction,
            "quantity": str(quantity),
            "orderType": order_type
        })
    
    def get_orders(self, account_id: str):
        """Получение активных ордеров"""
        return self._call('SandboxService/GetSandboxOrders', {"accountId": account_id})
    
    def cancel_order(self, account_id: str, order_id: str) -> bool:
        """Отмена ордера"""
        result = self._call('SandboxService/CancelSandboxOrder', {
            "accountId": account_id,
            "orderId": order_id
        })
        return result is not None


# === Пример использования ===

if __name__ == "__main__":
    TOKEN = "ваш_токен_здесь"
    
    # Создаём клиент
    client = TInvestClient(TOKEN)
    
    # Получаем аккаунт
    accounts = client.get_accounts()
    if accounts:
        account_id = accounts[0]['id']
        print(f"Используем аккаунт: {account_id}")
        
        # Получаем портфель
        portfolio = client.get_portfolio(account_id)
        print(f"Позиций в портфеле: {len(portfolio.get('positions', []))}")
        
        # Ищем инструмент
        nvtk = client.find_instrument('NVTK')
        if nvtk:
            print(f"Нашли: {nvtk['name']} (FIGI: {nvtk['figi']})")
            
            # Получаем цену
            price = client.get_last_price(nvtk['figi'])
            print(f"Цена: {price} руб.")
```

---

## Ограничения Sandbox

Важно знать ограничения тестовой среды:

1. **Ограниченные исторические данные** — свечи доступны только за короткий период
2. **Задержка данных** — цены могут отличаться от реальных
3. **Не все инструменты** — некоторые инструменты могут быть недоступны
4. **Виртуальные деньги** — пополнение ограничено лимитами
5. **Сохранность данных** — sandbox может быть очищен

---

## Результаты экспериментов с Sandbox

### Тестирование доступности исторических данных (25 марта 2026)

Мы протестировали 30+ популярных российских акций на доступность свечных данных в песочнице.

**Результаты:**

| Тикер | Свечей за 1 час | FIGI | Статус |
|-------|-----------------|------|--------|
| TATN | 60 | BBG004RVFFC0 | ✅ Есть данные |
| MGNT | 60 | BBG004RVFCY3 | ✅ Есть данные |
| MAGN | 28 | FUTMAGN06260 | ✅ Есть данные |
| SELG | 59 | BBG002458LF8 | ✅ Есть данные |
| MTSS | 19 | TCS007775219 | ✅ Есть данные |
| APTK | 37 | BBG000K3STR7 | ✅ Есть данные |
| KMAZ | 1 | FKMAZ0926000 | ⚠️ Минимум данных |
| MRKZ | 13 | BBG000TJ6F42 | ⚠️ Минимум данных |
| OGKB | 46 | BBG000RK52V1 | ✅ Есть данные |
| TATNP | 58 | BBG004S68829 | ✅ Есть данные |
| TGKA | 28 | BBG000QFH687 | ✅ Есть данные |
| TGKN | 36 | BBG000RG4ZQ4 | ✅ Есть данные |
| POSI | 46 | FPOSI0626000 | ✅ Есть данные |
| SBER | 0 | - | ❌ Нет данных |
| GAZP | 0 | - | ❌ Нет данных |
| LKOH | 0 | - | ❌ Нет данных |
| SNGS | 0 | - | ❌ Нет данных |
| NVTK | 0 | - | ❌ Нет данных |
| YNDX | 0 | - | ❌ Нет данных |
| POLY | 0 | - | ❌ Нет данных |

**Вывод:** Только ~20 тикеров имеют исторические данные в sandbox. Большинство популярных акций (SBER, GAZP, LKOH, YNDX, NVTK) **недоступны** для получения свечей.

---

### Тестирование торговли через API

Мы проверили какие инструменты доступны для торговли через API в песочнице:

**Результаты (flags `apiTradeAvailableFlag`):**

| Тикер | API Торговля | FIGI |
|-------|--------------|------|
| TATN | ✅ True | BBG004RVFFC0 |
| MGNT | ✅ True | BBG004RVFCY3 |
| MAGN | ✅ True | FUTMAGN06260 |
| SELG | ✅ True | BBG002458LF8 |
| APTK | ✅ True | BBG000K3STR7 |
| KMAZ | ✅ True | FKMAZ0926000 |
| MRKZ | ✅ True | BBG000TJ6F42 |
| MRKS | ✅ True | BBG000VJMH65 |
| KROT | ✅ True | BBG000NLB2G3 |
| MSTT | ✅ True | BBG004S68DD6 |
| OGKB | ✅ True | BBG000RK52V1 |
| PRFN | ✅ True | TCS00A0JNXF9 |
| TATNP | ✅ True | BBG004S68829 |
| TGKA | ✅ True | BBG000QFH687 |
| TGKB | ✅ True | BBG000Q7GJ60 |
| TGKN | ✅ True | BBG000RG4ZQ4 |
| POSI | ✅ True | FPOSI0626000 |
| MTSS | ❌ False | TCS007775219 |
| DIOD | ❌ False | BBG000R0L782 |
| KCHE | ❌ False | BG000J5X8M9 |

**Вывод:** 17 из 20 протестированных тикеров доступны для торговли через API.

---

### Ошибки при размещении ордеров

**Тестовая сделка (успешная):**

```json
{
  "orderId": "0cc5730e-92cb-425d-94a7-f0dd4783cbda",
  "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
  "lotsRequested": "10",
  "lotsExecuted": "10",
  "executedOrderPrice": {"currency": "rub", "units": "650", "nano": 100000000},
  "totalOrderAmount": {"currency": "rub", "units": "6501", "nano": 0},
  "executedCommission": {"currency": "rub", "units": "3", "nano": 250500000},
  "figi": "BBG004RVFFC0",
  "direction": "ORDER_DIRECTION_BUY"
}
```

**Ошибка: Insufficient balance (при пустом счёте):**

```
{"code":3,"message":"Not enough balance","description":"30034"}
```

Решение: Использовать `SandboxPayIn` для пополнения виртуального счёта.

**Ошибка: Instrument forbidden for trading (MTSS):**

```
{"code":3,"message":"Instrument forbidden for trading by API","description":"30052"}
```

Решение: Проверять флаг `apiTradeAvailableFlag` перед размещением ордера.

---

### Ограничения T-Invest Sandbox (Итог)

1. **Неполные исторические данные** — только ~20 тикеров имеют свечи
2. **Устаревшие цены** — для некоторых тикеров LastPrice датируется 2022 годом
3. **Ограниченная торговля** — не все инструменты доступны для API-торговли
4. **Лимиты пополнения** — максимальная сумма виртуальных средств ограничена

**Для реальной торговли** необходимо использовать production API с реальным токеном.

---

## Диагностика ошибок

| Код ошибки | Описание | Решение |
|------------|----------|---------|
| 400 | Bad Request | Проверьте параметры запроса |
| 401 | Unauthorized | Проверьте токен |
| 403 | Forbidden | Проверьте права токена |
| 404 | Not Found | Проверьте endpoint |
| 30034 | Not enough balance | Пополните счёт через SandboxPayIn |
| 30052 | Instrument forbidden | Инструмент недоступен для API-торговли |
| 429 | Too Many Requests | Слишком много запросов, подождите |
| 500 | Server Error | Повторите позже |

---

## Инцидент: Требование SMS-подтверждения

### 25 марта 2026

При попытке разместить ордер на TGKN получили ошибку:

```
{"code":9,"message":"Need confirmation: sms","description":"90001"}
```

**Причина:** T-Invest требует подтверждение по SMS для некоторых операций, даже в sandbox.

**Решение:**

1. **Через веб-интерфейс:**
   - Зайдите на https://www.tinkoff.ru/invest/
   - Перейдите в Настройки → Настройки безопасности
   - Отключите "Подтверждение операций по SMS"
   - Вместо этого используйте "Подтверждение через Push" или отключите полностью

2. **Через мобильное приложение:**
   - Настройки → Безопасность
   - Управление подтверждением операций

### Как избежать проблемы

- При создании токена выдайте права только на чтение и торговлю без подтверждения
- В настройках аккаунта отключите требование SMS для API-операций

---

## Retry-логика и обработка ошибок

При работе с API T-Invest возможны временные сбои:
- Разрыв соединения (ConnectionError)
- Rate Limiting (код 429)
- Временная недоступность сервера

### Реализация retry-логики

В классе `TInvestClient` реализована повторная попытка при ошибках:

```python
class TInvestClient:
    def __init__(self, token: str):
        # ...
        self.max_retries = 3      # Максимум 3 попытки
        self.retry_delay = 2      # Задержка между попытками (сек)
    
    def _call(self, method: str, endpoint: str, data: dict = None, retries: int = None):
        if retries is None:
            retries = self.max_retries
        
        for attempt in range(retries):
            try:
                resp = self.session.post(url, headers=self.headers, json=data or {}, timeout=30)
                
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limited - ждём и повторяем
                    wait_time = (attempt + 1) * self.retry_delay
                    print(f"  Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                # Обработка других ошибок...
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * self.retry_delay
                    print(f"  Connection error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
        
        return None
```

### Параметры retry

| Параметр | Значение | Описание |
|----------|---------|----------|
| `max_retries` | 3 | Максимальное количество попыток |
| `retry_delay` | 2 | Базовая задержка между попытками (сек) |
| `timeout` | 30 | Таймаут запроса (сек) |

### Стратегия backoff

При каждой неудачной попытке задержка увеличивается:
- Попытка 1: задержка 2 сек
- Попытка 2: задержка 4 сек  
- Попытка 3: задержка 6 сек

Это экспоненциальный backoff для избежания перегрузки сервера.

### Результат

После добавления retry-логики ордера успешно исполняются даже при временных сбоях сети.

---

## Ссылки

- [Документация T-Invest API](https://developer.tbank.ru/invest/intro/intro)
- [Sandbox методы](https://developer.tbank.ru/invest/intro/developer/sandbox/methods)
- [Swagger UI](https://russianinvestments.github.io/investAPI/swagger-ui/)
- [Старый SDK](https://tinkoff.github.io/investAPI/)
