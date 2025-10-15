# 📊 Bot History - История торговых ботов

Полное руководство по модулю логирования истории ботов.

---

## 🎯 Что это?

**`bot_history.py`** - модуль для автоматического логирования всех действий торговых ботов:

- 🚀 **Запуск ботов**
- 🛑 **Остановка ботов**  
- 📊 **Торговые сигналы**
- 📈 **Открытие позиций**
- 💰 **Закрытие позиций**
- 📉 **Статистика торговли**

---

## ✅ Статус

```
✅ Модуль реализован
✅ Интегрирован в bots.py
✅ API endpoints работают
✅ UI готов (вкладка "История")
✅ Данные сохраняются в data/bot_history.json
```

---

## 📦 Структура данных

### История действий (history):
```json
{
  "id": "start_bot_123_1234567890.123",
  "timestamp": "2025-10-15T02:30:00.123456",
  "action_type": "BOT_START",
  "action_name": "Запуск бота",
  "bot_id": "bot_123",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "details": "Запущен бот LONG для BTCUSDT"
}
```

### История сделок (trades):
```json
{
  "id": "trade_bot_123_1234567890.123",
  "timestamp": "2025-10-15T02:30:00.123456",
  "bot_id": "bot_123",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "size": 0.001,
  "entry_price": 67500.0,
  "exit_price": 68000.0,
  "pnl": 0.5,
  "roi": 0.74,
  "status": "CLOSED"
}
```

---

## 🔌 API Endpoints

### 1. Получить историю действий
```http
GET /api/bots/history?symbol=BTCUSDT&action_type=BOT_START&limit=100
```

**Параметры:**
- `symbol` (optional) - Фильтр по символу
- `action_type` (optional) - Тип действия (BOT_START, BOT_STOP, SIGNAL, POSITION_OPENED, POSITION_CLOSED)
- `limit` (optional, default=100) - Макс. кол-во записей

**Ответ:**
```json
{
  "success": true,
  "history": [...],
  "count": 50
}
```

### 2. Получить историю сделок
```http
GET /api/bots/trades?symbol=ETHUSDT&trade_type=LONG&limit=50
```

**Параметры:**
- `symbol` (optional) - Фильтр по символу
- `trade_type` (optional) - LONG или SHORT
- `limit` (optional, default=100) - Макс. кол-во записей

**Ответ:**
```json
{
  "success": true,
  "trades": [...],
  "count": 25
}
```

### 3. Получить статистику
```http
GET /api/bots/statistics?symbol=BTCUSDT
```

**Параметры:**
- `symbol` (optional) - Фильтр по символу (если нет - общая статистика)

**Ответ:**
```json
{
  "success": true,
  "statistics": {
    "total_trades": 100,
    "profitable_trades": 60,
    "losing_trades": 40,
    "win_rate": 60.0,
    "total_pnl": 125.50,
    "avg_pnl": 1.26,
    "best_trade": {...},
    "worst_trade": {...},
    "symbol": "BTCUSDT"
  }
}
```

### 4. Очистить историю
```http
POST /api/bots/history/clear
Content-Type: application/json

{
  "symbol": "BTCUSDT"  // optional, если нет - очистит всю историю
}
```

### 5. Создать демо-данные
```http
POST /api/bots/history/demo
```

Создает 20 тестовых записей для проверки UI.

---

## 💻 Использование в коде

### Импорт:
```python
from bot_history import (
    bot_history_manager,
    log_bot_start,
    log_bot_stop,
    log_bot_signal,
    log_position_opened,
    log_position_closed
)
```

### Примеры использования:

#### 1. Запуск бота:
```python
log_bot_start(
    bot_id="bot_123",
    symbol="BTCUSDT",
    direction="LONG",
    config={"stop_loss": 15, "volume": 10}
)
```

#### 2. Торговый сигнал:
```python
log_bot_signal(
    symbol="ETHUSDT",
    signal_type="ENTER_LONG",
    rsi=28.5,
    price=3500.0,
    details={"ema": 3450, "trend": "UP"}
)
```

#### 3. Открытие позиции:
```python
log_position_opened(
    bot_id="bot_123",
    symbol="BTCUSDT",
    direction="LONG",
    size=0.001,
    entry_price=67500.0,
    stop_loss=57375.0,  # -15%
    take_profit=None
)
```

#### 4. Закрытие позиции:
```python
log_position_closed(
    bot_id="bot_123",
    symbol="BTCUSDT",
    direction="LONG",
    exit_price=68000.0,
    pnl=0.5,
    roi=0.74,
    reason="Take Profit"
)
```

#### 5. Остановка бота:
```python
log_bot_stop(
    bot_id="bot_123",
    symbol="BTCUSDT",
    reason="Позиция закрыта",
    pnl=0.5
)
```

---

## 🔄 Автоматическая интеграция

Чтобы боты автоматически логировали свои действия, добавьте вызовы в класс `NewTradingBot`:

### В метод `__init__`:
```python
def __init__(self, symbol, config=None, exchange=None):
    # ... существующий код ...
    
    # Логируем запуск бота
    if BOT_HISTORY_AVAILABLE:
        log_bot_start(
            bot_id=self.bot_id,
            symbol=self.symbol,
            direction=self.direction,
            config=self.config
        )
```

### В метод открытия позиции:
```python
def open_position(self):
    # ... открытие позиции ...
    
    # Логируем открытие
    if BOT_HISTORY_AVAILABLE and self.position_opened:
        log_position_opened(
            bot_id=self.bot_id,
            symbol=self.symbol,
            direction=self.direction,
            size=self.position_size,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss_price
        )
```

### В метод закрытия позиции:
```python
def close_position(self, reason="Manual"):
    # ... закрытие позиции ...
    
    # Логируем закрытие
    if BOT_HISTORY_AVAILABLE:
        log_position_closed(
            bot_id=self.bot_id,
            symbol=self.symbol,
            direction=self.direction,
            exit_price=exit_price,
            pnl=pnl,
            roi=roi,
            reason=reason
        )
        
        log_bot_stop(
            bot_id=self.bot_id,
            symbol=self.symbol,
            reason=f"Позиция закрыта: {reason}",
            pnl=pnl
        )
```

---

## 📊 UI - Вкладка "История"

В UI уже есть готовая вкладка "История":

```html
<button class="bots-tab-btn" data-tab="history">
    📊 История
</button>
```

### Что показывается:
- ✅ Список всех действий ботов
- ✅ Фильтрация по символу
- ✅ Фильтрация по типу действия
- ✅ Статистика по сделкам
- ✅ История сделок с PnL
- ✅ Win Rate и общий профит

---

## 📁 Хранение данных

**Файл:** `data/bot_history.json`

**Структура:**
```json
{
  "history": [
    // Массив всех действий (до 10000 записей)
  ],
  "trades": [
    // Массив всех сделок (до 5000 записей)
  ],
  "last_update": "2025-10-15T02:30:00.123456"
}
```

**Автоматическая ротация:**
- История: последние 10000 записей
- Сделки: последние 5000 записей
- Старые записи автоматически удаляются

---

## 🛠️ Полезные команды

### Просмотр истории:
```bash
curl http://localhost:5001/api/bots/history?limit=10
```

### Просмотр сделок:
```bash
curl http://localhost:5001/api/bots/trades?limit=10
```

### Статистика:
```bash
curl http://localhost:5001/api/bots/statistics
```

### Очистка истории:
```bash
curl -X POST http://localhost:5001/api/bots/history/clear \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT"}'
```

### Создать демо-данные:
```bash
curl -X POST http://localhost:5001/api/bots/history/demo
```

---

## 🎯 Примеры запросов

### JavaScript (в UI):
```javascript
// Получить историю
const response = await fetch('/api/bots/history?symbol=BTCUSDT&limit=50');
const data = await response.json();
console.log(data.history);

// Получить статистику
const stats = await fetch('/api/bots/statistics?symbol=ETHUSDT');
const statsData = await stats.json();
console.log(statsData.statistics);

// Очистить историю
await fetch('/api/bots/history/clear', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({symbol: 'BTCUSDT'})
});
```

### Python:
```python
import requests

# Получить историю
response = requests.get('http://localhost:5001/api/bots/history', params={
    'symbol': 'BTCUSDT',
    'action_type': 'POSITION_CLOSED',
    'limit': 50
})
history = response.json()['history']

# Статистика
stats = requests.get('http://localhost:5001/api/bots/statistics').json()
print(f"Win Rate: {stats['statistics']['win_rate']:.2f}%")
print(f"Total PnL: {stats['statistics']['total_pnl']:.2f} USDT")
```

---

## 🐛 Troubleshooting

### Проблема: История не сохраняется

**Решение:**
```bash
# Проверьте права на запись
ls -la data/

# Проверьте что директория существует
mkdir -p data/

# Проверьте что модуль загружен
curl http://localhost:5001/api/status
```

### Проблема: "bot_history недоступен"

**Причина:** Модуль не найден

**Решение:**
```bash
# Проверьте наличие файла
ls -la bot_history.py

# Проверьте импорт
python -c "import bot_history; print('OK')"
```

### Проблема: UI не показывает историю

**Решение:**
```bash
# Проверьте API
curl http://localhost:5001/api/bots/history?limit=5

# Создайте демо-данные
curl -X POST http://localhost:5001/api/bots/history/demo

# Обновите страницу
```

---

## 📈 Статистика и аналитика

С помощью bot_history вы можете:

✅ **Отслеживать эффективность** каждого бота
✅ **Анализировать Win Rate** по символам
✅ **Находить лучшие/худшие** сделки
✅ **Видеть историю** всех действий
✅ **Дебажить проблемы** с ботами
✅ **Оптимизировать стратегии** на основе данных

---

## 🎉 Заключение

**`bot_history.py`** - это мощный инструмент для:
- 📊 Полного контроля над действиями ботов
- 💰 Анализа прибыльности
- 🐛 Отладки проблем
- 📈 Оптимизации стратегий

**Статус:** ✅ Готов к использованию!

---

**Дата:** 15 октября 2025  
**Версия:** 1.0  
**Статус:** ✅ Полностью реализован и протестирован

