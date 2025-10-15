# ⚙️ Настройка InfoBot Bots Service

Подробное руководство по настройке всех параметров системы.

---

## 📋 Файлы конфигурации

### 1. `app/config.py` - Основная конфигурация

**API ключи бирж:**
```python
EXCHANGES = {
    'BYBIT': {
        'api_key': 'ваш_api_key',
        'api_secret': 'ваш_secret_key'
    }
}
```

⚠️ **Важно:** Не коммитьте этот файл с реальными ключами!

---

### 2. `data/auto_bot_config.json` - Настройки автобота

**Структура:**
```json
{
  "enabled": false,
  "max_concurrent": 5,
  "check_interval": 180,
  
  "rsi_long_threshold": 29,
  "rsi_short_threshold": 71,
  "rsi_long_exit": 65,
  "rsi_short_exit": 35,
  
  "rsi_time_filter_enabled": true,
  "rsi_time_filter_candles": 4,
  "rsi_time_filter_upper": 65,
  "rsi_time_filter_lower": 35,
  
  "avoid_up_trend": true,
  "avoid_down_trend": true,
  "check_maturity": true,
  "exit_scam_filter_enabled": true,
  
  "stop_loss_percent": 15.0,
  "trailing_stop_activation": 300.0,
  "trailing_stop_distance": 150.0,
  
  "volume_mode": "usdt",
  "volume_value": 10,
  
  "min_candles_for_maturity": 200,
  "min_rsi_low": 35,
  "max_rsi_high": 65
}
```

---

## 🎛️ Параметры Auto Bot

### Основные настройки:

#### `enabled` (boolean)
- **По умолчанию:** `false`
- **Описание:** Включен ли автобот
- **⚠️ ВАЖНО:** Автобот **НЕ включается автоматически** при запуске!

#### `max_concurrent` (integer)
- **По умолчанию:** `5`
- **Описание:** Максимум одновременно активных ботов
- **Диапазон:** 1-20
- **Рекомендация:** 3-5 для начала

#### `check_interval` (integer, секунды)
- **По умолчанию:** `180` (3 минуты)
- **Описание:** Как часто проверять сигналы
- **Диапазон:** 60-600
- **Рекомендация:** 180 (не перегружать API)

---

### RSI пороги:

#### `rsi_long_threshold` (float)
- **По умолчанию:** `29`
- **Описание:** Порог для открытия LONG (RSI ≤ этого значения)
- **Диапазон:** 20-35
- **Рекомендация:** 25-30

#### `rsi_short_threshold` (float)
- **По умолчанию:** `71`
- **Описание:** Порог для открытия SHORT (RSI ≥ этого значения)
- **Диапазон:** 65-80
- **Рекомендация:** 70-75

#### `rsi_long_exit` (float)
- **По умолчанию:** `65`
- **Описание:** Выход из LONG при RSI ≥ этого значения
- **Диапазон:** 60-75

#### `rsi_short_exit` (float)
- **По умолчанию:** `35`
- **Описание:** Выход из SHORT при RSI ≤ этого значения
- **Диапазон:** 25-40

---

### RSI Временной фильтр:

#### `rsi_time_filter_enabled` (boolean)
- **По умолчанию:** `true`
- **Описание:** Включен ли гибридный временной фильтр
- **Рекомендация:** Держите включенным для безопасности

#### `rsi_time_filter_candles` (integer)
- **По умолчанию:** `4`
- **Описание:** Сколько "спокойных" свечей должно пройти после экстремума
- **Диапазон:** 3-10
- **Логика:**
  - Для SHORT: после пика ≥71 должно пройти N свечей в зоне ≥65
  - Для LONG: после лоя ≤29 должно пройти N свечей в зоне ≤35

#### `rsi_time_filter_upper` (float)
- **По умолчанию:** `65`
- **Описание:** "Спокойная зона" для SHORT (после пика)

#### `rsi_time_filter_lower` (float)
- **По умолчанию:** `35`
- **Описание:** "Спокойная зона" для LONG (после лоя)

**Зачем временной фильтр?**
Чтобы не входить слишком рано! Ждем "успокоения" после экстремума.

---

### Фильтры тренда:

#### `avoid_up_trend` (boolean)
- **По умолчанию:** `true`
- **Описание:** Не открывать SHORT в UP тренде
- **Рекомендация:** `true` для безопасности

#### `avoid_down_trend` (boolean)
- **По умолчанию:** `true`
- **Описание:** Не открывать LONG в DOWN тренде
- **Рекомендация:** `true` для безопасности

#### `check_maturity` (boolean)
- **По умолчанию:** `true`
- **Описание:** Проверять зрелость монеты перед входом
- **Рекомендация:** `true` (не торговать новые монеты)

#### `exit_scam_filter_enabled` (boolean)
- **По умолчанию:** `true`
- **Описание:** Фильтр против exit scam / pump-dump
- **Рекомендация:** `true` для защиты

---

### Защитные механизмы:

#### `stop_loss_percent` (float)
- **По умолчанию:** `15.0`
- **Описание:** Стоп-лосс в процентах
- **Пример:** 15% = остановка при убытке 15%
- **Диапазон:** 5-30
- **Рекомендация:** 10-20

#### `trailing_stop_activation` (float)
- **По умолчанию:** `300.0` (3%)
- **Описание:** При какой прибыли активируется trailing stop
- **Пример:** 300% = при прибыли 3x
- **Диапазон:** 100-500
- **Рекомендация:** 200-400

#### `trailing_stop_distance` (float)
- **По умолчанию:** `150.0` (1.5%)
- **Описание:** Расстояние trailing stop от цены
- **Пример:** 150% = 1.5% от цены
- **Диапазон:** 50-200
- **Рекомендация:** 100-200

---

### Параметры входа:

#### `volume_mode` (string)
- **Значения:** `"usdt"` | `"percent"` | `"coins"`
- **По умолчанию:** `"usdt"`
- **Описание:** 
  - `"usdt"` - фиксированная сумма в USDT
  - `"percent"` - процент от баланса
  - `"coins"` - количество монет

#### `volume_value` (float)
- **По умолчанию:** `10`
- **Описание:** Значение для volume_mode
- **Примеры:**
  - mode="usdt", value=10 → вход на 10 USDT
  - mode="percent", value=5 → вход на 5% от баланса
  - mode="coins", value=100 → вход на 100 монет

---

### Критерии зрелости:

#### `min_candles_for_maturity` (integer)
- **По умолчанию:** `200`
- **Описание:** Минимум свечей 6H для зрелости (~50 дней)
- **Диапазон:** 150-300

#### `min_rsi_low` (float)
- **По умолчанию:** `35`
- **Описание:** RSI должен был достичь этого минимума

#### `max_rsi_high` (float)
- **По умолчанию:** `65`
- **Описание:** RSI должен был достичь этого максимума

**Зачем?** Монета должна пройти полный цикл (низ и верх) для торговли.

---

## 🔧 Системные настройки

### Файл: `data/system_config.json`

```json
{
  "bot_status_update_interval": 30,
  "position_sync_interval": 30,
  "inactive_bot_cleanup_interval": 600,
  "inactive_bot_timeout": 600,
  "stop_loss_setup_interval": 300,
  "rsi_update_interval": 300
}
```

### Параметры:

#### `bot_status_update_interval` (секунды)
- **По умолчанию:** `30`
- **Описание:** Как часто обновлять кэш ботов

#### `position_sync_interval` (секунды)
- **По умолчанию:** `30`
- **Описание:** Как часто синхронизировать позиции с биржей

#### `rsi_update_interval` (секунды)
- **По умолчанию:** `300` (5 минут)
- **Описание:** Как часто обновлять RSI данные
- **⚠️ Не ставьте < 60:** перегрузка API!

---

## 🎯 Рекомендуемые конфигурации

### Консервативная (безопасная):
```json
{
  "enabled": false,
  "max_concurrent": 3,
  "rsi_long_threshold": 25,
  "rsi_short_threshold": 75,
  "stop_loss_percent": 20,
  "rsi_time_filter_candles": 6,
  "check_maturity": true
}
```

### Агрессивная (рискованная):
```json
{
  "enabled": false,
  "max_concurrent": 10,
  "rsi_long_threshold": 30,
  "rsi_short_threshold": 70,
  "stop_loss_percent": 10,
  "rsi_time_filter_candles": 3,
  "check_maturity": false
}
```

### Сбалансированная (рекомендуется):
```json
{
  "enabled": false,
  "max_concurrent": 5,
  "rsi_long_threshold": 29,
  "rsi_short_threshold": 71,
  "stop_loss_percent": 15,
  "rsi_time_filter_candles": 4,
  "check_maturity": true
}
```

---

## 🔄 Изменение настроек

### Через API:
```bash
curl -X POST http://localhost:5001/api/auto-bot/config \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 3,
    "rsi_long_threshold": 25,
    "stop_loss_percent": 20
  }'
```

### Через файл:
1. Остановите `bots.py` (Ctrl+C)
2. Отредактируйте `data/auto_bot_config.json`
3. Запустите `python bots.py`

### Через UI:
1. Откройте http://localhost:5000
2. Раздел "Боты" → "Настройки Auto Bot"
3. Измените параметры
4. Нажмите "Сохранить"

---

## 🧪 Тестирование конфигурации

### Проверка текущих настроек:
```bash
curl http://localhost:5001/api/auto-bot/config
```

### Восстановление к дефолту:
```bash
curl -X POST http://localhost:5001/api/auto-bot/restore-defaults
```

---

## 💡 Советы по настройке

### Для начинающих:
- ✅ Держите `enabled: false` пока тестируете
- ✅ Начните с `max_concurrent: 1`
- ✅ Используйте демо-счет
- ✅ Включите все фильтры

### Для опытных:
- Можно увеличить `max_concurrent`
- Можно настроить более агрессивные пороги RSI
- Можно отключить фильтры (на свой риск)

### Оптимальные значения (по опыту):
- RSI LONG: 27-30
- RSI SHORT: 70-73
- Stop Loss: 12-18%
- Max Concurrent: 3-7
- Time Filter: 4-6 свечей

---

## 📊 Мониторинг настроек

### Логи:
```bash
tail -f logs/bots.log | grep CONFIG
```

### API:
```bash
curl http://localhost:5001/api/auto-bot/config
```

### UI:
http://localhost:5000 → Боты → Настройки

---

**Следующее:** [AUTO_BOT.md](AUTO_BOT.md) - Как работать с автоботом

