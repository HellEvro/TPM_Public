# 🏗️ Архитектура InfoBot Bots Service

Подробное описание архитектуры системы после разбиения на модули.

---

## 📊 Общая архитектура

```
InfoBot Project
├── app.py (5000)           ← Основной веб-интерфейс
├── bots.py (5001)          ← Сервис управления ботами ⭐
├── protector.py (5002)     ← Защитный сервис
│
├── bot_engine/             ← Общая библиотека торговой логики
│   ├── utils/              ← RSI/EMA утилиты
│   ├── filters.py          ← Общие фильтры
│   ├── maturity_checker.py ← Проверка зрелости
│   ├── trading_bot.py      ← Базовый класс бота
│   └── ...
│
├── bots_modules/           ← Модули специфичные для bots.py ⭐ НОВОЕ!
│   ├── imports_and_globals.py
│   ├── calculations.py
│   ├── maturity.py
│   ├── filters.py
│   ├── bot_class.py
│   ├── sync_and_cache.py
│   ├── workers.py
│   ├── init_functions.py
│   └── api_endpoints.py
│
├── exchanges/              ← Адаптеры для бирж
│   ├── bybit_exchange.py
│   ├── binance_exchange.py
│   └── ...
│
└── data/                   ← Персистентные данные
    ├── bots_state.json
    ├── auto_bot_config.json
    ├── mature_coins.json
    └── optimal_ema.json
```

---

## 🔄 Разбиение bots.py

### До разбиения:
```
bots.py (7678 строк)
├── Импорты
├── Константы
├── Глобальные переменные
├── 135 функций
├── 1 класс (NewTradingBot)
└── 60+ API endpoints
```

### После разбиения:
```
bots.py (215 строк) - главный координатор
└── bots_modules/ (10 модулей по логическим блокам)
```

---

## 🧩 Модульная структура

### Принцип разделения:

```
bots_modules/
│
├── 📦 Слой данных и конфигурации
│   ├── imports_and_globals.py  ← Импорты, константы, Flask app
│   └── sync_and_cache.py       ← Кэш, состояние, синхронизация
│
├── 🧮 Слой вычислений
│   ├── calculations.py         ← RSI/EMA расчеты
│   ├── maturity.py            ← Проверка зрелости монет
│   └── optimal_ema.py         ← Оптимальные EMA периоды
│
├── 🎯 Слой бизнес-логики
│   ├── filters.py             ← Фильтры сигналов
│   ├── bot_class.py           ← Класс торгового бота
│   └── workers.py             ← Фоновые воркеры
│
├── 🔧 Слой инициализации
│   └── init_functions.py      ← Инициализация системы
│
└── 🌐 Слой API
    └── api_endpoints.py       ← REST API (60+ endpoints)
```

---

## 🔗 Взаимодействие модулей

### Граф зависимостей:

```
bots.py
  ↓
  ├→ imports_and_globals.py (базовый, импортируется первым)
  │    ├→ Flask app
  │    ├→ Константы (RSI_OVERSOLD, EMA_FAST, ...)
  │    ├→ Глобальные переменные (bots_data, coins_rsi_data, ...)
  │    └→ GlobalState (для обмена exchange между модулями)
  │
  ├→ calculations.py
  │    ├─ Использует: TREND_CONFIRMATION_BARS (из imports_and_globals)
  │    └─ Использует: get_optimal_ema_periods (из optimal_ema)
  │
  ├→ maturity.py
  │    ├─ Использует: bots_data_lock, bots_data (из imports_and_globals)
  │    └─ Использует: calculate_rsi_history (из calculations)
  │
  ├→ optimal_ema.py (независимый)
  │
  ├→ filters.py
  │    ├─ Использует: calculate_rsi, analyze_trend (из calculations)
  │    ├─ Использует: check_coin_maturity (из maturity)
  │    ├─ Использует: get_optimal_ema_periods (из optimal_ema)
  │    └─ Использует: RSI_OVERSOLD, BOT_STATUS (из imports_and_globals)
  │
  ├→ bot_class.py
  │    ├─ Использует: bots_data, BOT_STATUS (из imports_and_globals)
  │    └─ Использует: check_rsi_time_filter (из filters)
  │
  ├→ sync_and_cache.py
  │    ├─ Использует: bots_data, exchange (из imports_and_globals)
  │    ├─ Использует: get_exchange() (GlobalState)
  │    └─ Использует: ensure_exchange_initialized (из init_functions)
  │
  ├→ workers.py
  │    ├─ Использует: shutdown_flag, bots_data (из imports_and_globals)
  │    ├─ Использует: save_bots_state (из sync_and_cache)
  │    └─ Использует: process_auto_bot_signals (из filters)
  │
  ├→ init_functions.py
  │    ├─ Использует: exchange, system_initialized (из imports_and_globals)
  │    ├─ Использует: set_exchange() (GlobalState)
  │    ├─ Использует: load_mature_coins (из maturity)
  │    └─ Использует: load_all_coins_rsi (из filters)
  │
  └→ api_endpoints.py
       ├─ Использует: bots_app (из imports_and_globals)
       ├─ Использует: save_auto_bot_config (из sync_and_cache)
       ├─ Использует: get_effective_signal (из filters)
       └─ Регистрирует 60+ Flask routes
```

---

## 🔄 Жизненный цикл системы

### 1. Запуск (`python bots.py`)
```
1. Проверка порта 5001
2. Импорт всех модулей (from bots_modules.X import *)
3. Настройка логирования
4. Инициализация (init_bot_service):
   ├─ Загрузка конфигураций
   ├─ Инициализация биржи (ПЕРВЫМ!)
   ├─ Запуск Smart RSI Manager
   ├─ Синхронизация с биржей
   └─ Проверка конфликтов позиций
5. Запуск фоновых воркеров:
   ├─ Auto Save Worker (каждые 30 сек)
   └─ Auto Bot Worker (режим ожидания)
6. Запуск Flask сервера (блокирующий вызов)
```

### 2. Работа
```
┌─ Flask Thread ────────────────────┐
│  Обрабатывает HTTP запросы        │
│  GET /api/bots                    │
│  POST /api/bots/create            │
│  PUT /api/auto-bot/config         │
└───────────────────────────────────┘

┌─ Smart RSI Manager Thread ────────┐
│  Обновляет RSI каждые 5 минут     │
│  Вызывает load_all_coins_rsi()    │
│  583 монеты → RSI данные          │
└───────────────────────────────────┘

┌─ Auto Save Worker Thread ─────────┐
│  Сохраняет состояние каждые 30 сек│
│  bots_state.json                  │
│  mature_coins.json                │
└───────────────────────────────────┘

┌─ Auto Bot Worker Thread ──────────┐
│  Проверяет enabled каждые N сек   │
│  Если enabled=True:               │
│    └→ process_auto_bot_signals()  │
│  Обновляет кэш позиций            │
│  Проверяет стоп-лоссы             │
└───────────────────────────────────┘

┌─ Async Processor Thread ──────────┐
│  Асинхронная обработка сигналов   │
│  Синхронизация позиций (1 мин)    │
└───────────────────────────────────┘
```

### 3. Остановка (Ctrl+C)
```
1. Signal handler перехватывает SIGINT
2. Устанавливает shutdown_flag
3. Вызывает cleanup_bot_service():
   ├─ Останавливает async processor
   ├─ Сохраняет состояние ботов
   └─ Сохраняет зрелые монеты
4. os._exit(0) - мгновенный выход
```

---

## 🌐 API архитектура

### REST API (порт 5001)
```
GET  /api/status              ← Статус сервиса
GET  /api/bots                ← Список ботов
POST /api/bots/create         ← Создать бота
POST /api/bots/{id}/start     ← Запустить бота
POST /api/bots/{id}/stop      ← Остановить бота
DELETE /api/bots/{id}         ← Удалить бота
...и еще 54 endpoints
```

Полный список: [API_REFERENCE.md](API_REFERENCE.md)

---

## 💾 Хранение данных

### Файлы состояния:
```
data/
├── bots_state.json          ← Состояние всех ботов
│   ├─ bots: {}              ← Словарь ботов по символам
│   ├─ auto_bot_config: {}   ← Настройки автобота
│   └─ global_stats: {}      ← Общая статистика
│
├── auto_bot_config.json     ← Конфигурация автобота
│   ├─ enabled: false        ← Включен ли автобот
│   ├─ max_concurrent: 5     ← Макс. одновременных ботов
│   ├─ rsi_long_threshold: 29
│   └─ rsi_short_threshold: 71
│
├── mature_coins.json        ← Зрелые монеты (постоянное хранилище)
│   └─ {symbol: {maturity_data, timestamp}}
│
├── optimal_ema.json         ← Оптимальные EMA периоды
│   └─ {symbol: {ema_short, ema_long, accuracy}}
│
├── rsi_cache.json           ← Кэш RSI данных
│   └─ {coins: {}, last_update, ...}
│
├── process_state.json       ← Состояние процессов
│   ├─ smart_rsi_manager: {}
│   ├─ auto_bot_worker: {}
│   └─ auto_save_worker: {}
│
└── system_config.json       ← Системные настройки
    ├─ bot_status_update_interval: 30
    ├─ position_sync_interval: 30
    └─ rsi_update_interval: 300
```

---

## 🔐 Безопасность и изоляция

### Принцип разделения ответственности:

1. **bot_engine/** - Переиспользуемая библиотека
   - Может использоваться в app.py, protector.py
   - Общая торговая логика
   - Независимая от Flask

2. **bots_modules/** - Специфичные модули bots.py
   - Только для сервиса ботов
   - Flask API endpoints
   - Воркеры автосохранения
   - Интеграция с основным сервисом

### GlobalState pattern:

```python
# imports_and_globals.py
class GlobalState:
    exchange = None
    smart_rsi_manager = None
    system_initialized = False

_state = GlobalState()

def get_exchange():
    return _state.exchange

def set_exchange(exch):
    _state.exchange = exch
```

**Зачем?** Чтобы изменения в `exchange` были видны **во всех модулях**.

---

## 🔀 Потоки выполнения

### Главный поток (Main Thread):
- Инициализация системы
- Запуск Flask сервера (блокирует)

### Flask Worker Threads:
- Обработка HTTP запросов
- Управление ботами через API

### Smart RSI Manager Thread:
- Обновление RSI каждые 5 минут
- Отслеживание закрытия свечей 6H
- Вызов торговых сигналов

### Auto Save Worker Thread:
- Сохранение состояния каждые 30 секунд
- Атомарная запись с retry

### Auto Bot Worker Thread:
- Проверка enabled статуса
- Если enabled=True → создание ботов по сигналам
- Обновление кэша позиций
- Проверка стоп-лоссов

### Async Processor Thread:
- Асинхронная обработка сигналов
- Синхронизация позиций каждую минуту
- Оптимизация производительности

---

## 🔒 Потокобезопасность

### Блокировки (Locks):

```python
# Глобальные блокировки
bots_data_lock            ← Защищает bots_data
rsi_data_lock             ← Защищает coins_rsi_data
bots_cache_lock           ← Защищает bots_cache_data
mature_coins_lock         ← Защищает mature_coins_storage
coin_processing_lock      ← Управляет coin_processing_locks

# Блокировки для каждой монеты
coin_processing_locks = {
    'BTC': Lock(),
    'ETH': Lock(),
    ...
}
```

**Зачем?** Предотвращение race conditions при одновременном доступе из разных потоков.

---

## 📡 Взаимодействие с биржей

```
Bots Service
     ↓
ExchangeFactory.create_exchange('BYBIT', api_key, secret)
     ↓
BybitExchange (адаптер)
     ↓
Bybit API (REST)
     ├─ get_positions()
     ├─ place_order()
     ├─ cancel_order()
     ├─ set_trading_stop()
     └─ get_chart_data()
```

### Retry логика:
- Max 3 попытки для критичных операций
- Экспоненциальная задержка (0.1s → 0.2s → 0.4s)
- Rate limit handling

---

## 🎯 Торговая логика

### Схема принятия решения:

```
1. Smart RSI Manager обновляет RSI
   ↓
2. load_all_coins_rsi() загружает данные
   ↓
3. Для каждой монеты:
   get_coin_rsi_data()
   ├─ calculate_rsi(prices)           ← RSI значение
   ├─ analyze_trend_6h()              ← UP/DOWN/NEUTRAL
   ├─ check_coin_maturity()           ← Зрелая/Незрелая
   └─ perform_enhanced_rsi_analysis() ← Расширенный анализ
   ↓
4. get_effective_signal(coin)
   ├─ RSI <= 29 → ENTER_LONG
   ├─ RSI >= 71 → ENTER_SHORT
   ├─ RSI >= 65 (в лонге) → EXIT_LONG
   └─ RSI <= 35 (в шорте) → EXIT_SHORT
   ↓
5. Если Auto Bot enabled:
   process_auto_bot_signals()
   ├─ Проверяет фильтры:
   │  ├─ check_coin_maturity ✓
   │  ├─ check_rsi_time_filter ✓
   │  ├─ check_exit_scam_filter ✓
   │  └─ check_no_existing_position ✓
   │
   └─ Если все OK → create_bot(symbol)
   ↓
6. NewTradingBot управляет позицией
   ├─ Открытие по RSI сигналу
   ├─ Установка stop-loss
   ├─ Trailing stop при прибыли
   └─ Закрытие по RSI выхода
```

---

## 🔄 Обновление данных

### RSI Update Flow:

```
Smart RSI Manager (каждые 5 минут)
  ↓
load_all_coins_rsi(exchange)
  ↓
Параллельная обработка (ThreadPoolExecutor, 3 воркера)
  ├─ Пакет 1: 50 монет
  ├─ Пакет 2: 50 монет
  ...
  └─ Пакет 12: 33 монеты
  ↓
Для каждой монеты:
  get_coin_rsi_data(symbol, exchange)
    ├─ exchange.get_chart_data('6h', '30d')
    ├─ calculate_rsi(closes, 14)
    ├─ analyze_trend_6h(symbol)
    ├─ check_coin_maturity_with_storage()
    └─ perform_enhanced_rsi_analysis()
  ↓
Инкрементальное обновление coins_rsi_data
  ↓
save_rsi_cache() - сохранение в файл
```

---

## 💡 Ключевые паттерны

### 1. GlobalState для shared state
```python
_state = GlobalState()
exchange = set_exchange(new_exchange)  # Видно везде
```

### 2. Lock-based synchronization
```python
with bots_data_lock:
    bot = bots_data['bots'][symbol]
```

### 3. Try-except для импортов
```python
try:
    from bots_modules.X import func
except ImportError:
    def func(): pass  # Заглушка
```

### 4. Daemon threads
```python
thread = threading.Thread(target=worker, daemon=True)
thread.start()  # Автоматически останавливается при выходе
```

---

## 📈 Производительность

### Оптимизации:

1. **Параллельная загрузка RSI** (3 воркера)
2. **Инкрементальное обновление** (батчами по 50 монет)
3. **Кэширование данных** (RSI, зрелость, optimal EMA)
4. **Подавление повторяющихся логов** (should_log_message)
5. **Асинхронный процессор** для тяжелых операций
6. **Connection pooling** (100 пулов × 200 соединений)

### Метрики:

- **Загрузка 583 монет:** ~2-3 минуты
- **Обновление RSI:** ~1-2 минуты (инкрементально)
- **RAM usage:** ~500MB в обычном режиме
- **CPU usage:** <5% в режиме ожидания

---

## 🎯 Следующие шаги

Для более подробной информации:
- **Модули:** [MODULES.md](MODULES.md)
- **Настройка:** [CONFIGURATION.md](CONFIGURATION.md)
- **API:** [API_REFERENCE.md](API_REFERENCE.md)
- **Разработка:** [DEVELOPMENT.md](DEVELOPMENT.md)

