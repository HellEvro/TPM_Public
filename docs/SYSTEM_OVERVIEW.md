# 📚 InfoBot - Полный обзор системы

**Комплексная торговая система для криптовалютных бирж**

---

## 🎯 Что такое InfoBot?

InfoBot - это **двухкомпонентная система** для мониторинга и автоматической торговли на криптовалютных биржах:

1. **app.py** (Порт 5000) - Главное приложение с UI
2. **bots.py** (Порт 5001) - Сервис управления торговыми ботами

---

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                     INFOBOT ECOSYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐      ┌────────────────────────┐  │
│  │   APP.PY (Port 5000) │◄────►│  BOTS.PY (Port 5001)  │  │
│  │  Главное приложение  │      │  Сервис ботов          │  │
│  └──────────────────────┘      └────────────────────────┘  │
│           │                              │                   │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌──────────────────────┐      ┌────────────────────────┐  │
│  │     WEB UI           │      │  Trading Bots Engine   │  │
│  │  - Мониторинг        │      │  - Auto Bot            │  │
│  │  - Статистика        │      │  - Manual Bots         │  │
│  │  │  - Управление     │      │  - RSI Manager         │  │
│  └──────────────────────┘      └────────────────────────┘  │
│                                                               │
│                         ▼                                     │
│            ┌────────────────────────┐                        │
│            │    BYBIT EXCHANGE      │                        │
│            │  (или другие биржи)    │                        │
│            └────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Компоненты системы

### 1. APP.PY - Главное приложение (Port 5000)

**Основная задача:** Мониторинг позиций и UI

#### Функции:
- 📊 **Мониторинг позиций** в реальном времени
- 📈 **Статистика** прибыли/убытков
- 💰 **PnL трекинг** (закрытые позиции)
- 🔔 **Telegram уведомления**
- 🌐 **Web интерфейс** (HTML/JS)
- 📡 **Прокси для API ботов**

#### Воркеры:
- `background_update()` - Обновление позиций каждые 5 сек
- `send_daily_report()` - Ежедневный отчет в Telegram
- `background_cache_cleanup()` - Очистка кэша

#### API Endpoints:
```
GET  /                              - Главная страница
GET  /api/positions                 - Все позиции
GET  /api/closed_pnl                - Закрытые позиции
GET  /api/stats                     - Статистика
POST /api/telegram/toggle           - Вкл/выкл Telegram
POST /api/language                  - Смена языка

# Прокси для bots.py:
GET  /api/bots/list                 → 5001/api/bots
POST /api/bots/create               → 5001/api/bots/create
GET  /api/bots/auto-config          → 5001/api/auto-bot/config
... (все /api/bots/* проксируются на 5001)
```

---

### 2. BOTS.PY - Сервис ботов (Port 5001)

**Основная задача:** Управление торговыми ботами

#### Модули (10 файлов):
```
bots_modules/
├── imports_and_globals.py    # Импорты, константы, Flask app
├── calculations.py            # RSI, EMA расчеты
├── maturity.py                # Проверка зрелости монет
├── optimal_ema.py             # Оптимальные EMA периоды
├── filters.py                 # Фильтры сигналов
├── bot_class.py               # Класс NewTradingBot
├── sync_and_cache.py          # Синхронизация с биржей
├── workers.py                 # Фоновые воркеры
├── init_functions.py          # Инициализация системы
└── api_endpoints.py           # Flask API (60+ endpoints)
```

#### Функции:
- 🤖 **Auto Bot** - Автоматическая торговля по RSI
- 👤 **Manual Bots** - Ручное управление
- 📊 **Smart RSI Manager** - Умное обновление RSI
- 🔄 **Синхронизация** с биржей
- 💾 **Авто-сохранение** состояния
- 📈 **Optimal EMA Worker** - Расчет оптимальных EMA

#### Воркеры:
- `auto_save_worker()` - Сохранение каждые 30 сек
- `auto_bot_worker()` - Проверка сигналов каждые 3 мин
- `smart_rsi_manager` - Обновление RSI каждые 5 мин
- `async_processor` - Синхронизация позиций каждую мин
- `optimal_ema_worker` - Пересчет EMA каждые 6 часов

#### API Endpoints (основные):
```
GET  /api/status                    - Статус системы
GET  /api/bots                      - Список всех ботов
POST /api/bots/create               - Создать бота
POST /api/bots/{id}/close           - Закрыть бота

GET  /api/auto-bot/config           - Настройки Auto Bot
POST /api/auto-bot/config           - Изменить настройки
POST /api/auto-bot/enable           - Включить Auto Bot
POST /api/auto-bot/disable          - Выключить Auto Bot

GET  /api/rsi/all                   - RSI всех монет
GET  /api/rsi/{symbol}              - RSI конкретной монеты
POST /api/rsi/update                - Обновить RSI

GET  /api/mature-coins              - Зрелые монеты
POST /api/mature-coins/verify       - Проверить зрелость
POST /api/mature-coins/clear        - Очистить список

GET  /api/optimal-ema               - Оптимальные EMA
POST /api/optimal-ema/update        - Обновить EMA

GET  /api/system-config             - Системные настройки
POST /api/system-config             - Изменить настройки
```

---

## 🔄 Как компоненты взаимодействуют?

### Схема взаимодействия:

```
1. Пользователь открывает http://localhost:5000
   └─► app.py отдает HTML/JS интерфейс

2. UI делает запрос на /api/bots/list
   └─► app.py проксирует на http://localhost:5001/api/bots
       └─► bots.py возвращает список ботов
           └─► app.py передает UI

3. Пользователь кликает "Создать бота"
   └─► UI отправляет POST /api/bots/create
       └─► app.py проксирует на 5001
           └─► bots.py создает бота
               └─► NewTradingBot открывает позицию на бирже
                   └─► Биржа возвращает статус

4. background_update() в app.py обновляет позиции
   └─► Запрашивает биржу напрямую
       └─► Обновляет positions_data
           └─► UI видит изменения

5. auto_bot_worker() в bots.py проверяет сигналы
   └─► smart_rsi_manager обновляет RSI
       └─► process_auto_bot_signals() ищет возможности
           └─► Если найден сигнал → создает бота
               └─► Бот открывает позицию
```

---

## 🚀 Последовательность запуска

### Правильный порядок:

```bash
# 1. Запускаем app.py (главное приложение)
python app.py

# Терминал 1:
# * Running on http://localhost:5000
# ✅ Браузер откроется автоматически

# 2. В новом терминале запускаем bots.py
python bots.py

# Терминал 2:
# * Running on http://localhost:5001
# ✅ Bots Service запущен

# Теперь система работает полностью!
```

### Что происходит:

**app.py запускается:**
- Инициализирует Flask на порту 5000
- Запускает background_update()
- Запускает send_daily_report()
- Открывает браузер
- **НО** /api/bots/* не работают (bots.py еще не запущен)

**bots.py запускается:**
- Инициализирует Flask на порту 5001
- Подключается к бирже
- Загружает RSI данные (~5 минут)
- Запускает всех воркеров
- **ТЕПЕРЬ** /api/bots/* в app.py работают!

---

## 📊 Файлы данных

### Общие:
```
data/
├── bots_state.json              # Состояние ботов
├── auto_bot_config.json         # Настройки Auto Bot
├── system_config.json           # Системные настройки
├── rsi_cache.json               # Кэш RSI данных
├── mature_coins.json            # Зрелые монеты
├── optimal_ema.json             # Оптимальные EMA
├── process_state.json           # Состояние процессов
└── default_auto_bot_config.json # Дефолтная конфигурация
```

### Логи:
```
logs/
├── bots.log                     # Логи bots.py
├── app.log                      # Логи app.py
├── telegram.log                 # Telegram уведомления
└── optimal_ema.log              # Логи Optimal EMA Worker
```

---

## ⚙️ Конфигурация

### app/config.py (для app.py):
```python
# Биржа
EXCHANGES = {
    'BYBIT': {
        'api_key': 'ваш_api_key',
        'api_secret': 'ваш_secret'
    }
}

# Сервер
APP_HOST = '0.0.0.0'
APP_PORT = 5000

# Telegram
TELEGRAM_BOT_TOKEN = 'ваш_токен'
TELEGRAM_CHAT_ID = 'ваш_chat_id'

# Уведомления
TELEGRAM_NOTIFY = {
    'ENABLED': True,
    'HIGH_PNL_THRESHOLD': 100,
    'DAILY_REPORT': True,
    'REPORT_TIME': '21:00'
}
```

### data/auto_bot_config.json (для bots.py):
```json
{
  "enabled": false,
  "max_concurrent": 5,
  "rsi_long_threshold": 29,
  "rsi_short_threshold": 71,
  "stop_loss_percent": 15.0,
  "volume_mode": "usdt",
  "volume_value": 10,
  "rsi_time_filter_enabled": true,
  "check_maturity": true
}
```

---

## 🎯 Основные сценарии использования

### Сценарий 1: Только мониторинг
```bash
# Запускаем только app.py
python app.py

# Функции:
✅ Мониторинг позиций
✅ Статистика PnL
✅ Telegram уведомления
❌ Auto Bot недоступен
❌ Manual Bots недоступны
```

### Сценарий 2: Полная система
```bash
# Запускаем оба сервиса
python app.py      # Терминал 1
python bots.py     # Терминал 2

# Функции:
✅ Мониторинг позиций
✅ Статистика PnL
✅ Telegram уведомления
✅ Auto Bot
✅ Manual Bots
✅ RSI анализ
✅ Optimal EMA
```

### Сценарий 3: Только боты (без UI)
```bash
# Запускаем только bots.py
python bots.py

# Функции:
✅ Auto Bot
✅ Manual Bots (через API)
✅ RSI анализ
❌ Web UI недоступен
❌ Мониторинг UI недоступен
```

---

## 🔧 Мониторинг системы

### Проверка статуса app.py:
```bash
curl http://localhost:5000/api/positions
# Должны вернуться позиции
```

### Проверка статуса bots.py:
```bash
curl http://localhost:5001/api/status
# Должен вернуться статус системы
```

### Логи в реальном времени:
```bash
# app.py логи
tail -f logs/app.log

# bots.py логи
tail -f logs/bots.log
```

---

## 🛠️ Типичные проблемы

### Проблема: "Сервис ботов недоступен"
**Причина:** bots.py не запущен
**Решение:**
```bash
python bots.py
```

### Проблема: "Port 5000 already in use"
**Причина:** app.py уже запущен
**Решение:**
```bash
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Проблема: RSI данные не обновляются
**Причина:** Smart RSI Manager не запущен
**Решение:** Проверьте логи bots.py, перезапустите сервис

### Проблема: Auto Bot не создает ботов
**Причина:** Auto Bot выключен или нет сигналов
**Решение:**
```bash
# Проверить статус
curl http://localhost:5001/api/auto-bot/config

# Включить
curl -X POST http://localhost:5001/api/auto-bot/enable
```

---

## 📈 Производительность

### Ресурсы:
```
app.py:
- RAM: ~200-300 MB
- CPU: <2% (в покое), 5-10% (при обновлении)
- Сеть: Минимальная (только запросы к бирже)

bots.py:
- RAM: ~500-700 MB
- CPU: <5% (в покое), 10-20% (обновление RSI)
- Сеть: Средняя (загрузка RSI 583 монет)

Общие:
- Загрузка RSI: ~4-5 минут (583 монеты)
- Обновление позиций: ~1-2 секунды
- Создание бота: ~2-3 секунды
```

---

## 🔒 Безопасность

### Важно:
1. **Никогда не коммитьте** `app/config.py` с реальными ключами
2. **Используйте .gitignore** для config.py
3. **API ключи** должны иметь минимум прав (только торговля)
4. **Не открывайте** порты 5000/5001 в интернет без защиты
5. **Регулярно проверяйте** логи на подозрительную активность

### Рекомендации:
```python
# app/config.py
EXCHANGES = {
    'BYBIT': {
        'api_key': os.getenv('BYBIT_API_KEY'),  # Из переменных окружения
        'api_secret': os.getenv('BYBIT_SECRET')
    }
}
```

---

## 📚 Дополнительная документация

- **[QUICKSTART.md](QUICKSTART.md)** - Быстрый старт
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Подробная архитектура
- **[MODULES.md](MODULES.md)** - Описание модулей bots.py
- **[CONFIGURATION.md](CONFIGURATION.md)** - Настройка параметров
- **[API.md](API.md)** - Справочник API endpoints
- **[ИТОГОВАЯ_СПРАВКА.md](ИТОГОВАЯ_СПРАВКА.md)** - Итоги разбиения bots.py

---

## 🎉 Заключение

**InfoBot** - это **двухкомпонентная экосистема**:
- **app.py** = Мониторинг + UI
- **bots.py** = Торговые боты + Auto Bot

Работают **независимо**, но вместе дают полную функциональность!

**Запускайте оба сервиса для максимальной эффективности!** 🚀

---

**Дата обновления:** 15 октября 2025  
**Версия:** 1.0  
**Статус:** ✅ Система работает стабильно

