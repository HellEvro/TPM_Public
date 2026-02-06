# Аудит конфигурации InfoBot: трассировка от входа до применения

**Дата аудита:** 2025-02-03

## 1. Источники конфигов

### 1.1 Файлы
| Файл | Содержимое | Обновление |
|------|------------|------------|
| `bot_engine/bot_config.py` | DEFAULT_AUTO_BOT_CONFIG, DEFAULT_BOT_CONFIG, SystemConfig, AIConfig, RiskConfig | Через config_writer.py при POST API |
| `bot_engine/bot_config.example.py` | Шаблон (НЕ используется в runtime) | Только при первом создании bot_config.py |

### 1.2 API Endpoints
| Endpoint | Метод | Назначение |
|----------|-------|------------|
| `/api/bots/auto-bot` | GET | Чтение auto_bot_config (из файла + БД whitelist/blacklist) |
| `/api/bots/auto-bot` | POST | Сохранение auto_bot_config → bots_data → config_writer → bot_config.py |
| `/api/bots/system-config` | GET/POST | SystemConfig (интервалы, Enhanced RSI, таймфрейм) |
| `/api/ai/config` | GET/POST | AIConfig, RiskConfig (AI подтверждение, self-learning) |
| `/api/bots/individual-settings/<symbol>` | GET/POST/DELETE | Индивидуальные настройки монет (leverage, volume, RSI и т.д.) |

### 1.3 UI элементы → config keys
Маппинг в `bots_manager.js` → `mapElementIdToConfigKey()`:
- `leverage` (id) → `leverage`
- `breakEvenTrigger` (id) → `break_even_trigger_percent`
- `autoBotScope` → `scope`
- и т.д.

---

## 2. Трассировка критических ключей

### 2.1 leverage (кредитное плечо)

**Цепочка:**
1. **UI:** `#leverage` (глобальный) или `#leverageCoinInput` (модалка бота)  
2. **Сохранение:** `saveTradingAndRsiExits()` → POST `/api/bots/auto-bot` с `leverage: N`  
3. **API:** `bots_data['auto_bot_config']['leverage'] = N` → `save_auto_bot_config_to_py()` → `bot_config.py`  
4. **Загрузка:** `load_auto_bot_config()` → `DEFAULT_AUTO_BOT_CONFIG` → `bots_data['auto_bot_config']`  
5. **Применение при открытии позиции:**
   - **NewTradingBot.enter_position()** → `_build_trading_bot_bridge_config()` → `leverage` в bridge_config ✅ (исправлено)
   - **TradingBot._enter_position()** → `place_order(leverage=...)` ✅
   - **TradingBot._enter_position_with_limit_orders()** → `place_order(leverage=...)` ✅
   - **open_position_for_bot()** (imports_and_globals): bots_data → individual_settings → auto_bot_config ✅

**Индивидуальные настройки:** `get_individual_coin_settings(symbol)` может переопределить leverage для конкретной монеты. Хранится в БД (bots_database).

---

### 2.2 break_even_trigger / break_even_trigger_percent

**Два ключа в конфиге** (совместимость UI):
- UI отправляет: `break_even_trigger_percent` (маппинг от `breakEvenTrigger`)
- Файл содержит: `break_even_trigger`, `break_even_trigger_percent`

**Логика использования:**
- `bot_class.py`: `config.get('break_even_trigger_percent', config.get('break_even_trigger'))` — приоритет percent
- `_build_trading_bot_bridge_config`: использует `break_even_trigger`

**Потенциальная рассинхронизация:** При сохранении только `break_even_trigger_percent` ключ `break_even_trigger` в файле не обновляется. Рекомендация: при POST синхронизировать оба ключа одним значением.

---

### 2.3 scope, whitelist, blacklist

- **scope:** только в файле (bot_config.py), НЕ в БД
- **whitelist, blacklist:** в БД (`bots_database.load_coin_filters()`)
- **load_auto_bot_config:** merge: scope из файла, whitelist/blacklist из БД

---

### 2.4 Enhanced RSI (enhanced_rsi_enabled и др.)

- **Источник:** SystemConfig в bot_config.py
- **API:** POST `/api/bots/system-config` обновляет SystemConfig
- **UI:** saveEnhancedRsi() отправляет в system-config
- **Применение:** `bot_engine.filters`, `bot_engine.indicators`, smart_rsi_manager

---

### 2.5 AI настройки (ai_enabled, ai_min_confidence, ai_override_original)

- **Auto-bot:** в DEFAULT_AUTO_BOT_CONFIG
- **AI-специфичные:** RiskConfig, AIConfig в bot_config.py
- **GET /api/bots/auto-bot:** подмешивает ai_optimal_entry_enabled, self_learning_enabled и др. из RiskConfig/AIConfig
- **POST /api/ai/config:** сохраняет AI-настройки в bot_config.py (отдельно от auto-bot)

---

## 3. Выявленные проблемы и разрывы

### 3.1 ✅ ИСПРАВЛЕНО: leverage не передавался в TradingBot
- **Было:** `_build_trading_bot_bridge_config()` не включал `leverage` → TradingBot использовал DEFAULT_BOT_CONFIG (10 или 1)
- **Исправлено:** добавлен `leverage` в bridge_config; добавлена передача leverage в place_order для лимитных ордеров

### 3.2 ✅ ИСПРАВЛЕНО: leverage = 10 по умолчанию
- config_writer.py: при отсутствии leverage добавляется `config['leverage'] = 10`
- bot_config.example.py: `'leverage': 10` в DEFAULT_AUTO_BOT_CONFIG и DEFAULT_BOT_CONFIG

### 3.3 break_even_trigger / break_even_trigger_percent
- UI шлёт только `break_even_trigger_percent`
- Рекомендация: при получении `break_even_trigger_percent` в API также обновлять `break_even_trigger` тем же значением для согласованности.

### 3.4 Элементы UI без явного маппинга
- `collectFieldsFromElements` использует `mapElementIdToConfigKey` и fallback `camelToSnake`
- Некоторые поля (limit_orders, exit_scam) собираются вручную — проверено, покрыто.

### 3.5 Индивидуальные настройки → create_bot
- При создании бота: `individual_settings.get('leverage')` или `auto_bot_config.get('leverage', 1)`
- При запуске: init_functions передаёт leverage в config бота ✅

---

## 4. Сводная таблица: ключ → где применяется

| Ключ | Вход (UI/API) | Хранение | Применение |
|------|---------------|----------|------------|
| leverage | #leverage, #leverageCoinInput | bot_config.py, individual_settings (БД) | place_order, set_leverage, _build_trading_bot_bridge_config |
| break_even_trigger | breakEvenTrigger → break_even_trigger_percent | bot_config.py | bot_class protections, TradingBot |
| scope | autoBotScope | bot_config.py | filters (whitelist/blacklist logic) |
| whitelist/blacklist | UI списков | БД | load_auto_bot_config merge |
| rsi_long_threshold, rsi_short_threshold | rsiLongThreshold, rsiShortThreshold | bot_config.py | filters, bot_class should_open_long/short |
| rsi_exit_*_with_trend, *_against_trend | rsiExit*Global | bot_config.py | bot_class should_close_position |
| default_position_size, default_position_mode | defaultPositionSize, defaultPositionMode | bot_config.py | расчёт размера позиции |
| trailing_stop_*, break_even_* | соответствующие inputs | bot_config.py | protections, TradingBot |
| enhanced_rsi_* | enhancedRsi* (system) | SystemConfig | smart_rsi, indicators |

---

## 5. Рекомендации

1. **break_even_trigger sync:** В API POST /api/bots/auto-bot при получении `break_even_trigger_percent` дополнительно устанавливать `break_even_trigger` тем же значением.
2. ✅ **DONE:** leverage по умолчанию 10 в bot_config.example.py, config_writer, init_functions, filters, UI.
3. **Проверка маппинга:** При добавлении новых полей в UI обязательно добавлять их в `mapElementIdToConfigKey` и в CONFIG_NAMES (api_endpoints.py).
4. **Логирование:** При изменении leverage в config_writer уже есть логирование; при открытии позиции — лог «Используемое плечо: Nx» ✅
