# Аудит конфигов и цепочек загрузки/использования

Документ описывает **каждый конфиг**, **откуда** он берётся, **куда** попадает и **результат операции**. Цель — найти нарушения логики и рассинхрон.

---

## 1. ИСТОЧНИКИ КОНФИГУРАЦИИ (файлы)

| Файл | Назначение |
|------|------------|
| `configs/bot_config.py` | **Рабочий конфиг**. Классы: DefaultAutoBotConfig (шаблон сброса), AutoBotConfig (текущие настройки), SystemConfig, RiskConfig, FilterConfig, ExchangeConfig, AIConfig. |
| `configs/bot_config.example.py` | Пример/шаблон. Копируется в bot_config.py при первом запуске или при отсутствии bot_config.py. |
| `configs/keys.example.py` | Пример ключей API. Реальные ключи — в configs/keys.py (не в репо). |
| `configs/app_config.example.py` | Пример настроек приложения. |
| `bot_engine/config_loader.py` | **Единая точка входа**: импортирует классы из configs.bot_config, превращает в словари (ключи lower), экспортирует get_config_value, get_current_timeframe, get_rsi_from_coin_data и т.д. |
| `bot_engine/bot_config.py` | Создаётся из bot_engine/bot_config.example.py при первом импорте bot_engine. Содержит дубликаты get_rsi_from_coin_data → делегируют в config_loader. **Не используется для загрузки auto_bot_config.** |
| `config/.env` | Не используется основным приложением (см. configs/__init__.py). |
| БД (bots_database) | Хранит: whitelist/blacklist, individual_coin_settings, bots_state, rsi_cache и т.д. Таймфрейм — только в конфиге. |

---

## 2. ЦЕПОЧКА: Auto Bot Config (RSI, лимиты, торговля)

### 2.1 Загрузка в память

```
ОТКУДА:
  configs/bot_config.py
    → класс DefaultAutoBotConfig (атрибуты ENABLED, RSI_LONG_THRESHOLD, ...)
    → класс AutoBotConfig (рабочие значения; при сохранении из UI пишутся сюда)

КАК:
  1) bot_engine.config_loader при импорте:
     - from configs.bot_config import DefaultAutoBotConfig, AutoBotConfig, ...
     - DEFAULT_AUTO_BOT_CONFIG = config_class_to_dict(DefaultAutoBotConfig)  # ключи lower
     - AUTO_BOT_CONFIG = config_class_to_dict(AutoBotConfig)

  2) bots_modules.imports_and_globals.load_auto_bot_config():
     - reload_config()  # importlib.reload(configs.bot_config), importlib.reload(config_loader)
     - file_config = load_auto_bot_config_from_file(config_file_path)
       → config_writer читает configs/bot_config.py, находит блок "class AutoBotConfig",
         парсит строки вида "    ATTR = value  # comment" через _parse_attr_line(),
         возвращает dict {key_lower: value} (только то, что распарсилось)
     - merged_config = DEFAULT_AUTO_BOT_CONFIG.copy()
     - for key, value in file_config.items(): merged_config[key] = value
     - из БД подмешиваются whitelist, blacklist (scope остаётся из файла)
     - with bots_data_lock: bots_data['auto_bot_config'] = merged_config

КУДА:
  bots_data['auto_bot_config']  # глобальный словарь в imports_and_globals

РЕЗУЛЬТАТ:
  В памяти всегда актуальный merge: дефолты из DefaultAutoBotConfig + значения из блока AutoBotConfig в файле + whitelist/blacklist из БД.
```

**Возможное нарушение логики:**
- Если в `configs/bot_config.py` строка атрибута не распознаётся парсером (многострочное значение, нестандартный формат), ключ не попадёт в `file_config` → в merge останется значение из `DefaultAutoBotConfig`. Нужно, чтобы все строки в блоке AutoBotConfig были вида `    ATTR = value` или `    ATTR = value  # comment`.

---

## 3. ЦЕПОЧКА: Получение значения параметра (get_config_value)

```
ОТКУДА:
  config_dict — обычно bots_data['auto_bot_config'] (уже merge из п.2)

КАК:
  bot_engine.config_loader.get_config_value(config_dict, key):
    - if not config_dict: return DEFAULT_AUTO_BOT_CONFIG.get(key)
    - val = config_dict.get(key)
    - if val is not None: return val
    - return DEFAULT_AUTO_BOT_CONFIG.get(key)

КУДА:
  Вызывающий код (filters, bot_class, api_endpoints, …) получает число/строку/список.

РЕЗУЛЬТАТ:
  Значение из переданного словаря или fallback из DEFAULT_AUTO_BOT_CONFIG (после reload — из configs.bot_config.DefaultAutoBotConfig).
```

**Нарушение логики:** Нет, если везде передаётся актуальный `bots_data['auto_bot_config']` или merge с individual_settings (см. п.5).

---

## 4. ЦЕПОЧКА: Таймфрейм (system_timeframe)

**Единый источник: только configs/bot_config.py. В БД таймфрейм не хранится.**

```
ОТКУДА (по приоритету):
  1) bot_engine.config_loader._current_timeframe (кэш в памяти после загрузки из файла)
  2) config_loader._get_default_timeframe() = AutoBotConfig.SYSTEM_TIMEFRAME (из файла), fallback DefaultAutoBotConfig
  3) config_loader.TIMEFRAME = _get_default_timeframe() or SystemConfig.SYSTEM_TIMEFRAME (для обратной совместимости)

КАК:
  - get_current_timeframe(): если _current_timeframe is not None → return it; иначе return _get_default_timeframe() (из конфига).
  - set_current_timeframe(tf): устанавливает кэш в памяти; при смене ТФ в UI вызывается save_auto_bot_config_current_to_py({'system_timeframe': tf}) и save_system_config_to_py({'SYSTEM_TIMEFRAME': tf}) — запись в файл.
  - При старте и после reload таймфрейм восстанавливается из файла (AutoBotConfig.SYSTEM_TIMEFRAME).

КУДА:
  - Все вызовы get_rsi_key(timeframe), get_rsi_from_coin_data(coin_data, timeframe=…), get_current_timeframe() в фильтрах, автоботе, RSI-воркерах.

РЕЗУЛЬТАТ:
  Текущий таймфрейм только в конфиге; при перезапуске читается из configs/bot_config.py. Рассинхрон с БД устранён.
```

**Согласованность:** При сохранении ТФ из UI обновляются оба блока в файле (AutoBotConfig и SystemConfig), чтобы значение было одним.

---

## 5. ЦЕПОЧКА: Индивидуальные настройки монеты (individual_coin_settings)

```
ОТКУДА:
  - БД: таблица individual_coin_settings (через bot_engine.storage.load_individual_coin_settings → db.load_individual_coin_settings)
  - При первом обращении load_individual_coin_settings() загружает из БД в bots_data['individual_coin_settings']

КАК:
  - get_individual_coin_settings(symbol): нормализует symbol (upper), при необходимости перезагружает из storage (по mtime файла/БД), возвращает bots_data['individual_coin_settings'].get(symbol).
  - set_individual_coin_settings(symbol, settings, persist=True): пишет в bots_data['individual_coin_settings'][symbol], при persist вызывает storage_save_individual_coin_settings → БД.

КУДА:
  - get_config_snapshot(symbol): merged_config = deepcopy(global_config); if individual_settings: merged_config.update(individual_settings).
  - В фильтрах и bot_class: (individual_settings.get('rsi_long_threshold') or get_config_value(auto_config, 'rsi_long_threshold')).

РЕЗУЛЬТАТ:
  Для данной монеты подставляются переопределения (rsi_long_threshold, leverage и т.д.); иначе используются глобальные из auto_bot_config.
```

**Нарушение логики:** Если где-то читают только `auto_bot_config` без учёта individual_settings для символа — будет рассинхрон (например, глобально 18, для монеты 25 — без individual получим 18).

---

## 6. ЦЕПОЧКА: RSI по таймфрейму (критично для входа LONG/SHORT)

```
ОТКУДА:
  coins_rsi_data['coins'][symbol] — словарь с ключами rsi1m, rsi6h, trend1m, trend6h, … (заполняется воркерами RSI из биржи и расчётов).

КАК:
  bot_engine.config_loader.get_rsi_from_coin_data(coin_data, timeframe=None):
    - tf = timeframe or get_current_timeframe()
    - return coin_data.get(get_rsi_key(tf))   # только этот ключ, БЕЗ fallback на rsi6h/rsi

КУДА:
  - filters.get_effective_signal(coin): rsi = get_rsi_from_coin_data(coin, current_timeframe); если None → return 'WAIT'.
  - process_auto_bot_signals: rsi = get_rsi_from_coin_data(coin_data, timeframe=current_timeframe); проверка signal по порогам (LONG ≤ long_th, SHORT ≥ short_th).
  - bot_class.should_open_long / should_open_short: получают rsi уже из вызывающего кода (из той же цепочки).

РЕЗУЛЬТАТ:
  Решение о входе принимается только по RSI того таймфрейма, на котором работает пользователь (например 1m). Fallback на rsi6h убран, чтобы не открывать SHORT при 1m RSI 13 из-за 6h RSI 85.
```

**Нарушение логики (исправлено):** Раньше был fallback на rsi6h → подставлялся RSI другого ТФ. Сейчас — только запрошенный ТФ.

---

## 7. ЦЕПОЧКА: Сохранение настроек из UI в файл

```
ОТКУДА:
  API/UI отправляет payload (например rsi_long_threshold=18, rsi_short_threshold=81).

КАК:
  - api_endpoints или config_writer: save_auto_bot_config_current_to_py(config).
  - config_writer находит в configs/bot_config.py блок "class AutoBotConfig", для каждого ключа из config находит строку с атрибутом (по имени В ВЕРХНЕМ РЕГИСТРЕ), заменяет только значение, сохраняет файл.
  - После записи вызывается load_auto_bot_config() (или reload), чтобы bots_data['auto_bot_config'] обновился.

КУДА:
  configs/bot_config.py (класс AutoBotConfig), затем bots_data['auto_bot_config'].

РЕЗУЛЬТАТ:
  Файл и память синхронны. Следующий цикл автобота использует новые пороги.
```

**Возможное нарушение:** Если в файле атрибут записан в другом регистре или с опечаткой, парсер при следующей загрузке может не найти ключ → подставятся дефолты.

---

## 8. ЦЕПОЧКА: AIConfig (AI модули)

```
ОТКУДА:
  configs/bot_config.py → класс AIConfig (AI_ENABLED, AI_ANOMALY_DETECTION_ENABLED, …).

КАК:
  - bot_engine.config_live.get_ai_config_attr(name, default): после reload_bot_config_if_changed() импортирует AIConfig из config_loader, возвращает getattr(AIConfig, name, default).
  - reload_bot_config_if_changed() по mtime configs/bot_config.py вызывает config_loader.reload_config().

КУДА:
  Проверки ai_enabled, anomaly и т.д. в filters, bot_class, api.

РЕЗУЛЬТАТ:
  Включение/выключение AI и аномалий из того же файла configs/bot_config.py.
```

---

## 9. ЦЕПОЧКА: SystemConfig (порты, интервалы, RSI-константы для индикаторов)

```
ОТКУДА:
  configs/bot_config.py → класс SystemConfig (RSI_OVERSOLD, RSI_OVERBOUGHT, SYSTEM_TIMEFRAME, BOTS_SERVICE_PORT, …).

КАК:
  - config_loader при импорте: RSI_OVERSOLD = SystemConfig.RSI_OVERSOLD, TIMEFRAME = getattr(SystemConfig, 'SYSTEM_TIMEFRAME', None), и т.д.
  - Код импортирует из config_loader: from bot_engine.config_loader import RSI_OVERSOLD, TIMEFRAME, SystemConfig.

КУДА:
  Индикаторы, AI, бэктестеры — константы RSI/EMA. Таймфрейм по умолчанию — см. п.4.

РЕЗУЛЬТАТ:
  Единый источник для констант; после reload_config() значения из файла.
```

**Замечание:** Для решений о входе автобота используются пороги из **AutoBotConfig** (через bots_data['auto_bot_config'] и get_config_value), а не SystemConfig.RSI_OVERSOLD/OVERBOUGHT. SystemConfig — fallback для индикаторов и старых путей.

---

## 10. Сводка: откуда берутся RSI-пороги при входе LONG/SHORT

| Этап | Источник | Ключ/переменная | Пример значения |
|------|----------|------------------|------------------|
| Файл | configs/bot_config.py, класс AutoBotConfig | RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD | 18, 81 |
| Парсинг | load_auto_bot_config_from_file() | rsi_long_threshold, rsi_short_threshold | 18, 81 |
| Память | bots_data['auto_bot_config'] | rsi_long_threshold, rsi_short_threshold | 18, 81 |
| С индивидуальными | get_config_snapshot(symbol)['merged'] или individual_settings.get(…) or get_config_value(auto_config, …) | rsi_long_threshold, rsi_short_threshold | 18 или override по монете |
| Проверка входа | should_open_long(rsi, …): rsi <= rsi_long_threshold; should_open_short: rsi >= rsi_short_threshold | — | LONG только при RSI ≤ 18, SHORT только при RSI ≥ 81 |
| RSI для сравнения | get_rsi_from_coin_data(coin_data, timeframe=current_timeframe) | ключ rsi1m / rsi6h / … | Только по текущему ТФ, без fallback |

Если на каком-то шаге подставляется старый конфиг, другой таймфрейм или individual не учитывается — возможен неверный вход (например SHORT при низком RSI). Все исправления из предыдущих ответов (без fallback rsi6h, проверка порогов в get_effective_signal, повторная проверка перед enter_position) держат эту цепочку согласованной.

---

## 11. Рекомендации

1. **Один источник истины для автобота:** Только `configs/bot_config.py` (AutoBotConfig + DefaultAutoBotConfig) и `bots_data['auto_bot_config']` после load_auto_bot_config. Не дублировать пороги в хардкодах.
2. **Парсер конфига:** Следить, чтобы все атрибуты в блоке AutoBotConfig были в формате `    ATTR = value` или `    ATTR = value  # comment`, и не было вложенных классов в том же блоке.
3. **Таймфрейм:** Хранить в БД и восстанавливать при загрузке; для get_current_timeframe использовать _current_timeframe или DefaultAutoBotConfig.system_timeframe.
4. **Индивидуальные настройки:** Везде, где нужны пороги/лимиты по монете, использовать get_config_value(auto_config, key) с предварительным merge individual_settings или get_config_snapshot(symbol)['merged'].
5. **Регрессия RSI:** Тест tests/test_rsi_no_fallback.py проверяет отсутствие fallback на rsi6h при timeframe='1m'; не удалять и не ослаблять.

Если нужно, можно добавить в этот файл точечные проверки (чеклист) для деплоя или скрипт, который сверяет ключи в файле и в merged_config после load_auto_bot_config.

---

## 12. Потенциальные несоответствия (проверить вручную)

| Что проверить | Где | Риск |
|---------------|-----|------|
| AutoBotConfig.ENABLED = False в файле | configs/bot_config.py, класс AutoBotConfig | После перезапуска автобот будет выключен, пока не включат в UI. |
| DefaultAutoBotConfig vs AutoBotConfig | Оба в одном файле | При «Сбросить к стандарту» подставляются значения Default; при загрузке из файла парсится только блок AutoBotConfig — дефолты берутся из DEFAULT_AUTO_BOT_CONFIG (после reload). |
| Таймфрейм: SystemConfig vs DefaultAutoBotConfig | config_loader: _get_default_timeframe() vs TIMEFRAME | Оба из одного файла; если в одном классе поменять, в другом забыть — возможен рассинхрон. Сейчас в bot_config.py оба '1m'. |
| Individual settings: БД vs bots_data | load_individual_coin_settings() | При старте нужно вызвать load_individual_coin_settings(), иначе individual_coin_settings в памяти пустые. |
| Парсер: только строка вида `ATTR = value` | config_writer._parse_attr_line | Многострочные или экзотические значения не распарсятся → ключ не попадёт в file_config → используется дефолт. |
