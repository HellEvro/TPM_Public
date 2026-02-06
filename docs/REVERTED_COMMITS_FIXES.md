# Коммиты после f0a0c509: что исправляли и почему откатили

**Базовый рабочий коммит:** `f0a0c509` — «Закрытие по RSI по entry_timeframe; сохранение entry_timeframe в БД при открытии позиции (фикс 1m-ботов)».

Все коммиты ниже были откачены (hard reset к f0a0c509), т.к. в сумме ломали систему (сделки не открывались, конфиг, блокировки, UI). Здесь — только суть каждого коммита и как выборочно вернуть нужные исправления без полного повторения коммита.

---

## 1. `688da3ec` — Закрытие LONG по RSI: пороги из конфига (65/60), fallback RSI по ТФ

**Что делал:**
- **bots_modules/bot_class.py:** пороги выхода по RSI брал из `auto_config` и `individual_settings` (через `get_individual_coin_settings`), с fallback на константы. Раньше использовались только константы из bot_config.
- **bots_modules/filters.py:**  
  - то же: пороги выхода из конфига/индивидуальных настроек (`exit_long_with`, `exit_long_against`, `exit_short_*`) вместо констант `RSI_EXIT_LONG_WITH_TREND` и т.д.;  
  - fallback RSI: если по таймфрейму бота RSI нет — брать RSI по системному ТФ (`get_current_timeframe()`);  
  - лог при EXIT_LONG по RSI.

**Почему ломало:** в коммите завязались на `get_current_timeframe()` и на чтение из `auto_bot_config`/individual; вместе с последующими изменениями конфига/патчинга это привело к проблемам с конфигом.

**Как внедрить безболезненно (только пороги из конфига):**
- В **filters.py** в блоке проверки выхода по RSI (около 2811–2820): вместо констант `RSI_EXIT_LONG_WITH_TREND` и т.д. брать значения так:
  - `auto_config = bots_data.get('auto_bot_config', {})`
  - `individual_settings = get_individual_coin_settings(symbol)` (уже есть или добавить импорт)
  - `exit_long_with = (individual_settings.get('rsi_exit_long_with_trend') or auto_config.get('rsi_exit_long_with_trend') or RSI_EXIT_LONG_WITH_TREND)` — и аналогично для `exit_long_against`, `exit_short_with`, `exit_short_against`.
  - Использовать эти переменные вместо констант в условиях `thr = ...`.
- В **bot_class.py** в `check_should_close_by_rsi`: пороги брать из `auto_config` и individual_settings с fallback на константы (как в патче 688da3ec), без менять остальную логику.
- Fallback RSI по системному ТФ и лишние логи из 688da3ec можно не тащить на первом шаге — только «пороги из конфига».

---

## 2. `4eec83dd` — AI настройки в bot_config.example.py + правки filters/sync_and_cache

**Что делал:**
- **bot_engine/bot_config.example.py:** добавлены AI-настройки (триггеры обучения, симуляции, `AI_USE_SAVED_SETTINGS_AS_BASE`).
- **bots_modules/filters.py:** правки логики (около 45 строк).
- **bots_modules/sync_and_cache.py:** большой рефакторинг блокировок (137 строк): сохранение состояния и обновление позиций с минимизацией времени удержания lock и/или снимками без lock.
- **exchanges/base_exchange.py, bybit_exchange.py:** добавлен `get_tickers_batch`.
- Удалены/обнулены .pyc AI-модулей (случайно).

**Почему ломало:** рефакторинг sync_and_cache (lock/save) привёл к тому, что сделки перестали открываться; откат к f0a0c509 вернул старую логику lock/save.

**Как внедрить безболезненно:**
- Из **bot_config.example.py** можно порциями переносить только недостающие AI-поля в свой example (и при необходимости в patch_bot_config), не трогая sync_and_cache.
- **sync_and_cache** и логику блокировок не повторять из этого коммита; при необходимости уменьшать contention только точечно (например, только вынос сетевых вызовов из-под lock в одном месте), с тестами.
- **get_tickers_batch** в base/bybit можно оставить как есть (уже есть в текущем коде после отката или добавить отдельным минимальным патчем без изменения логики сохранения).

---

## 3. `555bdfd6` — Удалён дубликат example.bot_config.py

**Что делал:** удалён файл `bot_engine/example.bot_config.py`, оставлен только `bot_config.example.py`.

**Риск:** низкий; единственное — если что-то явно импортирует `example.bot_config`, это нужно заменить на `bot_config.example` или на логику без этого файла.

**Как внедрить:** убедиться, что нигде нет импорта/упоминания `example.bot_config.py`, после чего файл можно не восстанавливать (оставить один example).

---

## 4. `5ee7eb80` — bot_config.py убран из git, автопатчинг конфигурации

**Что делал:**
- **.gitignore:** добавлен bot_config.py.
- **app.py:** правки (9 строк).
- **bot_engine/__init__.py:** автопатчинг конфига при импорте (73 строки) — вызов patch_bot_config при загрузке.
- **bot_engine/bot_config.py:** удалён из репо (690 строк удалено).
- **scripts/patch_bot_config.py:** новый скрипт (315 строк) — дополнение bot_config из example.
- **scripts/sync_to_public.py:** исключение bot_config из синка.

**Почему ломало:** автопатч при импорте и удаление bot_config из репо меняли поведение при старте и при обновлении конфига; вместе с последующими коммитами это приводило к «конфиг не работает».

**Как внедрить безболезненно:**
- Не подключать автопатч при импорте в `bot_engine/__init__.py` до стабилизации конфига.
- Скрипт `patch_bot_config.py` можно держать и запускать вручную или из CI, без вызова при каждом импорте.
- bot_config.py в git можно не коммитить (оставить в .gitignore), но локально файл должен быть и подгружаться как сейчас в f0a0c509.

---

## 5. `01c5a439` — Функции таймфрейма в bot_config.example.py + правки sync_and_cache

**Что делал:**
- **bot_engine/bot_config.example.py:** добавлены `get_current_timeframe`, `set_current_timeframe`, `reset_timeframe_to_config`, `get_rsi_key`, `get_trend_key`, `get_timeframe_suffix`, `get_rsi_from_coin_data`, `get_trend_from_coin_data` (134 строки).
- **bots_modules/sync_and_cache.py:** изменения в логике (55 строк, упрощение/рефакторинг).

**Почему ломало:** зависимость от этих функций в других коммитах (например, 688da3ec использует `get_current_timeframe` в filters); изменение sync_and_cache могло влиять на сохранение/блокировки.

**Как внедрить безболезненно:**
- В **bot_config.example.py** (и при необходимости в реальный bot_config) можно добавить только недостающие функции таймфрейма/RSI-ключей по одной, проверяя после каждого шага.
- Из **sync_and_cache** из этого коммита ничего не восстанавливать без отдельного анализа; оставить логику f0a0c509.

---

## 6. `0b199d12` — patch_bot_config: функции верхнего уровня + автопатч, ai_manager заглушка

**Что делал:**
- **bot_engine/__init__.py:** автопатч с новой сигнатурой (parse/generate/apply с top_level функциями).
- **bot_engine/ai/__init__.py:** заглушка `get_ai_manager` при отсутствии ai_manager, импорт `from bot_engine.ai import get_ai_manager` вместо `from bot_engine.ai.ai_manager import ...`.
- **bot_engine/ai/auto_trainer.py, api/endpoints_ai.py, trading_bot.py, bots.py, filters.py:** замена импорта на `from bot_engine.ai import get_ai_manager`.
- **scripts/patch_bot_config.py:** поддержка добавления функций верхнего уровня из example.

**Почему ломало:** автопатч и массовая замена импортов меняли порядок загрузки и зависимости; вместе с отказом конфига и lock-логикой это усугубляло сбои.

**Как внедрить безболезненно:**
- Импорт `from bot_engine.ai import get_ai_manager` в остальных модулях — полезен, если ai_manager может отсутствовать; можно вернуть только эти изменения (один коммит «импорт ai из пакета»), без автопатча и без правок patch_bot_config.
- Заглушку в **bot_engine/ai/__init__.py** (get_ai_manager при отсутствии модуля) можно вернуть отдельно.
- Автопатч при импорте и расширение patch_bot_config на функции верхнего уровня — не восстанавливать до стабилизации конфига и деплоя.

---

## 7. `8c551f8f` — Восстановлены .pyc AI модулей

**Что делал:** восстановлены бинарные файлы `_ai_launcher.pyc`, `ai_manager.pyc`, `hardware_id_source.pyc`, `license_checker.pyc`.

**Риск:** только если у тебя эти .pyc не в репо по политике; тогда не коммитить, а восстанавливать из бэкапа при необходимости.

**Как внедрить:** при необходимости скопировать .pyc из бэкапа/другой ветки в `bot_engine/ai/` и не трогать остальной код.

---

## 8. `77ce259e` — Улучшение загрузки данных на вкладке «Фильтры монет»

**Что делал:** **static/js/managers/bots_manager.js** — вызов `loadCoinsRsiData()` при переключении на вкладку «Фильтры монет», обновление результатов поиска после загрузки, проверка `data.coins` (18 строк).

**Почему откатили:** откат был общий до f0a0c509; сам по себе этот коммит только UI.

**Как внедрить безболезненно:** выборочно применить только изменения в `bots_manager.js` для вкладки «Фильтры монет» (подгрузка монет при открытии вкладки и обновление поиска) — без правок бэкенда и конфига.

---

## План пошагового внедрения (как договорились)

**Текущая база:** f0a0c509. Выполнять строго по порядку.

---

### 1. `688da3ec` — Пороги RSI из конфига (65/60), без fallback по ТФ

**Делать:** внедрить как предлагалось — только пороги выхода из конфига.

- **filters.py:** в блоке проверки выхода по RSI брать `exit_long_with`, `exit_long_against`, `exit_short_with`, `exit_short_against` из `auto_config` и `individual_settings`, fallback на константы `RSI_EXIT_*`. Не добавлять fallback RSI по `get_current_timeframe()`.
- **bot_class.py:** в `check_should_close_by_rsi` пороги брать из individual_settings → bot_data → auto_config → константы (helper `_thresh`).
- Не трогать sync_and_cache, не добавлять вызовы `get_current_timeframe` в filters.

---

### 2. `4eec83dd` — AI настройки в bot_config.example.py (без правок filters/sync_and_cache)

**Делать:** только AI-настройки в example, с автонастройкой.

- **bot_engine/bot_config.example.py:** добавить блок AI-конфига (триггеры обучения, симуляции, `AI_USE_SAVED_SETTINGS_AS_BASE` и т.д.) — как было с автонастройкой.
- **Не делать:** правки filters.py и sync_and_cache.py из этого коммита. Конфиг из репо изначально исключили — чиним только последствия через example.

---

### 3. `555bdfd6` — Удалить дубликат example.bot_config.py

**Делать:** обязательно.

- Удалить файл **bot_engine/example.bot_config.py**.
- Во всех местах, где используется `example.bot_config.py`, заменить на **bot_config.example.py** (bots.py, launcher, .cursorrules и т.д.).
- Проверить, что нет импортов `example.bot_config`.

---

### 4. `5ee7eb80` — bot_config.py убрать из git + автопатч

**Делать:** обязательно, но только после п.2 (автонастройка в example).

- **.gitignore:** добавить `bot_engine/bot_config.py`.
- **bot_engine/__init__.py:** при импорте вызывать автопатч (создание/дополнение bot_config из example).
- **scripts/patch_bot_config.py:** скрипт, который только дополняет классы из example (без добавления функций верхнего уровня).
- **scripts/sync_to_public.py:** исключить bot_config из синка.
- При необходимости: `git rm --cached bot_engine/bot_config.py` и коммит.

---

### 5. `01c5a439` — НЕ внедрять (функции таймфрейма + sync_and_cache)

**Не делать:** именно это сильнее всего ломало систему, давало кучу ошибок.

- **Не добавлять** в bot_config.example.py и в bot_config функции: `get_current_timeframe`, `set_current_timeframe`, `reset_timeframe_to_config`, `get_rsi_key`, `get_trend_key`, `get_timeframe_suffix`, `get_rsi_from_coin_data`, `get_trend_from_coin_data`.
- Их внедряли из-за того, что п.1 был реализован неправильно и в конфиг добавляли рабочую логику (которая уже была реализована в коде). Сейчас п.1 делаем без этих функций — проблем не будет.
- Из **sync_and_cache** из этого коммита ничего не восстанавливать.

---

### 6. `0b199d12` — Заглушка ai_manager + импорт из пакета (без патча функций)

**Делать:** по предложенному варианту.

- **bot_engine/ai/__init__.py:** заглушка `get_ai_manager` при отсутствии ai_manager (возвращать объект с `is_available() == False` и т.д.).
- Во всех модулях заменить `from bot_engine.ai.ai_manager import get_ai_manager` на **`from bot_engine.ai import get_ai_manager`**.
- **Не делать:** добавление в patch_bot_config поддержки функций верхнего уровня из example (см. п.5).

---

### 7. `8c551f8f` — .pyc AI модулей не удалять

**Делать:** ничего не менять в коде — просто не удалять .pyc модули, они сейчас работают.

---

### 8. `77ce259e` — Загрузка данных на вкладке «Фильтры монет»

**Делать:** по предложенному варианту.

- **static/js/managers/bots_manager.js:** при переключении на вкладку «Фильтры монет» вызывать `loadCoinsRsiData()`, обновлять результаты поиска после загрузки, проверять наличие `data.coins` в ответе API.

---

## Краткий чеклист порядка

| № | Коммит    | Действие |
|---|-----------|----------|
| 1 | 688da3ec  | Пороги RSI из конфига (filters + bot_class), без get_current_timeframe |
| 2 | 4eec83dd  | AI настройки только в bot_config.example.py |
| 3 | 555bdfd6  | Удалить example.bot_config.py, поправить ссылки |
| 4 | 5ee7eb80  | bot_config в .gitignore + автопатч (после п.2) |
| 5 | 01c5a439  | **Не внедрять** функции таймфрейма и правки sync_and_cache |
| 6 | 0b199d12  | ai_manager заглушка + импорт из bot_engine.ai |
| 7 | 8c551f8f  | Не удалять .pyc |
| 8 | 77ce259e  | loadCoinsRsiData на вкладке «Фильтры монет» |

---

## Очистка «зоопарка веток» в Git

Если история веток запуталась (множество merge, detached HEAD, «14↑ 4↓») и нужно оставить только последнее состояние в одном линейном коммите:

```batch
scripts\fix_git_clean_history.bat
scripts\fix_git_clean_history.bat \\EVROMINI\InfoBot
REM или напрямую .cmd (без PowerShell):
scripts\fix_git_clean_history.cmd \\EVROMINI\InfoBot
```

Скрипт:
- Снимает `skip-worktree` с `bot_config.py`, делает резервную копию
- Создаёт orphan-ветку с одним коммитом из текущего состояния
- Заменяет `main` на линейную историю
- Восстанавливает `bot_config.py` и `skip-worktree`

После выполнения: `git push origin main --force` (осторожно — перезаписывает историю на сервере).

---

*Документ обновлён после hard reset к f0a0c509. Текущая база — f0a0c509.*
