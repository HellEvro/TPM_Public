# План рефакторинга bots_manager.js

## Проблема
Файл `static/js/managers/bots_manager.js` содержит ~13500 строк и один класс `BotsManager` с ~170 методами. Файл трудно сопровождать и модифицировать.

## Подход: Модульное разбиение через Object.assign

**Стратегия:** Каждый модуль добавляет методы к `BotsManager.prototype` через `Object.assign`. Все методы остаются в одном прототипе — взаимные вызовы через `this.methodName()` работают как прежде.

**Без сборки:** Загрузка через несколько `<script>` тегов в `index.html` в нужном порядке. Не требуется webpack/rollup.

## Структура модулей

```
static/js/managers/
├── bots_manager/
│   ├── 00_core.js        # Класс, constructor, init, логирование, getTranslation
│   ├── 01_interface.js   # initializeInterface, tabs, applyReadabilityStyles
│   ├── 02_search.js      # Поиск монет
│   ├── 03_filters.js     # RSI фильтры, whitelist, blacklist
│   ├── 04_coins_display.js  # renderCoinsList, generateWarningIndicator, selectCoin...
│   ├── 05_service.js     # checkBotsService, loadCoinsRsiData, loadDelistedCoins...
│   ├── 06_bot_controls.js # createBot, stopBot, управление ботами
│   ├── 07_configuration.js # populateConfigurationForm, collectConfigurationData, config UI
│   ├── 08_history.js     # История действий, сделок, сигналов
│   ├── 09_analytics.js   # Аналитика, AI training
│   ├── 10_utils.js       # translate, showNotification, formatTimestamp...
│   └── index.js          # window.BotsManager, window.enableBotForCurrentCoin
└── bots_manager.js       # Будет заменён на тонкую обёртку (опционально) или удалён
```

## Порядок загрузки в index.html

```html
<script src=".../bots_manager/00_core.js"></script>
<script src=".../bots_manager/01_interface.js"></script>
<script src=".../bots_manager/02_search.js"></script>
<script src=".../bots_manager/03_filters.js"></script>
<script src=".../bots_manager/04_coins_display.js"></script>
<script src=".../bots_manager/05_service.js"></script>
<script src=".../bots_manager/06_bot_controls.js"></script>
<script src=".../bots_manager/07_configuration.js"></script>
<script src=".../bots_manager/08_history.js"></script>
<script src=".../bots_manager/09_analytics.js"></script>
<script src=".../bots_manager/10_utils.js"></script>
<script src=".../bots_manager/index.js"></script>
```

## Этапы внедрения

1. Создать папку `bots_manager/` и базовые файлы
2. Перенести методы по модулям, сохраняя исходный код 1:1
3. Обновить `index.html` — заменить один тег на список
4. Удалить или оставить `bots_manager.js` как заглушку (редирект)
5. Проверить: страница /bots, все вкладки, создание/остановка ботов, конфигурация
6. Commit

## Риски и митигация

| Риск | Митигация |
|------|-----------|
| Опечатка в границах кода | Точное копирование строк, проверка через поиск |
| Изменение порядка вызовов | Все методы на одном prototype, порядок не влияет |
| Циклические зависимости | Нет — один класс, один prototype |
| Кеш браузера | Добавить ?v= к скриптам при деплое |

## Откат

При проблемах: в `templates/index.html` заменить блок модулей на один тег:
```html
<script src="{{ url_for('static', filename='js/managers/bots_manager.js') }}?v=..."></script>
```

## Выполнено

- [x] Создана папка `static/js/managers/bots_manager/` с 17 модулями
- [x] `index.html` обновлён для загрузки модулей
- [x] Оригинал `bots_manager.js` сохранён как резервная копия
