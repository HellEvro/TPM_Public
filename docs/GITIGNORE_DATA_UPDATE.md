# Обновление .gitignore для сохранения данных

## Дата: 18 октября 2025

## Проблема

Ранее директория `data/` была полностью исключена из Git, что приводило к потере важных данных:
- Обученные AI модели
- Исторические данные для обучения
- Конфигурации системы

## Решение

Обновлена конфигурация `.gitignore` для сохранения всех критичных данных в Git.

## Изменения в .gitignore

### 1. Раскомментирована директория data/
```gitignore
# Data files (trading data, positions, etc.)
# data/  ← Было закомментировано
```

### 2. Добавлены исключения для AI данных
```gitignore
!data/ai/
!data/ai/models/
!data/ai/models/*.pkl      # Anomaly Detector, scalers
!data/ai/models/*.h5       # Старый формат LSTM
!data/ai/models/*.keras    # Новый формат LSTM (Keras 3)
!data/ai/models/*.pt       # PyTorch модели
!data/ai/models/*.pth      # PyTorch модели (альтернативное расширение)
!data/ai/models/*.onnx     # ONNX модели
!data/ai/models/*.json     # Конфигурации моделей
```

### 3. Исторические данные и обучение
```gitignore
!data/ai/historical/       # Исторические данные
!data/ai/historical/*.csv  # CSV файлы с историей
!data/ai/training/         # Данные для обучения
```

### 4. Раскомментированы базы данных
```gitignore
# *.db         ← Теперь базы данных сохраняются
# *.sqlite     ← Если будут использоваться
# *.sqlite3
```

## Что теперь сохраняется в Git

### AI Модели (критично!)
- ✅ `data/ai/models/anomaly_detector.pkl` (836 KB)
- ✅ `data/ai/models/anomaly_scaler.pkl` (959 B)
- ✅ `data/ai/models/lstm_predictor.keras` (новый формат)
- ✅ `data/ai/models/lstm_scaler.pkl` (132 B)
- ✅ `data/ai/models/lstm_config.json` (285 B)

### Конфигурации системы
- ✅ `data/async_state.json`
- ✅ `data/bots_state.json`
- ✅ `data/default_auto_bot_config.json`
- ✅ `data/optimal_ema.json`
- ✅ `data/process_state.json`

### Исторические данные
- ✅ `data/ai/historical/*.csv` (для обучения моделей)

## Преимущества

1. **Безопасность данных** - критичные модели и конфигурации не будут потеряны
2. **Воспроизводимость** - можно развернуть систему на новом сервере со всеми моделями
3. **История изменений** - Git отслеживает изменения в конфигурациях
4. **Коллаборация** - другие разработчики получат все необходимые данные

## Что НЕ сохраняется

### Временные файлы
- ❌ `*.tmp`, `*.temp`
- ❌ `*.backup`, `*.bak`
- ❌ `*.json.backup`, `*.json.bak`

### Логи
- ❌ `*.log`
- ❌ `logs/`

### Кэш
- ❌ `*.cache`
- ❌ `cache/`

### Большие файлы обучения
- ❌ `data/ai/training/*.npz` (могут быть очень большими)

## Миграция на новый формат

После обновления `.gitignore`:

1. **Добавить все AI модели в Git:**
   ```bash
   git add -f data/ai/models/*.pkl
   git add -f data/ai/models/*.keras
   git add -f data/ai/models/*.json
   ```

2. **Добавить конфигурации:**
   ```bash
   git add data/*.json
   ```

3. **Проверить статус:**
   ```bash
   git status data/
   ```

## Рекомендации

### Для разработки
- Регулярно коммитьте обученные модели после улучшения
- Добавляйте осмысленные commit messages при обновлении моделей
- Используйте теги Git для важных версий моделей

### Для продакшена
- Убедитесь, что все модели присутствуют перед деплоем
- Проверьте целостность моделей после клонирования репозитория
- Сохраняйте резервные копии критичных моделей

## Связанные документы

- `docs/LSTM_KERAS3_MIGRATION.md` - Миграция на новый формат Keras
- `.gitignore` - Основной файл конфигурации
- `docs/AI_MODULES.md` - Документация AI модулей

## Статус

✅ Конфигурация обновлена
✅ AI модели в безопасности
✅ Данные сохраняются в Git

