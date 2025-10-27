# Распространение Обученных ML Моделей

## 📦 Файлы с Обученными Моделями

### 1. ML Risk Predictor (`data/ai/models/`)

```
data/ai/models/
├── risk_predictor.pkl          # Обученная модель для предсказания SL/TP
├── risk_scaler.pkl              # Scaler для нормализации features
└── model_metadata.json         # Метаданные модели (версия, дата обучения)
```

#### `risk_predictor.pkl`
- **Содержит:** Обученная GradientBoosting модель
- **Предсказывает:** Оптимальные SL% и TP% на основе market features
- **Features:** RSI, volatility, trend strength, volume, price, stop history
- **Размер:** ~50-100 KB

#### `risk_scaler.pkl`
- **Содержит:** StandardScaler для нормализации features
- **Необходим:** Для корректной работы модели
- **Размер:** ~1 KB

#### `model_metadata.json`
```json
{
  "version": "1.0",
  "trained_date": "2025-10-27T00:00:00",
  "training_samples": 150,
  "model_type": "GradientBoostingRegressor",
  "features": ["rsi", "volatility", "trend_strength", "volume", "price", "coin_stops_count", "avg_stop_duration_hours"],
  "mae_sl": 2.3,
  "mae_tp": 12.5,
  "accuracy": 0.72
}
```

---

## 🎯 Как Использовать Готовые Модели

### Шаг 1: Получите обученные файлы от продавца

Продавец передает вам:
- `risk_predictor.pkl` - обученная модель
- `risk_scaler.pkl` - нормализатор
- `model_metadata.json` - метаданные (опционально)

### Шаг 2: Разместите файлы

Скопируйте файлы в вашу установку:

```bash
# Создайте директорию если нет
mkdir -p data/ai/models/

# Скопируйте файлы
cp risk_predictor.pkl data/ai/models/
cp risk_scaler.pkl data/ai/models/
cp model_metadata.json data/ai/models/
```

### Шаг 3: Проверьте лицензию

Убедитесь что лицензия Premium активна:

```bash
python scripts/activate_premium.py
```

### Шаг 4: Перезапустите систему

```bash
python bots.py
```

### Шаг 5: Проверьте что модель загружена

В логах должно появиться:

```
[MLPredictor] ✅ ML модель готова
[MLPredictor] ✅ Модель загружена
[SmartRiskManager] 🤖 ML модель подключена
```

---

## 🔄 Как Обучить Свою Модель

### Автоматическое обучение

Модель **автоматически обучается** при следующих условиях:

1. **Накопление данных:** Нужно минимум 20 закрытых позиций с feedback
2. **Вызов обучения:** В цикле каждые 6 часов (или вручную)
3. **Минимум примеров:** 20+ deals для обучения

### Ручной запуск обучения

```python
from bot_engine.ai.smart_risk_manager import SmartRiskManager

smart_risk = SmartRiskManager()
smart_risk.learn_from_feedback()  # Обучает ML модель
```

### Где хранятся данные для обучения

```
data/ai/training/
├── feedback/                    # Feedback от каждой сделки
│   ├── OG.json                   # История сделок по OG
│   ├── BTC.json                  # История сделок по BTC
│   └── ...
└── ml_training_data.json        # Собранные данные для обучения
```

---

## 📊 Как Работает Модель

### При открытии позиции:

1. **Сбор features:**
   ```python
   features = {
       'rsi': 16.7,
       'volatility': 1.2,
       'trend_strength': 0.68,
       'volume': 1000000,
       'price': 13.9387,
       'coin_stops_count': 5,
       'avg_stop_duration_hours': 12.5
   }
   ```

2. **Предсказание:**
   ```python
   prediction = ml_predictor.predict(features)
   # {'optimal_sl': 12.5, 'optimal_tp': 105.3, 'confidence': 0.75}
   ```

3. **Использование:** Бот открывает позицию с предсказанными SL/TP

### После закрытия позиции:

1. **Оценка точности:**
   ```python
   feedback = {
       'predicted_sl': 12.5,
       'predicted_tp': 105.3,
       'actual_roi': -14.99,
       'score': 0.35  # Плохо, т.к. -14.99 < -12.5
   }
   ```

2. **Сохранение:** Feedback сохраняется в `data/ai/training/feedback/{symbol}.json`

3. **Переобучение:** Когда накопится 20+ feedbacks → модель переобучается

---

## 🔐 Защита Лицензией

### Проверки лицензии:

1. **При импорте модуля:**
   ```python
   from bot_engine.ai.ml_risk_predictor import MLRiskPredictor
   # ❌ Без лицензии: ImportError
   ```

2. **При загрузке модели:**
   - Проверка лицензии при `__init__`
   - Модель не загружается без лицензии

3. **При использовании:**
   - Все методы проверяют наличие лицензии
   - Без лицензии → возвращается дефолтное предсказание

### Как передавать лицензии:

Для каждого клиента:
1. Создается уникальная лицензия: `python scripts/create_customer_license.py`
2. Передается через безопасный канал
3. Активируется: `python scripts/activate_premium.py`

---

## 📈 Мониторинг Качества Модели

### Метрики в `model_metadata.json`:

- **mae_sl:** Mean Absolute Error для SL (меньше = лучше)
- **mae_tp:** Mean Absolute Error для TP (меньше = лучше)
- **accuracy:** Точность предсказаний (0-1)
- **training_samples:** Количество примеров в обучающей выборке

### Оптимальные значения:

- **mae_sl < 3%** → Отлично
- **mae_sl < 5%** → Хорошо
- **mae_sl > 7%** → Требуется дообучение

- **accuracy > 0.7** → Отлично
- **accuracy > 0.5** → Хорошо
- **accuracy < 0.4** → Требуется дообучение

---

## 🚀 Продажа Обученных Моделей

### Опция 1: Предобученные модели

1. Обучите модель на своих данных (минимум 100+ сделок)
2. Скопируйте файлы `risk_predictor.pkl`, `risk_scaler.pkl`
3. Распространяйте вместе с Premium лицензией

### Опция 2: Сетевое обучение (будущее)

Клиенты могут подключиться к серверу для:
- Получения обновленных моделей
- Федеративного обучения (безопасная передача данных)
- Синхронизации параметров

### Опция 3: Автоматическое облако (будущее)

Система автоматически:
- Выгружает модель на ваш сервер
- Обновляет модель по мере накопления данных
- Распространяет улучшения всем клиентам

---

## 📝 Checklist для Дистрибуции

- [ ] Обучить модель минимум на 100+ сделках
- [ ] Проверить метрики качества (`model_metadata.json`)
- [ ] Упаковать файлы:
  - `risk_predictor.pkl`
  - `risk_scaler.pkl`
  - `model_metadata.json`
- [ ] Создать уникальную лицензию для клиента
- [ ] Передать файлы через безопасный канал
- [ ] Инструкции для клиента по установке

---

## 🔍 Отладка

### Модель не загружается:

```bash
# Проверить лицензию
python scripts/check_license.py

# Проверить файлы
ls -lh data/ai/models/

# Логи загрузки
grep "MLPredictor" logs/bots.log
```

### Плохие предсказания:

1. Проверьте количество feedback данных: `ls data/ai/training/feedback/`
2. Запустите переобучение: `smart_risk.learn_from_feedback()`
3. Проверьте метрики: `cat data/ai/models/model_metadata.json`

### Ошибка "Not enough training data":

Нужно накопить минимум 20 feedback записей. Откройте несколько позиций и дождитесь их закрытия.

