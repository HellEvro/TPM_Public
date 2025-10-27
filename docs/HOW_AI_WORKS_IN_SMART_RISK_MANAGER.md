# Как работает ИИ в SmartRiskManager

## 🔗 Используемые ИИ модули

### 1. 🤖 LSTM Predictor
**Что делает:**
- Предсказывает направление движения цены (UP/DOWN)
- Даёт confidence для своего предсказания

**Как используется:**
```python
lstm_prediction = self.lstm_predictor.predict(candles, current_price)
# Если предсказал правильное направление → +15% к confidence
if predicted_direction == direction:
    confidence += 0.15
```

### 2. 📊 Anomaly Detector  
**Что делает:**
- Обнаруживает аномалии в цене (pump/dump/manipulation)
- Возвращает severity (0-1)

**Как используется:**
```python
anomaly_score = self.anomaly_detector.detect(candles)
# Если аномалия → -30% от severity к confidence
if anomaly_score.get('is_anomaly'):
    confidence -= severity * 0.3
```

### 3. 🎯 Risk Manager (DynamicRiskManager)
**Что делает:**
- Оптимизирует SL/TP на основе волатильности
- Оценивает силу тренда
- Предсказывает вероятность разворота

**Как используется:**
```python
risk_analysis = self.risk_manager.calculate_dynamic_sl(symbol, candles, direction)
optimal_sl_from_risk = risk_analysis.get('sl_percent')  # Адаптивный SL

risk_tp_analysis = self.risk_manager.calculate_dynamic_tp(symbol, candles, direction)
optimal_tp_from_risk = risk_tp_analysis.get('tp_percent')  # Адаптивный TP
```

### 4. 📈 История стопов (обучение)
**Что делает:**
- Анализирует почему сделки закрылись по стопу
- Выявляет паттерны (высокий RSI, низкая волатильность, быстрые стопы)
- Корректирует SL/TP на основе истории

**Как используется:**
```python
# Получаем стопы для монеты
coin_stops = self._get_coin_stops(symbol)

if coin_stops:
    optimal_sl_from_history = self._optimal_sl_for_coin(coin_stops, volatility)
    optimal_tp_from_history = self._optimal_tp_for_coin(coin_stops, trend_strength)
```

## 🎯 Объединение рекомендаций

```python
# Взвешенное среднее: 60% ИИ рекомендации, 40% история
optimal_sl = (optimal_sl_from_risk * 0.6) + (optimal_sl_from_history * 0.4)
optimal_tp = (optimal_tp_from_risk * 0.6) + (optimal_tp_from_history * 0.4)

# Итоговый confidence:
confidence = base_confidence + lstm_bonus - anomaly_penalty
```

## 📊 Пример работы

### Без ИИ (базовая версия):
- SL: 15% (фиксировано)
- TP: 100% (фиксировано)
- Confidence: 0.5

### С ИИ (премиум):
- SL: 13.2% (LSTM 12%, History 15%, волатильность средняя)
- TP: 105% (Risk Manager 110%, History 95%)
- Confidence: 0.72 (базовый 0.5 + LSTM +0.15 + нет аномалий)
- Anomaly: False
- Trend strength: 0.68 (сильный тренд)

## 🔄 Обучение на стопах

Когда сделка закрывается по стопу:
1. Сохраняются данные: RSI на входе, волатильность, max profit, duration
2. Анализируется через LSTM - почему стоп?
3. Проверяется через Anomaly Detector - была ли аномалия?
4. Паттерн записывается для обучения: "Вход при RSI > 70 → частый стоп"

При следующем входе:
- Если текущий RSI > 70 → увеличить SL с 12% до 14%
- Если низкая волатильность → уменьшить TP с 100% до 80%
- Если быстрые стопы в истории → увеличить TP с 80% до 120%

## ✅ Преимущества

1. **Учится на ошибках** - каждый стоп делает систему умнее
2. **Адаптивные параметры** - SL/TP меняются под условия рынка
3. **Учитывает аномалии** - избегает входов при pump/dump
4. **Предсказывает направление** - LSTM даёт дополнительный сигнал
5. **Работает только с лицензией** - премиум функция

## 🔒 Без лицензии

Если лицензии нет:
- Используется только `DynamicRiskManager` (встроенный)
- Фиксированные 15% SL, 100% TP
- Без LSTM, Anomaly Detector, обучения на стопах
- Без бэктестов

