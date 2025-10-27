# Как работает обучение ИИ на стопах

## 🎯 Цель обучения

ИИ должен научиться:
- **Предсказывать оптимальный SL/TP** для каждой монеты
- **Понимать, когда его предсказания были правильными**
- **Корректировать свои параметры** на основе опыта

## 📊 Текущее состояние: ЧТО РАБОТАЕТ

### 1. ✅ Сбор данных при закрытии позиции

При закрытии позиции (особенно по стопу) сохраняются:
```python
entry_data = {
    'entry_price': 13.9387,
    'rsi': 16.7,  # RSI на входе
    'volatility': 1.2,  # Волатильность
    'trend': 'UP',  # Тренд на входе
    'duration_hours': 12.5,  # Длительность позиции
    'max_profit_achieved': 3.2  # Максимальная прибыль %
}
```

### 2. ✅ Анализ паттернов в стопах

Находит общие причины стопов:
```python
patterns = {
    'high_rsi_stops': 15,  # 15 стопов когда RSI был >70
    'low_volatility_stops': 8,  # 8 стопов при низкой волатильности
    'rapid_stops': 23,  # 23 стопа закрылись быстро (<6ч)
    'trailing_stops': 5  # 5 стопов по trailing
}
```

### 3. ❌ НЕТ: Обратной связи (обучения)

**Проблема:** ИИ не знает, правильно ли он предложил SL/TP!

**Пример:**
1. ИИ предложил: SL=12%, TP=100%
2. Бот открыл позицию с этими параметрами
3. Позиция закрылась по стопу при -15%
4. **Но ИИ не знает**, что его рекомендация была неверной!

## 🔧 ЧТО НУЖНО ДОБАВИТЬ

### Система обратной связи

```python
class SmartRiskManager:
    
    def evaluate_prediction(self, symbol, backtest_result, actual_outcome):
        """
        Оценивает насколько правильно ИИ предсказал SL/TP
        
        Args:
            symbol: Символ монеты
            backtest_result: Что ИИ предсказал (из backtest_coin)
            actual_outcome: Что реально произошло
        
        Returns:
            Оценка: {'correct': True/False, 'score': 0-1}
        """
        
        # Получаем предсказание
        predicted_sl = backtest_result['optimal_sl_percent']
        predicted_tp = backtest_result['optimal_tp_percent']
        
        # Получаем реальный результат
        actual_entry = actual_outcome['entry_price']
        actual_exit = actual_outcome['exit_price']
        actual_result = actual_outcome['roi']
        
        # Оцениваем точность
        # Если позиция закрылась между predicted_sl и predicted_tp → хорошо
        # Если вышла за пределы → плохо
        
        # Сохраняем для обучения
        self._save_feedback(symbol, {
            'predicted_sl': predicted_sl,
            'predicted_tp': predicted_tp,
            'actual_result': actual_result,
            'score': self._calculate_score(predicted_sl, predicted_tp, actual_result)
        })
    
    def _save_feedback(self, symbol, feedback):
        """Сохраняет обратную связь для обучения"""
        # Сохраняем в файл для последующего обучения моделей
        # data/ai/training/feedback_{symbol}.json
```

### Обучение на основе обратной связи

```python
def learn_from_feedback(self):
    """Обучается на основе обратной связи"""
    
    # Загружаем все feedback
    all_feedback = self._load_all_feedback()
    
    # Анализируем успешность
    for symbol, feedbacks in all_feedback.items():
        # Средний score для этой монеты
        avg_score = np.mean([f['score'] for f in feedbacks])
        
        # Если score низкий → корректируем параметры
        if avg_score < 0.5:
            # Увеличиваем SL для этой монеты
            self._adjust_parameters(symbol, {
                'sl_multiplier': 1.2,  # +20% к SL
                'tp_multiplier': 0.9   # -10% к TP
            })
```

## 💾 Куда сохраняется опыт?

### 1. История стопов (`data/bot_history.json`)
```json
{
  "trades": [
    {
      "symbol": "OG",
      "direction": "LONG",
      "entry_price": 13.9387,
      "exit_price": 13.7715,
      "roi": -14.99,
      "close_reason": "STOP_LOSS_-15%",
      "entry_data": {
        "rsi": 16.7,
        "volatility": 1.2,
        "trend": "UP"
      }
    }
  ]
}
```

### 2. Обратная связь (`data/ai/training/feedback_*.json`)
```json
{
  "symbol": "OG",
  "feedback": [
    {
      "predicted_sl": 12,
      "predicted_tp": 100,
      "actual_result": -14.99,
      "score": 0.3,
      "timestamp": "2025-10-27T04:00:00"
    }
  ]
}
```

### 3. Оптимизированные параметры (`data/ai/training/optimized_params.json`)
```json
{
  "OG": {
    "sl_multiplier": 1.2,
    "tp_multiplier": 0.9,
    "last_updated": "2025-10-27",
    "accuracy": 0.65
  }
}
```

## 🔄 Полный цикл обучения

```
1. БОТ ОТКРЫВАЕТ ПОЗИЦИЮ
   └─ SmartRiskManager.backtest_coin() предлагает: SL=12%, TP=100%
   └─ Бот открывает с этими параметрами

2. ПОЗИЦИЯ ЗАКРЫВАЕТСЯ
   └─ Причина: STOP_LOSS (реальный SL был -15%)
   └─ Сохраняем: entry_data, market_data

3. ОБРАТНАЯ СВЯЗЬ
   └─ SmartRiskManager.evaluate_prediction()
   └─ Сравниваем: предсказанный SL vs реальный ROI
   └─ Score: 0.3 (плохо, т.к. -15% > -12%)

4. ОБУЧЕНИЕ
   └─ smart_risk_manager.learn_from_feedback()
   └─ Для OG: увеличить SL с 12% до 14.4%
   └─ Сохраняем в optimized_params.json

5. СЛЕДУЮЩИЙ ВХОД В OG
   └─ SmartRiskManager.backtest_coin()
   └─ Использует: SL=14.4% (из оптимизированных параметров)
   └─ Более точное предсказание!
```

## 📈 Метрики успешности

```python
def _calculate_score(self, predicted_sl, predicted_tp, actual_result):
    """
    Оценивает насколько хорошо ИИ предсказал
    
    Возвращает 0-1:
    - 1.0 = идеально (позиция закрылась в диапазоне SL-TP)
    - 0.5 = средне (близко к SL)
    - 0.0 = плохо (выйшло за пределы)
    """
    
    if actual_result < -predicted_sl:
        # Вышли за SL → плохо
        return 0.0
    
    if actual_result > predicted_tp:
        # Превысили TP → хорошо, но не идеально
        return 0.8
    
    # В пределах SL-TP
    normalized = (actual_result + predicted_sl) / (predicted_tp + predicted_sl)
    return normalized
```

## 🎯 Итог

**Сейчас работает:**
- ✅ Сбор данных о стопах
- ✅ Анализ паттернов
- ✅ Использование ИИ модулей (LSTM, Anomaly, Risk Manager)

**Нужно добавить:**
- ❌ Систему обратной связи
- ❌ Сохранение оценок (scores)
- ❌ Обучение на основе scores
- ❌ Автоматическую корректировку параметров

