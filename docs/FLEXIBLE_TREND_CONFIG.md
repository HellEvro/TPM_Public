# 🎯 ГИБКАЯ НАСТРОЙКА ПАРАМЕТРОВ ТРЕНДА

**Дата:** 2025-10-17  
**Версия:** 1.0

---

## 📋 ОБЗОР

Реализована **гибкая система настройки** параметров определения тренда:
- ✅ Глобальные настройки по умолчанию
- ✅ Индивидуальные настройки для каждой монеты
- ✅ Опциональные критерии подтверждения
- ✅ Мягкие условия (N из M подтверждений)

---

## ⚙️ ГЛОБАЛЬНЫЕ НАСТРОЙКИ

### **bot_engine/bot_config.py**

```python
# Параметры подтверждения тренда (по умолчанию)
TREND_CONFIRMATION_BARS = 3  # Количество свечей для проверки
TREND_MIN_CONFIRMATIONS = 2  # Минимум подтверждений из 3 возможных

# Опциональность критериев
TREND_REQUIRE_SLOPE = False  # Наклон EMA_long (опциональный)
TREND_REQUIRE_PRICE = True   # Цена vs EMA_long (обязательный)
TREND_REQUIRE_CANDLES = True # N свечей подряд (обязательный)
```

### **SystemConfig**

```python
class SystemConfig:
    # Параметры подтверждения тренда
    TREND_CONFIRMATION_BARS = 3
    TREND_MIN_CONFIRMATIONS = 2
    
    # Опциональность проверок
    TREND_REQUIRE_SLOPE = False
    TREND_REQUIRE_PRICE = True
    TREND_REQUIRE_CANDLES = True
```

---

## 🎨 ИНДИВИДУАЛЬНЫЕ НАСТРОЙКИ ДЛЯ МОНЕТ

### **Структура data/optimal_ema.json**

Теперь для каждой монеты можно задать индивидуальные параметры:

```json
{
  "BTC": {
    "ema_short_period": 50,
    "ema_long_period": 200,
    "accuracy": 95.5,
    
    // Индивидуальные параметры подтверждения (опционально)
    "trend_confirmation_bars": 3,
    "trend_min_confirmations": 2,
    "trend_require_slope": false,
    "trend_require_price": true,
    "trend_require_candles": true
  },
  
  "ETH": {
    "ema_short_period": 60,
    "ema_long_period": 180,
    "accuracy": 97.2,
    
    // Для волатильной монеты - более мягкие условия
    "trend_confirmation_bars": 2,
    "trend_min_confirmations": 1,
    "trend_require_slope": false,
    "trend_require_price": false,
    "trend_require_candles": true
  }
}
```

**Примечание:** Если параметр не указан (`null`), используется глобальное значение из `SystemConfig`.

---

## 🔍 ЛОГИКА ОПРЕДЕЛЕНИЯ ТРЕНДА

### **Алгоритм (calculations.py, analyze_trend_6h)**

```python
def analyze_trend_6h(symbol, exchange_obj=None):
    # 1. Получаем параметры (индивидуальные ИЛИ глобальные)
    ema_periods = get_optimal_ema_periods(symbol)
    trend_confirmation_bars = ema_periods.get('trend_confirmation_bars') or SystemConfig.TREND_CONFIRMATION_BARS
    trend_min_confirmations = ema_periods.get('trend_min_confirmations') or SystemConfig.TREND_MIN_CONFIRMATIONS
    trend_require_slope = ema_periods.get('trend_require_slope') if ... else SystemConfig.TREND_REQUIRE_SLOPE
    trend_require_price = ema_periods.get('trend_require_price') if ... else SystemConfig.TREND_REQUIRE_PRICE
    trend_require_candles = ema_periods.get('trend_require_candles') if ... else SystemConfig.TREND_REQUIRE_CANDLES
    
    # 2. Рассчитываем EMA и другие показатели
    # ...
    
    # 3. Проверяем UP Trend
    if ema_short > ema_long:  # Основной сигнал: крест вверх
        confirmations = 0
        required_confirmations = 0
        
        # Цена выше EMA_long
        if current_close > ema_long:
            confirmations += 1
        if trend_require_price:  # Обязательный критерий?
            required_confirmations += 1
        
        # Наклон EMA_long вверх
        if ema_long_slope > 0:
            confirmations += 1
        if trend_require_slope:  # Обязательный критерий?
            required_confirmations += 1
        
        # N свечей подряд выше EMA_long
        if closes_above >= trend_min_confirmations:
            confirmations += 1
        if trend_require_candles:  # Обязательный критерий?
            required_confirmations += 1
        
        # Определяем тренд: выполнены обязательные + достаточно опциональных
        if confirmations >= max(trend_min_confirmations, required_confirmations):
            trend = 'UP'
    
    # 4. DOWN Trend - аналогично
    # ...
```

---

## 💡 ПРИМЕРЫ НАСТРОЕК

### **Пример 1: Консервативные условия (по умолчанию)**

```python
TREND_CONFIRMATION_BARS = 3      # Проверяем 3 свечи
TREND_MIN_CONFIRMATIONS = 2      # Нужно минимум 2 подтверждения
TREND_REQUIRE_SLOPE = False      # Наклон - опциональный (дает +1 балл)
TREND_REQUIRE_PRICE = True       # Цена - обязательный
TREND_REQUIRE_CANDLES = True     # Свечи - обязательный
```

**Результат:**
- UP: крест вверх + (цена > EMA_long ИЛИ наклон > 0) + (≥2 свечи из 3 > EMA_long)
- **Точность:** высокая
- **Чувствительность:** средняя
- **Применение:** стабильные монеты, консервативная торговля

### **Пример 2: Мягкие условия (волатильные монеты)**

```python
TREND_CONFIRMATION_BARS = 2      # Проверяем 2 свечи
TREND_MIN_CONFIRMATIONS = 1      # Нужна хотя бы 1 свеча
TREND_REQUIRE_SLOPE = False      # Наклон - опциональный
TREND_REQUIRE_PRICE = False      # Цена - опциональный
TREND_REQUIRE_CANDLES = True     # Свечи - обязательный
```

**Результат:**
- UP: крест вверх + (≥1 свеча из 2 > EMA_long) + любые 1-2 доп. подтверждения
- **Точность:** средняя
- **Чувствительность:** высокая
- **Применение:** волатильные монеты, агрессивная торговля

### **Пример 3: Строгие условия (осторожная торговля)**

```python
TREND_CONFIRMATION_BARS = 5      # Проверяем 5 свечей
TREND_MIN_CONFIRMATIONS = 4      # Нужно минимум 4 подтверждения
TREND_REQUIRE_SLOPE = True       # Наклон - обязательный
TREND_REQUIRE_PRICE = True       # Цена - обязательный
TREND_REQUIRE_CANDLES = True     # Свечи - обязательный
```

**Результат:**
- UP: крест вверх + цена > EMA + наклон > 0 + (≥4 свечи из 5 > EMA_long)
- **Точность:** очень высокая
- **Чувствительность:** низкая (много NEUTRAL)
- **Применение:** очень осторожная торговля

---

## 🎯 КАК НАСТРОИТЬ ДЛЯ КОНКРЕТНОЙ МОНЕТЫ

### **Вариант 1: Вручную отредактировать optimal_ema.json**

1. Откройте `data/optimal_ema.json`
2. Найдите нужную монету
3. Добавьте параметры:

```json
{
  "MYCOIN": {
    "ema_short_period": 50,
    "ema_long_period": 200,
    "accuracy": 98.5,
    "trend_confirmation_bars": 2,    // Добавляем
    "trend_min_confirmations": 1,    // мягкие
    "trend_require_slope": false,    // условия
    "trend_require_price": false,
    "trend_require_candles": true
  }
}
```

4. Сохраните файл
5. Нажмите "🔄 Hot Reload" в UI или перезапустите сервер

### **Вариант 2: Программно**

```python
from bot_engine.optimal_ema_manager import optimal_ema_data, save_optimal_ema_periods

# Обновляем настройки для монеты
optimal_ema_data['MYCOIN']['trend_confirmation_bars'] = 2
optimal_ema_data['MYCOIN']['trend_min_confirmations'] = 1
optimal_ema_data['MYCOIN']['trend_require_slope'] = False

# Сохраняем
save_optimal_ema_periods()
```

---

## 📊 ТЕСТИРОВАНИЕ НАСТРОЕК

### **Тест 1: Сравнение разных настроек**

```bash
python tests/test_trend_methods.py
```

Покажет как разные параметры влияют на определение тренда.

### **Тест 2: Проверка для конкретной монеты**

```python
from bots_modules.calculations import analyze_trend_6h

# Тестируем с текущими настройками
result = analyze_trend_6h('BTC')
print(f"Trend: {result['trend']}")
print(f"Confirmations: {result['min_confirmations']}")
```

---

## 🔧 РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ

### **По типу монеты:**

| Тип монеты | confirmation_bars | min_confirmations | require_slope | require_price | require_candles |
|------------|-------------------|-------------------|---------------|---------------|-----------------|
| **Стабильная** (BTC, ETH) | 3 | 2 | False | True | True |
| **Волатильная** (альткоины) | 2 | 1 | False | False | True |
| **Очень волатильная** (мемы) | 1 | 1 | False | False | False |
| **Консервативная** (крупные суммы) | 5 | 4 | True | True | True |

### **По стилю торговли:**

| Стиль | confirmation_bars | min_confirmations | Описание |
|-------|-------------------|-------------------|----------|
| **Агрессивный** | 1-2 | 1 | Максимум сигналов, больше рисков |
| **Сбалансированный** | 2-3 | 2 | Баланс точности и сигналов (рекомендуется) |
| **Консервативный** | 4-5 | 3-4 | Высокая точность, меньше сигналов |

---

## ✅ ПРЕИМУЩЕСТВА НОВОЙ СИСТЕМЫ

1. **🎨 Гибкость**
   - Настройки для каждой монеты отдельно
   - Можно адаптировать под волатильность

2. **⚙️ Простота**
   - Все параметры в одном месте (`optimal_ema.json`)
   - Не нужно менять код для настройки

3. **🔄 Обратная совместимость**
   - Если параметры не заданы → используются глобальные
   - Старые монеты продолжают работать

4. **📈 Оптимизация**
   - Можно A/B тестировать разные настройки
   - Легко найти оптимальный баланс

5. **🚀 Hot Reload**
   - Изменения применяются без перезапуска
   - Быстрое тестирование

---

## 🎯 ИТОГ

### **Что реализовано:**

✅ Глобальные настройки в `bot_config.py` и `SystemConfig`  
✅ Индивидуальные настройки в `optimal_ema.json`  
✅ Гибкая логика в `analyze_trend_6h()`  
✅ Опциональные и обязательные критерии  
✅ Мягкие условия (N из M подтверждений)  
✅ Обратная совместимость  
✅ Hot Reload поддержка  

### **Текущие настройки (по умолчанию):**

```python
TREND_CONFIRMATION_BARS = 3      # 3 свечи
TREND_MIN_CONFIRMATIONS = 2      # Минимум 2 подтверждения
TREND_REQUIRE_SLOPE = False      # Наклон опциональный
TREND_REQUIRE_PRICE = True       # Цена обязательна
TREND_REQUIRE_CANDLES = True     # Свечи обязательны
```

**Это сбалансированные настройки** - дают хорошую точность без излишней строгости!

---

**Готово к использованию!** 🚀

