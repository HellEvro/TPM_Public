# Архитектура Premium: Обучение на стопах и бэктестинг

## 🎯 Цель

Создать премиум-модули, которые:
1. **Обучаются на стопах** - анализируют почему сделки закрылись по стопу
2. **Бэктестируют каждую монету** перед входом в позицию
3. **Предлагают оптимальные SL/TP** на основе бэктеста и истории стопов
4. **Определяют лучшие точки входа** с учетом исторических данных

## ⚠️ КРИТИЧНО: Не ломаем существующий функционал!

**Без лицензии:**
- ✅ Всё работает как раньше
- ✅ Базовый риск-менеджмент (фиксированные 15% SL, 100% TP)
- ✅ Нет анализа стопов, нет бэктестов

**С лицензией:**
- ✅ + ИИ обучение на стопах
- ✅ + Бэктест каждой монеты перед входом
- ✅ + Динамические SL/TP на основе истории
- ✅ + Предсказание лучших точек входа

## 🏗️ Архитектура

### 1. Модуль: `SmartRiskManager` (`bot_engine/ai/smart_risk_manager.py`)

```python
class SmartRiskManager:
    """Умный риск-менеджмент с обучением на стопах (Premium)"""
    
    def __init__(self):
        # Проверяем лицензию
        from bot_engine.ai import check_premium_license
        if not check_premium_license():
            raise ImportError("Premium license required")
        
        self.backtest_engine = None
        self.stop_analyzer = None
    
    def analyze_stopped_trades(self, limit=100):
        """Анализирует последние стопы для обучения ИИ"""
        # Получаем стопы из истории
        stopped_trades = bot_history_manager.get_stopped_trades(limit)
        
        # Анализируем паттерны
        patterns = self._extract_patterns(stopped_trades)
        
        # Сохраняем для обучения
        self._save_for_training(patterns)
        
        return {
            'total_stops': len(stopped_trades),
            'common_reasons': self._analyze_reasons(stopped_trades),
            'optimal_sl': self._optimize_stop_loss(stopped_trades),
            'optimal_tp': self._optimize_take_profit(stopped_trades)
        }
    
    def backtest_coin(self, symbol, candles, direction):
        """Бэктест монеты перед входом в позицию"""
        # Используем существующий BacktestEngine
        from bot_engine.ai.backtester import BacktestEngine
        
        # Запускаем бэктест на последних N свечах
        result = self.backtest_engine.run_quick_backtest(
            symbol, candles[-50:], direction
        )
        
        return {
            'optimal_entry': result['best_entry_price'],
            'optimal_sl': result['best_stop_loss'],
            'optimal_tp': result['best_take_profit'],
            'win_rate': result['win_rate'],
            'expected_return': result['expected_return'],
            'confidence': result['confidence']
        }
```

### 2. Интеграция в `bot_class.py`

```python
# При открытии позиции
def _open_position_on_exchange(self, ...):
    # 🔴 СТАРЫЙ КОД (работает всегда)
    if max_loss_percent:
        stop_loss_price = calculate_stop_loss(...)
    
    # ✅ НОВЫЙ КОД (только с лицензией)
    try:
        if RiskConfig.STOP_ANALYSIS_ENABLED and check_premium_license():
            from bot_engine.ai.smart_risk_manager import SmartRiskManager
            smart_risk = SmartRiskManager()
            
            # Бэктест перед входом
            backtest_result = smart_risk.backtest_coin(
                self.symbol, candles, side
            )
            
            # Используем оптимальные значения из бэктеста
            if backtest_result.get('confidence', 0) > 0.7:
                stop_loss_price = backtest_result['optimal_sl']
                take_profit_price = backtest_result['optimal_tp']
                logger.info(f"[PREMIUM] Используем оптимизированные SL/TP из бэктеста")
    except ImportError:
        # Лицензии нет - работаем как раньше
        pass
```

### 3. Сохранение данных о стопах (расширенное)

```python
# В bot_class.py при закрытии позиции
def _close_position_on_exchange(self, reason, ...):
    # Подготавливаем данные для обучения ИИ (только если стоп)
    entry_data = None
    market_data = None
    
    if 'STOP' in reason.upper() and check_premium_license():
        entry_data = {
            'entry_price': self.entry_price,
            'rsi': self.entry_rsi,
            'volatility': self.entry_volatility,
            'trend': self.entry_trend,
            'duration_hours': duration,
            'max_profit_achieved': self.max_profit_achieved
        }
        
        market_data = {
            'volatility': current_volatility,
            'trend': current_trend,
            'price_movement': price_change_pct
        }
    
    # Сохраняем в историю
    bot_history_manager.log_position_closed(
        bot_id=self.symbol,
        symbol=self.symbol,
        direction=self.position_side,
        exit_price=exit_price,
        pnl=pnl,
        roi=roi,
        reason=reason,
        entry_data=entry_data,  # Добавляем премиум данные
        market_data=market_data  # Добавляем премиум данные
    )
```

### 4. UI для анализа стопов (с проверкой лицензии)

```javascript
// В bots_manager.js

async loadStoppedTrades() {
    // Проверяем лицензию
    const licenseStatus = await this.checkPremiumLicense();
    
    if (!licenseStatus.valid) {
        this.showNotification('⚠️ Премиум-функции требуют лицензии', 'warning');
        return;
    }
    
    // Загружаем стопы
    const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/stops`);
    const data = await response.json();
    
    if (data.success) {
        this.displayStoppedTradesAnalysis(data.trades);
    }
}

displayStoppedTradesAnalysis(trades) {
    // Отображаем:
    // - График частоты стопов по причинам
    // - Топ-3 причин стопов
    // - Рекомендации ИИ по улучшению
    // - Оптимальные SL/TP для следующих сделок
}
```

### 5. API endpoint с проверкой лицензии

```python
@bots_app.route('/api/bots/stops', methods=['GET'])
def get_stopped_trades():
    """Получает стопы (премиум функция)"""
    
    # Проверяем лицензию
    try:
        from bot_engine.ai import check_premium_license
        if not check_premium_license():
            return jsonify({
                'success': False,
                'error': 'Premium license required',
                'license_required': True
            }), 403
    except Exception as e:
        logger.warning(f"[API] License check failed: {e}")
        return jsonify({
            'success': False,
            'error': 'License check failed'
        }), 500
    
    # Если лицензия есть - возвращаем данные
    stopped_trades = bot_history_manager.get_stopped_trades(limit)
    
    return jsonify({
        'success': True,
        'trades': stopped_trades,
        'count': len(stopped_trades),
        'premium': True
    })
```

## 📋 План реализации

1. ✅ **Готово**: Расширен `BotHistoryManager` для сохранения детальных данных
2. ✅ **Готово**: Создан API endpoint `/api/bots/stops`
3. ✅ **Готово**: Создан план (`AI_STOP_ANALYSIS_PLAN.md`)
4. 🔨 **TODO**: Создать `SmartRiskManager` с проверкой лицензии
5. 🔨 **TODO**: Интегрировать в `bot_class.py` (с fallback на старое поведение)
6. 🔨 **TODO**: Добавить UI вкладку "Анализ стопов" (с проверкой лицензии)
7. 🔨 **TODO**: Интегрировать `BacktestEngine` для бэктеста перед входом

## 🛡️ Гарантии

**Без лицензии:**
- `check_premium_license()` возвращает `False`
- `SmartRiskManager` не импортируется
- Все премиум-вызовы игнорируются
- Система работает как раньше с фиксированными 15% SL

**С лицензией:**
- `check_premium_license()` возвращает `True`
- `SmartRiskManager` импортируется и работает
- Бэктесты запускаются перед входом
- UI показывает анализ стопов
- SL/TP оптимизируются на основе бэктестов

## ✅ Критерии готовности

- ✅ Без лицензии всё работает как раньше
- ✅ С лицензией работают премиум-функции
- ✅ Бэктесты запускаются автоматически перед входом
- ✅ Анализ стопов работает и даёт рекомендации
- ✅ UI показывает анализ с проверкой лицензии
- ✅ Нет ошибок при отсутствии лицензии

