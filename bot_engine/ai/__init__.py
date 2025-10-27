"""
ИИ модули для торгового бота InfoBot

Этот пакет содержит различные ИИ модули для улучшения торговых решений:

Модули:
--------
- **anomaly_detector.py** - Обнаружение аномалий (pump/dump) с помощью Isolation Forest
  Используется для улучшения ExitScam фильтра.
  
- **lstm_predictor.py** - Предсказание направления движения цены с помощью LSTM
  Предсказывает UP/DOWN/NEUTRAL на основе исторических данных.
  
- **pattern_detector.py** - Распознавание графических паттернов с помощью CNN
  Находит паттерны: флаги, треугольники, голова-плечи, и т.д.
  
- **risk_manager.py** - Динамический риск-менеджмент
  Оптимизирует SL/TP на основе волатильности и предсказывает развороты.
  
- **ai_manager.py** - Главный менеджер всех ИИ модулей
  Координирует работу всех модулей и объединяет их рекомендации.

Использование:
--------------
```python
from bot_engine.ai import get_ai_manager

# Получаем глобальный экземпляр AI Manager
ai_manager = get_ai_manager()

# Анализируем монету
ai_analysis = ai_manager.analyze_coin(symbol, coin_data, candles)

# Получаем финальную рекомендацию
recommendation = ai_manager.get_final_recommendation(
    symbol, system_signal, ai_analysis
)
```

Настройка:
----------
Все настройки находятся в `bot_engine/bot_config.py`:
- AI_ENABLED - мастер-переключатель для всех ИИ модулей
- AI_ANOMALY_DETECTION_ENABLED - включить обнаружение аномалий
- AI_LSTM_ENABLED - включить LSTM предсказания
- AI_PATTERN_ENABLED - включить распознавание паттернов
- AI_RISK_MANAGEMENT_ENABLED - включить динамический риск-менеджмент

См. также:
----------
- docs/AI_IMPLEMENTATION_ROADMAP.md - полный план внедрения
- docs/AI_IMPLEMENTATION_CHECKLIST.md - детальный чеклист задач
- docs/AI_INTEGRATION_IDEAS.md - идеи и концепции
- docs/LSTM_VS_RL_EXPLAINED.md - различия между подходами
"""

__version__ = '0.1.0'
__author__ = 'InfoBot Team'

# Экспорты будут добавлены по мере создания модулей
try:
    from .ai_manager import AIManager, get_ai_manager
    __all__ = ['AIManager', 'get_ai_manager']
except ImportError:
    # Модули еще не созданы - это нормально на этапе разработки
    __all__ = []


def check_premium_license() -> bool:
    """
    Проверяет наличие валидной премиум лицензии
    
    Returns:
        True если лицензия валидна и премиум функции доступны
    """
    try:
        from .ai_manager import get_ai_manager
        ai_manager = get_ai_manager()
        return ai_manager.is_available() and ai_manager._license_valid
    except Exception as e:
        return False