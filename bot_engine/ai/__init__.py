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

import logging
from pathlib import Path

# Экспорты будут добавлены по мере создания модулей
try:
    from .ai_manager import AIManager, get_ai_manager
    __all__ = ['AIManager', 'get_ai_manager']
except ImportError:
    # Модули еще не созданы - это нормально на этапе разработки
    __all__ = []

# Экспорт главного модуля AI системы (новый модуль)
# ai.py находится в корне проекта
try:
    import sys
    import os
    # Добавляем корень проекта в путь для импорта
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ai import get_ai_system
    __all__.append('get_ai_system')
except ImportError:
    pass

_license_logger = logging.getLogger('AI.License')
_LICENSE_STATUS = None
_LICENSE_INFO = None
_PROJECT_ROOT = None


def _detect_project_root() -> Path:
    """
    Определяет корень проекта (директория, где лежит ai.py и .lic файлы).
    Работает корректно как для .py, так и для .pyc файлов внутри __pycache__.
    """
    current = Path(__file__).resolve()
    search_paths = [current.parent] + list(current.parents)

    def is_root(candidate: Path) -> bool:
        return (candidate / 'ai.py').exists() and (candidate / 'bot_engine').exists()

    for candidate in search_paths:
        if candidate and is_root(candidate):
            return candidate

    cwd = Path.cwd()
    if is_root(cwd):
        return cwd

    # Фолбек: поднимаемся на три уровня (ai -> bot_engine -> project)
    try:
        return current.parents[3]
    except IndexError:
        return current.parent


def _get_project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = _detect_project_root()
    return _PROJECT_ROOT


def check_premium_license(force_refresh: bool = False) -> bool:
    """
    Проверяет наличие валидной премиум лицензии
    
    Returns:
        True если лицензия валидна и премиум функции доступны
    """
    global _LICENSE_STATUS, _LICENSE_INFO

    if _LICENSE_STATUS is not None and not force_refresh:
        return _LICENSE_STATUS

    try:
        from .license_checker import get_license_checker
    except Exception as exc:
        _license_logger.debug("License checker module unavailable: %s", exc)
        _LICENSE_STATUS = False
        _LICENSE_INFO = None
        return False

    try:
        project_root = _get_project_root()
        license_checker = get_license_checker(project_root=project_root)
        _LICENSE_STATUS = license_checker.is_valid()
        _LICENSE_INFO = license_checker.get_info() if _LICENSE_STATUS else None
        return _LICENSE_STATUS
    except Exception as exc:
        _license_logger.debug("License validation failed: %s", exc, exc_info=True)
        _LICENSE_STATUS = False
        _LICENSE_INFO = None
        return False