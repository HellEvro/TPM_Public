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
import sys
import os
from pathlib import Path

_logger = logging.getLogger('AI')

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


# Экспорты будут добавлены по мере создания модулей
# ВАЖНО: Используем .pyc файлы для защиты логики лицензирования
# .pyc файлы должны быть скомпилированы при сборке проекта через license_generator/compile_all.py
# Поддерживаются версионированные .pyc файлы для Python 3.12+ (pyc_312 для 3.12, pyc_314 для 3.14+)

def _get_module_pyc_path(module_name):
    """Определяет путь к .pyc модулю (без версионирования)."""
    base_dir = Path(__file__).resolve().parent
    pyc_path = base_dir / f"{module_name}.pyc"
    
    if pyc_path.exists():
        return pyc_path
    return None

def _load_pyc_module(module_name, module_import_name):
    """Загружает .pyc модуль используя importlib."""
    import importlib.util
    import importlib.machinery
    
    pyc_path = _get_module_pyc_path(module_name)
    if pyc_path is None:
        _logger.debug(f"[AI] .pyc файл для {module_name} не найден")
        return None
    
    try:
        loader = importlib.machinery.SourcelessFileLoader(module_import_name, str(pyc_path))
        spec = importlib.util.spec_from_loader(module_import_name, loader)
        if spec is None:
            _logger.error(f"[AI] Не удалось создать spec для {module_name}")
            return None
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        _logger.debug(f"[AI] Модуль {module_name} загружен из {pyc_path}")
        return module
    except Exception as e:
        err_msg = str(e).lower()
        if "bad magic number" in err_msg or "bad magic" in err_msg:
            _logger.error(f"[AI] {module_name}.pyc несовместим с текущей версией Python")
            return None
        _logger.error(f"[AI] Ошибка загрузки {module_name}: {e}")
        return None

# Пытаемся загрузить ai_manager из версионированной директории
try:
    ai_manager_module = _load_pyc_module('ai_manager', 'bot_engine.ai.ai_manager')
    if ai_manager_module is not None:
        AIManager = ai_manager_module.AIManager
        get_ai_manager = ai_manager_module.get_ai_manager
        __all__ = ['AIManager', 'get_ai_manager']
    else:
        # Fallback к обычному импорту (для обратной совместимости)
        from .ai_manager import AIManager, get_ai_manager
        __all__ = ['AIManager', 'get_ai_manager']
except ImportError as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        # Если .pyc несовместим - сообщаем пользователю
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _logger.error(f"[AI] [ERROR] ai_manager.pyc несовместим с текущей версией Python: {python_version}")
        _logger.error("[AI] [ERROR] Модуль был скомпилирован под другую версию Python.")
        _logger.error("[AI] [ERROR] Обратитесь к разработчику для получения правильной версии модулей.")
        __all__ = []
    else:
        # Модули еще не созданы - это нормально на этапе разработки
        __all__ = []
except Exception as e:
    # Другие ошибки - пробуем обычный импорт
    try:
        from .ai_manager import AIManager, get_ai_manager
        __all__ = ['AIManager', 'get_ai_manager']
    except:
        __all__ = []

# Экспорт главного модуля AI системы (новый модуль)
# ai.py находится в корне проекта
# ВАЖНО: Используем ленивый импорт, чтобы избежать циклического импорта
# ai.py импортирует bot_engine.ai, поэтому нельзя импортировать ai.py здесь напрямую
# Импорт выполняется только при вызове функции, а не при импорте модуля
def get_ai_system(*args, **kwargs):
    """
    Ленивый импорт get_ai_system из ai.py
    Вызывается только при необходимости, чтобы избежать циклического импорта
    
    ВАЖНО: Эта функция НЕ должна вызываться во время импорта модуля,
    только после того, как ai.py полностью загружен
    """
    # Проверяем, не находимся ли мы в процессе импорта ai.py
    import sys
    if 'ai' in sys.modules and hasattr(sys.modules['ai'], '__file__'):
        # ai.py уже загружен, можно импортировать
        try:
            from ai import get_ai_system as _get_ai_system
            return _get_ai_system(*args, **kwargs)
        except (ImportError, AttributeError) as e:
            # Если импорт не удался, пробуем через sys.path
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            try:
                from ai import get_ai_system as _get_ai_system
                return _get_ai_system(*args, **kwargs)
            except ImportError:
                raise ImportError(f"Не удалось импортировать get_ai_system из ai.py: {e}")
    else:
        # ai.py еще не загружен, пробуем импортировать
        try:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from ai import get_ai_system as _get_ai_system
            return _get_ai_system(*args, **kwargs)
        except ImportError as e:
            raise ImportError(f"Не удалось импортировать get_ai_system из ai.py (возможно, циклический импорт): {e}")

# НЕ добавляем get_ai_system в __all__ при импорте модуля
# Это предотвращает циклический импорт при загрузке bot_engine.ai
# Функция доступна через bot_engine.ai.get_ai_system(), но не экспортируется автоматически

_license_logger = logging.getLogger('AI.License')
# Убеждаемся, что логгер наследует уровень от корневого логгера
_license_logger.setLevel(logging.DEBUG)
# Если у логгера нет обработчиков, добавляем обработчик от корневого
if not _license_logger.handlers:
    _license_logger.propagate = True
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
        # Загружаем license_checker из .pyc
        license_checker_module = _load_pyc_module('license_checker', 'bot_engine.ai.license_checker')
        get_license_checker = None
        
        if license_checker_module is not None:
            if hasattr(license_checker_module, 'get_license_checker'):
                get_license_checker = license_checker_module.get_license_checker
        
        # Fallback к обычному импорту
        if get_license_checker is None:
            try:
                from .license_checker import get_license_checker
            except ImportError:
                _LICENSE_STATUS = False
                _LICENSE_INFO = None
                return False
    except Exception as exc:
        _license_logger.error(f"[AI] [ERROR] Ошибка при загрузке license_checker: {exc}", exc_info=True)
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
        _license_logger.debug(f"License validation failed: {exc}", exc_info=True)
        _LICENSE_STATUS = False
        _LICENSE_INFO = None
        return False


# ===================================================================
# ЗАГРУЗКА ЗАЩИЩЁННЫХ AI МОДУЛЕЙ ИЗ .pyc ФАЙЛОВ
# ===================================================================
# Все AI модули загружаются из скомпилированных .pyc файлов
# Исходники (.py) остаются только в приватном репозитории
# ===================================================================

def _load_ai_module(module_name: str):
    """
    Загружает AI модуль из .pyc файла.
    Возвращает модуль или None если загрузка не удалась.
    """
    import_name = f'bot_engine.ai.{module_name}'
    module = _load_pyc_module(module_name, import_name)
    if module is None:
        # Fallback: пробуем импортировать .py файл (для разработки)
        try:
            import importlib
            module = importlib.import_module(f'.{module_name}', 'bot_engine.ai')
        except ImportError:
            pass
    return module


# Ленивая загрузка модулей
_smart_money_features_module = None
_lstm_predictor_module = None
_transformer_predictor_module = None
_bayesian_optimizer_module = None
_drift_detector_module = None
_ensemble_module = None
_monitoring_module = None
_rl_agent_module = None
_sentiment_module = None
_pattern_detector_module = None
_ai_integration_module = None


def get_smart_money_features():
    """Получить класс SmartMoneyFeatures"""
    global _smart_money_features_module
    if _smart_money_features_module is None:
        _smart_money_features_module = _load_ai_module('smart_money_features')
    if _smart_money_features_module:
        return getattr(_smart_money_features_module, 'SmartMoneyFeatures', None)
    return None


def get_lstm_predictor():
    """Получить класс LSTMPredictor"""
    global _lstm_predictor_module
    if _lstm_predictor_module is None:
        _lstm_predictor_module = _load_ai_module('lstm_predictor')
    if _lstm_predictor_module:
        return getattr(_lstm_predictor_module, 'LSTMPredictor', None)
    return None


def get_transformer_predictor():
    """Получить класс TransformerPredictor"""
    global _transformer_predictor_module
    if _transformer_predictor_module is None:
        _transformer_predictor_module = _load_ai_module('transformer_predictor')
    if _transformer_predictor_module:
        return getattr(_transformer_predictor_module, 'TransformerPredictor', None)
    return None


def get_bayesian_optimizer():
    """Получить класс BayesianOptimizer"""
    global _bayesian_optimizer_module
    if _bayesian_optimizer_module is None:
        _bayesian_optimizer_module = _load_ai_module('bayesian_optimizer')
    if _bayesian_optimizer_module:
        return getattr(_bayesian_optimizer_module, 'BayesianOptimizer', None)
    return None


def get_drift_detector():
    """Получить класс DataDriftDetector"""
    global _drift_detector_module
    if _drift_detector_module is None:
        _drift_detector_module = _load_ai_module('drift_detector')
    if _drift_detector_module:
        return getattr(_drift_detector_module, 'DataDriftDetector', None)
    return None


def get_ensemble():
    """Получить класс VotingEnsemble"""
    global _ensemble_module
    if _ensemble_module is None:
        _ensemble_module = _load_ai_module('ensemble')
    if _ensemble_module:
        return getattr(_ensemble_module, 'VotingEnsemble', None)
    return None


def get_ai_monitor():
    """Получить класс AIPerformanceMonitor"""
    global _monitoring_module
    if _monitoring_module is None:
        _monitoring_module = _load_ai_module('monitoring')
    if _monitoring_module:
        return getattr(_monitoring_module, 'AIPerformanceMonitor', None)
    return None


def get_rl_trader():
    """Получить класс RLTrader"""
    global _rl_agent_module
    if _rl_agent_module is None:
        _rl_agent_module = _load_ai_module('rl_agent')
    if _rl_agent_module:
        return getattr(_rl_agent_module, 'RLTrader', None)
    return None


def get_sentiment_analyzer():
    """Получить класс SentimentAnalyzer"""
    global _sentiment_module
    if _sentiment_module is None:
        _sentiment_module = _load_ai_module('sentiment')
    if _sentiment_module:
        return getattr(_sentiment_module, 'SentimentAnalyzer', None)
    return None


def get_cnn_pattern_detector():
    """Получить класс CNNPatternDetector"""
    global _pattern_detector_module
    if _pattern_detector_module is None:
        _pattern_detector_module = _load_ai_module('pattern_detector')
    if _pattern_detector_module:
        return getattr(_pattern_detector_module, 'CNNPatternDetector', None)
    return None


def get_ai_integration():
    """Получить модуль ai_integration"""
    global _ai_integration_module
    if _ai_integration_module is None:
        _ai_integration_module = _load_ai_module('ai_integration')
    return _ai_integration_module


# Добавляем функции в __all__ для экспорта
__all__.extend([
    'check_premium_license',
    'get_smart_money_features',
    'get_lstm_predictor', 
    'get_transformer_predictor',
    'get_bayesian_optimizer',
    'get_drift_detector',
    'get_ensemble',
    'get_ai_monitor',
    'get_rl_trader',
    'get_sentiment_analyzer',
    'get_cnn_pattern_detector',
    'get_ai_integration',
])