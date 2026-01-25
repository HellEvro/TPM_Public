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
# Поддерживаются версионированные .pyc файлы для Python 3.14 и 3.12

def _get_versioned_module_path(module_name):
    """Определяет путь к версионированному .pyc модулю на основе текущей версии Python."""
    base_dir = Path(__file__).resolve().parent
    python_version = sys.version_info[:2]
    
    # Определяем версию Python и соответствующую директорию
    if python_version >= (3, 14):
        version_dir = base_dir / 'pyc_314'
    elif python_version == (3, 12):
        version_dir = base_dir / 'pyc_312'
    else:
        # Для других версий используем основную директорию
        version_dir = base_dir
    
    # Путь к версионированному .pyc файлу
    versioned_path = version_dir / f"{module_name}.pyc"
    
    # Подробное логирование для диагностики (используем info, чтобы было видно всегда)
    _logger.info(f"[AI] [INFO] Поиск модуля {module_name} для Python {python_version[0]}.{python_version[1]}")
    _logger.info(f"[AI] [INFO] Базовая директория: {base_dir}")
    _logger.info(f"[AI] [INFO] Версионированная директория: {version_dir}")
    _logger.info(f"[AI] [INFO] Ожидаемый путь: {versioned_path}")
    _logger.info(f"[AI] [INFO] Файл существует: {versioned_path.exists()}")
    
    # Если версионированный файл не найден, пробуем основную директорию (для обратной совместимости)
    if not versioned_path.exists():
        fallback_path = base_dir / f"{module_name}.pyc"
        _logger.warning(f"[AI] [WARNING] Версионированный файл не найден, пробуем fallback: {fallback_path}")
        _logger.warning(f"[AI] [WARNING] Fallback файл существует: {fallback_path.exists()}")
        if fallback_path.exists():
            return fallback_path
        # Проверяем, существует ли вообще директория с версионированными файлами
        if version_dir.exists():
            _logger.error(f"[AI] [ERROR] Директория {version_dir} существует, но файл {module_name}.pyc не найден")
            # Показываем список файлов в директории
            try:
                files_in_dir = list(version_dir.glob("*.pyc"))
                _logger.error(f"[AI] [ERROR] Файлы в {version_dir}: {[f.name for f in files_in_dir]}")
            except Exception as e:
                _logger.error(f"[AI] [ERROR] Ошибка при чтении директории: {e}")
        else:
            _logger.error(f"[AI] [ERROR] Директория {version_dir} не существует")
        return None
    
    return versioned_path

def _load_versioned_module(module_name, module_import_name):
    """Загружает версионированный .pyc модуль используя importlib."""
    import importlib.util
    import importlib.machinery
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _logger.info(f"[AI] [INFO] Попытка загрузки модуля {module_name} для Python {python_version}")
    
    pyc_path = _get_versioned_module_path(module_name)
    if pyc_path is None:
        _logger.warning(f"[AI] [WARNING] Версионированный .pyc файл для {module_name} не найден (Python {python_version})")
        _logger.warning(f"[AI] [WARNING] Ожидаемый путь: bot_engine/ai/pyc_{sys.version_info.major}{sys.version_info.minor}/{module_name}.pyc")
        # Пробуем обычный импорт (для обратной совместимости)
        return None
    
    if not pyc_path.exists():
        _logger.warning(f"[AI] [WARNING] Файл {pyc_path} не существует для Python {python_version}")
        return None
    
    _logger.info(f"[AI] [INFO] Найден файл {pyc_path}, начинаю загрузку...")
    
    try:
        loader = importlib.machinery.SourcelessFileLoader(module_import_name, str(pyc_path))
        spec = importlib.util.spec_from_loader(module_import_name, loader)
        if spec is None:
            _logger.error(f"[AI] [ERROR] Не удалось создать spec для {module_name} из {pyc_path}")
            return None
        _logger.info(f"[AI] [INFO] Spec создан, создаю модуль...")
        module = importlib.util.module_from_spec(spec)
        _logger.info(f"[AI] [INFO] Модуль создан, выполняю exec_module...")
        loader.exec_module(module)
        _logger.info(f"[AI] [INFO] Модуль {module_name} успешно загружен из {pyc_path}")
        # Проверяем, что модуль имеет содержимое
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        _logger.info(f"[AI] [INFO] Модуль содержит {len(attrs)} публичных атрибутов: {attrs[:10]}...")
        return module
    except Exception as e:
        err_msg = str(e).lower()
        if "bad magic number" in err_msg or "bad magic" in err_msg:
            _logger.error(f"[AI] [ERROR] {module_name}.pyc несовместим с текущей версией Python: {python_version}")
            _logger.error(f"[AI] [ERROR] Путь к файлу: {pyc_path}")
            _logger.error("[AI] [ERROR] Модуль был скомпилирован под другую версию Python.")
            _logger.error("[AI] [ERROR] Обратитесь к разработчику для получения правильной версии модулей.")
            # Не выбрасываем исключение, возвращаем None чтобы система могла продолжить работу
            return None
        # Для других ошибок тоже возвращаем None вместо raise
        _logger.error(f"[AI] [ERROR] Ошибка загрузки модуля {module_name} из {pyc_path}: {e}", exc_info=True)
        return None

# Пытаемся загрузить ai_manager из версионированной директории
try:
    ai_manager_module = _load_versioned_module('ai_manager', 'bot_engine.ai.ai_manager')
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
        # Пытаемся загрузить license_checker из версионированной директории
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _license_logger.info(f"[AI] [INFO] Попытка загрузки license_checker для Python {python_version}")
        
        license_checker_module = _load_versioned_module('license_checker', 'bot_engine.ai.license_checker')
        if license_checker_module is not None:
            _license_logger.info(f"[AI] [INFO] Модуль license_checker успешно загружен из версионированной директории")
            # Проверяем, что модуль имеет нужный атрибут
            if hasattr(license_checker_module, 'get_license_checker'):
                get_license_checker = license_checker_module.get_license_checker
                _license_logger.info(f"[AI] [INFO] Функция get_license_checker найдена в модуле")
            else:
                _license_logger.error(f"[AI] [ERROR] Модуль license_checker загружен, но не содержит get_license_checker")
                _license_logger.error(f"[AI] [ERROR] Доступные атрибуты: {dir(license_checker_module)}")
                _LICENSE_STATUS = False
                _LICENSE_INFO = None
                return False
        else:
            _license_logger.warning(f"[AI] [WARNING] Не удалось загрузить license_checker из версионированной директории, пробуем fallback")
            # Fallback к обычному импорту
            try:
                from .license_checker import get_license_checker
                _license_logger.info(f"[AI] [INFO] Модуль license_checker загружен через fallback импорт")
            except ImportError as import_err:
                _license_logger.error(f"[AI] [ERROR] Не удалось загрузить license_checker даже через fallback: {import_err}")
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