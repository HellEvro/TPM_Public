#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
Поддерживает версионированные .pyc файлы для Python 3.14 и 3.12.
"""

import importlib.machinery
import sys
import logging
from pathlib import Path

_logger = logging.getLogger('AI.Protected')

_COMPILED_NAME = "_ai_launcher.pyc"

def _get_versioned_pyc_path():
    """Определяет путь к правильной версии .pyc файла на основе текущей версии Python."""
    base_dir = Path(__file__).resolve().parent
    python_version = sys.version_info[:2]
    
    # Определяем версию Python и соответствующую директорию
    if python_version >= (3, 14):
        version_dir = base_dir / 'pyc_314'
        version_name = "3.14"
    elif python_version == (3, 12):
        version_dir = base_dir / 'pyc_312'
        version_name = "3.12"
    else:
        # Для других версий пробуем найти подходящую или используем основную директорию
        version_dir = base_dir
        version_name = f"{python_version[0]}.{python_version[1]}"
        _logger.warning(f"[WARNING] Python {version_name} не имеет специальной версии .pyc, используем основную директорию")
    
    # Путь к версионированному .pyc файлу
    versioned_path = version_dir / _COMPILED_NAME
    
    # Если версионированный файл не найден, пробуем основную директорию (для обратной совместимости)
    if not versioned_path.exists():
        fallback_path = base_dir / _COMPILED_NAME
        if fallback_path.exists():
            _logger.warning(f"[WARNING] Версионированный .pyc для Python {version_name} не найден, используем основной файл")
            return fallback_path
        else:
            raise RuntimeError(
                f"Не найден защищённый AI модуль для Python {version_name}: {versioned_path}\n"
                f"Также не найден файл в основной директории: {fallback_path}\n"
                f"Обратитесь к разработчику для получения правильной версии модулей."
            )
    
    return versioned_path

_compiled_path = _get_versioned_pyc_path()

# Пытаемся загрузить модуль
try:
    _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
    _loader.exec_module(sys.modules[__name__])
except (ImportError, ValueError) as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        # Модуль скомпилирован под другую версию Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _logger.error(f"[ERROR] _ai_launcher.pyc несовместим с текущей версией Python: {python_version}")
        _logger.error("[ERROR] Модуль был скомпилирован под другую версию Python.")
        _logger.error("[ERROR] Обратитесь к разработчику для получения правильной версии модулей.")
        raise RuntimeError(
            f"_ai_launcher.pyc несовместим с Python {python_version}. "
            f"Модуль должен быть скомпилирован разработчиком под эту версию Python."
        )
    else:
        # Другая ошибка - пробрасываем дальше
        raise
