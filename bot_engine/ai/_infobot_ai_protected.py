#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
Поддерживает версионированные .pyc файлы для Python 3.12+ (pyc_312 для 3.12, pyc_314 для 3.14+).
"""

import importlib.machinery
import sys
from pathlib import Path

def _get_versioned_pyc_path():
    """Определяет путь к версионированному _ai_launcher.pyc на основе текущей версии Python."""
    base_dir = Path(__file__).resolve().parent
    python_version = sys.version_info[:2]

    # Определяем версию Python и соответствующую директорию
    # Поддерживаем Python 3.12+ (pyc_312 для 3.12, pyc_314 для 3.14+)
    if python_version >= (3, 14):
        version_dir = base_dir / 'pyc_314'
    elif python_version == (3, 12):
        version_dir = base_dir / 'pyc_312'
    else:
        # Для других версий используем основную директорию (fallback)
        version_dir = base_dir

    # Путь к версионированному .pyc файлу
    versioned_path = version_dir / "_ai_launcher.pyc"

    # Если версионированный файл не найден, пробуем основную директорию
    if not versioned_path.exists():
        fallback_path = base_dir / "_ai_launcher.pyc"
        if fallback_path.exists():
            return fallback_path
        return None

    return versioned_path

_compiled_path = _get_versioned_pyc_path()

if _compiled_path is None:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise RuntimeError(
        f"Не найден защищённый AI модуль для Python {python_version}.\n"
        f"Версионированный файл отсутствует в директории pyc_{sys.version_info.major}{sys.version_info.minor}/.\n"
        f"Обратитесь к разработчику для получения правильной версии модулей."
    )

try:
    _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
    _loader.exec_module(sys.modules[__name__])
except Exception as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"_ai_launcher.pyc несовместим с Python {python_version}.\n"
            f"Модуль был скомпилирован под другую версию Python.\n"
            f"Обратитесь к разработчику для получения правильной версии модулей."
        )
    raise
