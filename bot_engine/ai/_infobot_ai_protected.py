#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
"""

import importlib.machinery
import sys
import logging
from pathlib import Path

_logger = logging.getLogger('AI.Protected')

_COMPILED_NAME = "_ai_launcher.pyc"
_compiled_path = Path(__file__).resolve().with_name(_COMPILED_NAME)

if not _compiled_path.exists():
    raise RuntimeError(f"Не найден защищённый AI модуль: {_compiled_path}")

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
        if python_version == "3.12":
            _logger.error("[ERROR] Или пересоздайте .venv_gpu: python scripts/setup_python_gpu.py")
        else:
            _logger.error("[ERROR] Или пересоздайте .venv: python scripts/ensure_python314_venv.py")
        raise RuntimeError(
            f"_ai_launcher.pyc несовместим с Python {python_version}. "
            f"Модуль должен быть скомпилирован разработчиком под эту версию Python."
        )
    else:
        # Другая ошибка - пробрасываем дальше
        raise
