#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
"""

import importlib.machinery
import sys
from pathlib import Path

_COMPILED_NAME = "_ai_launcher.pyc"
_compiled_path = Path(__file__).with_name(_COMPILED_NAME)

# Пробуем загрузить .pyc файл
if not _compiled_path.exists():
    raise RuntimeError(f"Не найден защищённый AI модуль: {_compiled_path}. Выполните: python license_generator/build_ai_launcher.py")

try:
    _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
    _loader.exec_module(sys.modules[__name__])
except (ImportError, ValueError, OSError) as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg or "invalid" in err_msg:
        # Если .pyc несовместим - сообщаем пользователю
        python_version = sys.version.split()[0]
        raise RuntimeError(
            f"_ai_launcher.pyc несовместим с текущей версией Python: {python_version}. "
            f"Выполните: python license_generator/build_ai_launcher.py"
        )
    else:
        raise
