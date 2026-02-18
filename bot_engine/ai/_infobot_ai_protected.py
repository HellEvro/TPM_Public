#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
"""

import importlib.machinery
import sys
from pathlib import Path

def _get_launcher_pyc_path():
    """Путь к _ai_launcher.pyc в bot_engine/ai/."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "_ai_launcher.pyc"

_compiled_path = _get_launcher_pyc_path()

if _compiled_path is None or not _compiled_path.exists():
    raise RuntimeError(
        "Не найден защищённый AI модуль (_ai_launcher.pyc).\n"
        "Выполните: python -m license_generator.compile_all"
    )

try:
    _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
    _loader.exec_module(sys.modules[__name__])
except Exception as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        raise RuntimeError(
            "_ai_launcher.pyc несовместим с текущей версией Python.\n"
            "Пересоберите: python -m license_generator.compile_all"
        )
    raise
