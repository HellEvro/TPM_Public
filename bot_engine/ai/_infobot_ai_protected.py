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

if not _compiled_path.exists():
    raise RuntimeError(f"Не найден защищённый AI модуль: {_compiled_path}")

_loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
_loader.exec_module(sys.modules[__name__])
