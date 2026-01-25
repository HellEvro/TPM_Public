#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Loader stub для защищённого AI лаунчера.
Подгружает bot_engine/ai/_ai_launcher.pyc и регистрирует его как модуль.
"""

import importlib.machinery
import sys
import subprocess
import logging
from pathlib import Path

_logger = logging.getLogger('AI.Protected')

_COMPILED_NAME = "_ai_launcher.pyc"
_compiled_path = Path(__file__).resolve().with_name(_COMPILED_NAME)

def _auto_recompile():
    """Автоматически перекомпилирует модули при несовместимости"""
    try:
        project_root = _compiled_path.parents[2]  # bot_engine/ai -> bot_engine -> project
        compile_script = project_root / 'license_generator' / 'compile_all.py'
        
        if not compile_script.exists():
            _logger.warning(f"Скрипт компиляции не найден: {compile_script}")
            return False
        
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _logger.info(f"[INFO] Автоматическая перекомпиляция модулей под Python {python_version}...")
        
        result = subprocess.run(
            [sys.executable, str(compile_script)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            _logger.info("[OK] Модули успешно перекомпилированы")
            return True
        else:
            _logger.warning(f"[WARNING] Ошибка перекомпиляции: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return False
    except Exception as e:
        _logger.warning(f"[WARNING] Не удалось автоматически перекомпилировать: {e}")
        return False

if not _compiled_path.exists():
    raise RuntimeError(f"Не найден защищённый AI модуль: {_compiled_path}")

# Пытаемся загрузить модуль
try:
    _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
    _loader.exec_module(sys.modules[__name__])
except (ImportError, ValueError) as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        # Автоматически перекомпилируем модули
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _logger.error(f"[ERROR] _ai_launcher.pyc несовместим с текущей версией Python: {python_version}")
        _logger.info("[INFO] Попытка автоматической перекомпиляции...")
        
        if _auto_recompile():
            # Пробуем загрузить снова
            try:
                _loader = importlib.machinery.SourcelessFileLoader(__name__, str(_compiled_path))
                _loader.exec_module(sys.modules[__name__])
                _logger.info("[OK] Модуль успешно загружен после перекомпиляции")
            except Exception as e2:
                _logger.error(f"[ERROR] Не удалось загрузить модуль после перекомпиляции: {e2}")
                if python_version == "3.12":
                    _logger.error("[ERROR] Выполните вручную: python scripts/setup_python_gpu.py")
                    _logger.error("[ERROR] Или: .venv_gpu\\Scripts\\python license_generator\\compile_all.py")
                else:
                    _logger.error("[ERROR] Выполните вручную: python scripts/ensure_python314_venv.py")
                    _logger.error("[ERROR] Или: python license_generator/compile_all.py")
                raise
        else:
            # Перекомпиляция не удалась, показываем инструкции
            if python_version == "3.12":
                _logger.error("[ERROR] Выполните вручную: python scripts/setup_python_gpu.py")
                _logger.error("[ERROR] Или: .venv_gpu\\Scripts\\python license_generator\\compile_all.py")
            else:
                _logger.error("[ERROR] Выполните вручную: python scripts/ensure_python314_venv.py")
                _logger.error("[ERROR] Или: python license_generator/compile_all.py")
            raise
    else:
        # Другая ошибка - пробрасываем дальше
        raise
