#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Оболочка для защищённого AI лаунчера.
Вся рабочая логика находится в bot_engine/ai/_ai_launcher.pyc
"""

# ⚠️ КРИТИЧНО: Устанавливаем переменную окружения для идентификации процесса ai.py
# Это гарантирует, что функции из filters.py будут сохранять свечи в ai_data.db, а не в bots_data.db
import os
import sys
from pathlib import Path

# Проверяем наличие виртуального окружения с Python 3.12 для GPU
# Если найдено .venv_gpu, используем его вместо системного Python
venv_gpu_path = Path(__file__).parent / '.venv_gpu'
if venv_gpu_path.exists():
    if sys.platform == 'win32':
        venv_python = venv_gpu_path / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_gpu_path / 'bin' / 'python'
    
    if venv_python.exists():
        # Перезапускаем скрипт с Python из venv_gpu
        import subprocess
        try:
            subprocess.run([str(venv_python)] + sys.argv, check=True)
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except Exception:
            # Если не удалось перезапустить, продолжаем с текущим Python
            pass

os.environ['INFOBOT_AI_PROCESS'] = 'true'

# InfoBot требует Python 3.12. Если текущий — не 3.12, пробуем перезапуск с py -3.12 / python3.12
def _find_python312():
    import subprocess
    _check = 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)'
    candidates = (
        [(['py', '-3.12', '-c', _check], ['py', '-3.12'])] if sys.platform == 'win32' else []
    ) + [
        (['python3.12', '-c', _check], ['python3.12']),
        (['python312', '-c', _check], ['python312']),
    ]
    for check, run_cmd in candidates:
        try:
            r = subprocess.run(check, capture_output=True, timeout=5)
            if r.returncode == 0:
                return run_cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

_major, _minor = sys.version_info.major, sys.version_info.minor
if (_major, _minor) != (3, 12):
    py312 = _find_python312()
    if py312:
        import subprocess
        try:
            subprocess.run(py312 + sys.argv, check=True)
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except Exception:
            pass
    # Python 3.12 не найден — выходим с сообщением
    import logging
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger('AI')
    _log.error("=" * 80)
    _log.error("InfoBot требует Python 3.12. Текущий: %s.%s", _major, _minor)
    _log.error("Установите Python 3.12 или выполните: python scripts/setup_python_gpu.py")
    _log.error("  https://www.python.org/downloads/release/python-3120/")
    _log.error("=" * 80)
    sys.exit(1)

# Настройка логирования ПЕРЕД импортом защищенного модуля
import logging
try:
    from bot_engine.ai.ai_launcher_config import AILauncherConfig
    from utils.color_logger import setup_color_logging
    console_levels = getattr(AILauncherConfig, 'CONSOLE_LOG_LEVELS', [])
    setup_color_logging(console_log_levels=console_levels if console_levels else None)
except Exception as e:
    # Если не удалось загрузить конфиг, используем стандартную настройку
    try:
        from utils.color_logger import setup_color_logging
        setup_color_logging()
    except Exception as setup_error:
        import sys
        sys.stderr.write(f"❌ Ошибка настройки логирования: {setup_error}\n")

# Автоматическая проверка и установка TensorFlow с поддержкой GPU
# Выполняется ПЕРЕД импортом защищенного модуля
# ВАЖНО: Выполняется только в главном процессе, чтобы избежать дублирования
try:
    import multiprocessing
    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    
    if is_main_process:
        from bot_engine.ai.tensorflow_setup import ensure_tensorflow_setup
        logger = logging.getLogger('AI')
        logger.info("=" * 80)
        logger.info("ПРОВЕРКА И НАСТРОЙКА TENSORFLOW")
        logger.info("=" * 80)
        ensure_tensorflow_setup()
        logger.info("=" * 80)
    else:
        # В дочерних процессах только краткая проверка без установки
        logger = logging.getLogger('AI')
        logger.debug("Дочерний процесс - пропускаем полную проверку TensorFlow")
except Exception as tf_setup_error:
    # Если проверка не удалась, продолжаем работу
    logger = logging.getLogger('AI')
    logger.warning(f"Не удалось проверить TensorFlow: {tf_setup_error}")
    logger.info("Продолжаем работу...")

from typing import TYPE_CHECKING, Any

try:
    from bot_engine.ai import _infobot_ai_protected as _protected_module
except ImportError as e:
    err_msg = str(e).lower()
    if "bad magic number" in err_msg or "bad magic" in err_msg:
        _log = logging.getLogger("AI")
        current_version = sys.version.split()[0] if sys.version else "?"
        _major, _minor = sys.version_info.major, sys.version_info.minor
        _log.error("=" * 80)
        if (_major, _minor) == (3, 12):
            # Python 3.12, но .pyc несовместим - нужно перекомпилировать
            _log.error(f"[ERROR] AI модуль несовместим с текущей версией Python: {current_version}")
            _log.error("[ERROR] Выполните: python license_generator/compile_all.py")
            _log.error("[ERROR] Или: python license_generator/build_ai_launcher.py")
        else:
            # Не Python 3.12
            _log.error(f"[ERROR] AI модуль собран под Python 3.12. Текущий: {current_version}")
            _log.error("Выполните: python scripts/setup_python_gpu.py  либо используйте Python 3.12.")
        _log.error("=" * 80)
        sys.exit(1)
    raise


if TYPE_CHECKING:
    def main(*args: Any, **kwargs: Any) -> Any: ...


# Патч для перенаправления data_service.json в БД
def _patch_ai_system_update_data_status():
    """
    Патчит метод _update_data_status в классе AISystem для сохранения в БД вместо файла
    """
    try:
        # Импортируем helper для работы с БД
        from bot_engine.ai.data_service_status_helper import update_data_service_status_in_db

        # Получаем класс AISystem из защищенного модуля
        if hasattr(_protected_module, 'AISystem'):
            AISystem = _protected_module.AISystem

            # Сохраняем оригинальный метод (на случай если понадобится)
            original_update_data_status = AISystem._update_data_status

            # Заменяем метод на версию, которая сохраняет в БД
            def patched_update_data_status(self, **kwargs):
                """Патченная версия _update_data_status - сохраняет в БД вместо файла"""
                try:
                    update_data_service_status_in_db(**kwargs)
                except Exception as e:
                    # В случае ошибки пробуем оригинальный метод (fallback)
                    try:
                        original_update_data_status(self, **kwargs)
                    except:
                        pass

            # Применяем патч
            AISystem._update_data_status = patched_update_data_status

    except Exception as e:
        # Если патч не удался, продолжаем работу без него
        pass

# Применяем патч ПЕРЕД импортом глобальных переменных
_patch_ai_system_update_data_status()


_globals = globals()
_skip = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__'}

for _key, _value in _protected_module.__dict__.items():
    if _key in _skip:
        continue
    _globals[_key] = _value

del _globals, _skip, _key, _value


if __name__ == '__main__':
    _protected_module.main()
