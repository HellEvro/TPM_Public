#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Оболочка для защищённого AI лаунчера.
Вся рабочая логика находится в bot_engine/ai/_ai_launcher.pyc
"""

import sys
import os
import subprocess
from pathlib import Path

# ⚠️ КРИТИЧНО: Автоматическое переключение на .venv_gpu для TensorFlow
# TensorFlow требует Python 3.12, а основное приложение работает на Python 3.14+
if sys.version_info >= (3, 14) and os.environ.get('INFOBOT_AI_VENV_RESTART') != 'true':
    project_root = Path(__file__).resolve().parent
    venv_gpu = project_root / '.venv_gpu'
    
    # Определяем путь к Python в .venv_gpu
    if os.name == 'nt':  # Windows
        venv_python = venv_gpu / 'Scripts' / 'python.exe'
    else:  # Linux/macOS
        venv_python = venv_gpu / 'bin' / 'python'
    
    # Если .venv_gpu не существует, создаем его
    if not venv_gpu.exists() or not venv_python.exists():
        print("[INFO] Python 3.14 обнаружен, создаю .venv_gpu (Python 3.12) для TensorFlow...")
        setup_script = project_root / 'scripts' / 'setup_python_gpu.py'
        if setup_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(setup_script)],
                    cwd=str(project_root),
                    timeout=600,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"[ERROR] Ошибка создания .venv_gpu: {result.stderr}")
                    sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Ошибка запуска setup_python_gpu.py: {e}")
                sys.exit(1)
        else:
            print("[ERROR] Скрипт setup_python_gpu.py не найден!")
            sys.exit(1)
    
    # Проверяем, установлен ли TensorFlow в .venv_gpu
    if venv_python.exists():
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'import tensorflow; print(tensorflow.__version__)'],
                capture_output=True,
                text=True,
                timeout=10
            )
            tensorflow_installed = result.returncode == 0
        except:
            tensorflow_installed = False
        
        # Если TensorFlow не установлен, устанавливаем зависимости
        if not tensorflow_installed:
            print("[INFO] TensorFlow не найден в .venv_gpu, устанавливаю зависимости...")
            requirements = project_root / 'requirements.txt'
            if requirements.exists():
                try:
                    # Временно модифицируем requirements.txt для Python 3.12
                    import tempfile
                    import re
                    with open(requirements, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Заменяем TensorFlow на версию для Python 3.12
                    content = re.sub(r'tensorflow[^;\n]*;.*python_version.*', 'tensorflow==2.15.0', content)
                    content = re.sub(r'tf-nightly[^;\n]*;.*python_version.*', 'tensorflow==2.15.0', content)
                    content = content.replace('tf-nightly>=2.21.0.dev', 'tensorflow==2.15.0')
                    content = content.replace('tensorflow>=2.15.0; python_version < "3.13"', 'tensorflow==2.15.0')
                    content = content.replace('tensorflow>=2.16.0', 'tensorflow==2.15.0')
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    
                    try:
                        result = subprocess.run(
                            [str(venv_python), '-m', 'pip', 'install', '-r', tmp_path, '--no-warn-script-location'],
                            cwd=str(project_root),
                            timeout=600,
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            print("[OK] Зависимости установлены (включая TensorFlow 2.15.0)")
                        else:
                            print(f"[WARNING] Ошибка установки зависимостей: {result.stderr}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                except Exception as e:
                    print(f"[WARNING] Ошибка установки зависимостей: {e}")
        
        # Перезапускаем себя с Python из .venv_gpu
        print("[INFO] Python 3.14 обнаружен, переключаюсь на .venv_gpu (Python 3.12) для TensorFlow")
        os.environ['INFOBOT_AI_VENV_RESTART'] = 'true'
        try:
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        except Exception as e:
            print(f"[ERROR] Ошибка переключения на .venv_gpu: {e}")
            sys.exit(1)

# ⚠️ КРИТИЧНО: Устанавливаем переменную окружения для идентификации процесса ai.py
# Это гарантирует, что функции из filters.py будут сохранять свечи в ai_data.db, а не в bots_data.db
os.environ['INFOBOT_AI_PROCESS'] = 'true'

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

from typing import TYPE_CHECKING, Any
from bot_engine.ai import _infobot_ai_protected as _protected_module


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
