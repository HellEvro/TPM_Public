#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Оболочка для защищённого AI лаунчера.
Вся рабочая логика находится в bot_engine/ai/_ai_launcher.pyc
"""

# ⚠️ КРИТИЧНО: Устанавливаем переменную окружения для идентификации процесса ai.py
# Это гарантирует, что функции из filters.py будут сохранять свечи в ai_data.db, а не в bots_data.db
import os
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

# КРИТИЧНО: Проверка и установка TensorFlow ПЕРЕД импортом защищенного модуля
try:
    from bot_engine.ai.tensorflow_setup import ensure_tensorflow_setup
    # Настраиваем логгер для tensorflow_setup, чтобы видеть сообщения
    tf_setup_logger = logging.getLogger('TensorFlowSetup')
    tf_setup_logger.setLevel(logging.INFO)
    # Вызываем проверку и установку TensorFlow
    ensure_tensorflow_setup()
    
    # КРИТИЧНО: Принудительная настройка GPU для TensorFlow
    try:
        import tensorflow as tf
        import os
        
        # Настраиваем переменные окружения для CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        # Добавляем пути к CUDA в PATH (если CUDA установлен)
        cuda_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
        ]
        for path in cuda_paths:
            if os.path.exists(path) and path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
        
        # Настраиваем GPU при запуске
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Включаем рост памяти GPU
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
            # Устанавливаем видимость GPU для всех операций
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logging.info(f"✅ GPU настроен: {gpus[0].name}")
            except:
                logging.info(f"✅ GPU найден: {gpus[0].name}")
        else:
            # Пробуем использовать системные CUDA библиотеки
            cuda_available = tf.test.is_built_with_cuda()
            if not cuda_available:
                logging.warning("⚠️ TensorFlow собран без CUDA. GPU недоступен.")
                logging.warning("⚠️ Для использования GPU рекомендуется:")
                logging.warning("   1. Установить Python 3.11: https://www.python.org/downloads/release/python-3110/")
                logging.warning("   2. Запустить: python scripts/install_tensorflow_gpu_python311.py")
                logging.warning("   3. Или использовать: .venv_gpu\\Scripts\\python ai.py (после установки Python 3.11)")
            else:
                logging.info("ℹ️ GPU устройства не найдены, но CUDA поддержка есть")
    except Exception as gpu_error:
        logging.debug(f"Ошибка настройки GPU: {gpu_error}")
except Exception as e:
    # Если не удалось проверить TensorFlow, продолжаем работу
    import sys
    sys.stderr.write(f"⚠️ Предупреждение: не удалось проверить TensorFlow: {e}\n")
    import traceback
    traceback.print_exc()

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
