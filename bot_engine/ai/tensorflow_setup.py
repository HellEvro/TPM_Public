#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическая проверка и установка TensorFlow с поддержкой GPU
Вызывается автоматически при запуске ai.py
"""

import sys
import subprocess
import logging
import platform

logger = logging.getLogger('TensorFlowSetup')

# Глобальные флаги для предотвращения дублирования
_gpu_warning_shown = False
_tensorflow_checked = False

def check_python_version():
    """Проверяет версию Python. Проект требует Python 3.14+."""
    version = sys.version_info
    major, minor = version.major, version.minor

    if major == 3 and minor >= 14:
        return {
            'supported': True,
            'gpu_supported': False,  # TensorFlow не поддерживает 3.14+
            'message': f'Python {major}.{minor} поддерживается, но TensorFlow требует Python 3.12 для GPU',
            'recommended': 'PyTorch в .venv (TensorFlow не используется)'
        }

    # Версии ниже 3.14 не поддерживаются
    return {
        'supported': False,
        'gpu_supported': False,
        'message': f'Python {major}.{minor} не поддерживается. Требуется Python 3.14+. Выполните: python scripts/ensure_python314_venv.py или установите Python 3.14+.',
        'recommended': 'Python 3.14+'
    }

def check_gpu_available():
    """Проверяет наличие NVIDIA GPU в системе"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

def check_tensorflow_installation():
    """Проверяет установку TensorFlow и поддержку CUDA"""
    try:
        logger.info("Импорт TensorFlow...")
        import tensorflow as tf

        logger.info("Получение информации о TensorFlow...")
        version = tf.__version__

        logger.info("Проверка поддержки CUDA...")
        cuda_built = tf.test.is_built_with_cuda()

        logger.info("Поиск GPU устройств (это может занять несколько секунд)...")
        # Поиск GPU может занимать время, особенно при первом запуске
        gpus = []
        try:
            gpus = tf.config.list_physical_devices('GPU')
        except Exception as e:

            gpus = []

        return {
            'installed': True,
            'version': version,
            'cuda_built': cuda_built,
            'gpus_found': len(gpus),
            'gpu_devices': gpus
        }
    except ImportError:
        return {
            'installed': False,
            'version': None,
            'cuda_built': False,
            'gpus_found': 0,
            'gpu_devices': []
        }
    except Exception as e:
        logger.warning(f"Ошибка при проверке TensorFlow: {e}")
        # Возвращаем частичную информацию
        try:
            import tensorflow as tf
            return {
                'installed': True,
                'version': tf.__version__,
                'cuda_built': False,
                'gpus_found': 0,
                'gpu_devices': []
            }
        except:
            return {
                'installed': False,
                'version': None,
                'cuda_built': False,
                'gpus_found': 0,
                'gpu_devices': []
            }

def install_tensorflow_with_gpu(has_gpu=False):
    """УСТАРЕЛО: Установка TensorFlow выполняется ТОЛЬКО через requirements.txt"""
    logger.info("💡 TensorFlow устанавливается через: pip install -r requirements.txt")
    if has_gpu:
        logger.info("💡 Для GPU поддержки: pip install tensorflow[and-cuda]")
    return False, "Установка TensorFlow выполняется через requirements.txt"

def suggest_python_downgrade():
    """Напоминание: проект требует Python 3.12"""
    global _gpu_warning_shown
    if _gpu_warning_shown:
        return
    _gpu_warning_shown = True
    logger.warning("=" * 80)
    logger.warning("InfoBot требует Python 3.12+. Выполните: python scripts/setup_python_gpu.py (PyTorch в .venv)")
    logger.warning("=" * 80)

def ensure_tensorflow_setup():
    """
    Главная функция: проверяет наличие TensorFlow (установка выполняется через requirements.txt)
    Вызывается автоматически при импорте модуля
    """
    global _tensorflow_checked

    # Проверяем только один раз во всей программе
    if _tensorflow_checked:

        return True

    # Проверяем, что мы в главном процессе (для предотвращения дублирования в дочерних процессах)
    try:
        import multiprocessing
        is_main_process = multiprocessing.current_process().name == 'MainProcess'
        if not is_main_process:

            return True
    except:
        # Если multiprocessing недоступен, продолжаем
        pass

    _tensorflow_checked = True

    try:
        # Проверяем версию Python
        python_info = check_python_version()

        # Проверяем наличие GPU
        logger.info("Проверка наличия GPU в системе...")
        has_gpu = check_gpu_available()
        if has_gpu:
            logger.info("✅ NVIDIA GPU обнаружен в системе")
        else:
            logger.info("ℹ️ NVIDIA GPU не обнаружен, будет использоваться CPU")

        # Если есть GPU, но Python не поддерживает GPU - предлагаем даунгрейд
        if has_gpu and not python_info['gpu_supported']:
            suggest_python_downgrade()
            # Продолжаем с CPU версией, но предупреждаем пользователя

        # Проверяем установку TensorFlow
        logger.info("Проверка установки TensorFlow...")
        tf_info = check_tensorflow_installation()

        if not tf_info['installed']:
            # TensorFlow должен устанавливаться через requirements.txt
            logger.warning("⚠️ TensorFlow не установлен")
            logger.info("💡 Установите TensorFlow через: pip install -r requirements.txt")
            if has_gpu:
                logger.info("💡 Для GPU поддержки: pip install tensorflow[and-cuda]")
            logger.info("ℹ️ AI система будет работать без TensorFlow (LSTM и некоторые функции будут недоступны).")

        # Выводим информацию о TensorFlow
        if tf_info['installed']:
            logger.info(f"TensorFlow версия: {tf_info['version']}")

            if tf_info['cuda_built']:
                logger.info("✅ TensorFlow скомпилирован с поддержкой CUDA")
                if tf_info['gpus_found'] > 0:
                    logger.info(f"✅ Найдено GPU устройств: {tf_info['gpus_found']}")
                    for i, gpu in enumerate(tf_info['gpu_devices']):
                        logger.info(f"   GPU {i}: {gpu.name}")
                else:
                    logger.warning("⚠️ GPU устройства не найдены TensorFlow")
                    if check_gpu_available():
                        logger.warning("   GPU обнаружен в системе, но TensorFlow его не видит")
                        logger.warning("   Возможно, требуется установка CUDA библиотек вручную")
            else:
                logger.warning("⚠️ TensorFlow установлен БЕЗ поддержки CUDA (CPU версия)")
                if has_gpu:
                    if not python_info['gpu_supported']:
                        # Сообщение о даунгрейде уже показано выше через suggest_python_downgrade()
                        # Не дублируем его здесь
                        pass
                    else:
                        logger.warning("   GPU обнаружен в системе, но TensorFlow не может его использовать")
                        logger.warning("   ⚠️ ВАЖНО: TensorFlow для Python 3.12 может быть собран только с CPU поддержкой")
                        logger.warning("   ⚠️ Официальные сборки TensorFlow для Python 3.12 не включают GPU поддержку")
                        logger.info("   💡 РЕШЕНИЯ:")
                        logger.info("      1. Используйте Python 3.11 для TensorFlow GPU (рекомендуется)")
                        logger.info("      2. Запустите: python scripts/setup_python_gpu.py для создания venv с Python 3.11")
                        logger.info("      3. Или используйте TensorFlow CPU версию (уже установлена)")
                        logger.info("   ℹ️ Система будет работать на CPU, но медленнее для обучения LSTM моделей")
    except Exception as e:
        logger.warning(f"Ошибка при проверке TensorFlow: {e}")
        logger.info("Продолжаем работу...")
        return True

    return True

# НЕ вызываем автоматически при импорте - только по явному запросу
# Это предотвращает множественные вызовы из разных модулей
