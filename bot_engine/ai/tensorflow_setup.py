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

def check_python_version():
    """Проверяет версию Python и возвращает рекомендации"""
    version = sys.version_info
    major, minor = version.major, version.minor
    
    # Python 3.13 на Windows не поддерживает GPU в TensorFlow
    if platform.system() == 'Windows' and major == 3 and minor == 13:
        return {
            'supported': False,
            'gpu_supported': False,
            'message': 'Python 3.13 на Windows не поддерживает GPU в TensorFlow. Рекомендуется Python 3.11 или 3.12 для GPU.',
            'recommended': 'Python 3.11 или 3.12'
        }
    
    # Python 3.11 и 3.12 поддерживают GPU
    if major == 3 and minor in [11, 12]:
        return {
            'supported': True,
            'gpu_supported': True,
            'message': f'Python {major}.{minor} поддерживает GPU в TensorFlow',
            'recommended': None
        }
    
    # Python 3.9 и 3.10 также поддерживают GPU
    if major == 3 and minor in [9, 10]:
        return {
            'supported': True,
            'gpu_supported': True,
            'message': f'Python {major}.{minor} поддерживает GPU в TensorFlow',
            'recommended': None
        }
    
    return {
        'supported': True,
        'gpu_supported': False,
        'message': f'Python {major}.{minor} - GPU поддержка может быть ограничена',
        'recommended': None
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
        import tensorflow as tf
        version = tf.__version__
        cuda_built = tf.test.is_built_with_cuda()
        gpus = tf.config.list_physical_devices('GPU')
        
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

def install_tensorflow_with_gpu():
    """Пытается установить TensorFlow с поддержкой GPU"""
    python_info = check_python_version()
    
    if not python_info['gpu_supported']:
        logger.warning(f"⚠️ {python_info['message']}")
        logger.info("Устанавливается TensorFlow (CPU версия)...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow>=2.13.0'],
                check=True,
                capture_output=True
            )
            return True, "CPU версия установлена"
        except subprocess.CalledProcessError as e:
            return False, f"Ошибка установки: {e}"
    
    # Пытаемся установить tensorflow[and-cuda]
    logger.info("Попытка установки TensorFlow с поддержкой GPU...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow[and-cuda]>=2.13.0'],
            check=True,
            capture_output=True,
            text=True
        )
        return True, "TensorFlow с GPU установлен"
    except subprocess.CalledProcessError:
        # Если не получилось, устанавливаем базовый TensorFlow
        logger.warning("Не удалось установить tensorflow[and-cuda], устанавливается базовая версия...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow>=2.13.0'],
                check=True,
                capture_output=True
            )
            return True, "TensorFlow установлен (базовая версия)"
        except subprocess.CalledProcessError as e:
            return False, f"Ошибка установки: {e}"

def suggest_python_downgrade():
    """Предлагает даунгрейд до Python 3.12 для работы с GPU"""
    logger.warning("")
    logger.warning("=" * 80)
    logger.warning("РЕКОМЕНДАЦИЯ: ДЛЯ РАБОТЫ С GPU НУЖЕН PYTHON 3.12")
    logger.warning("=" * 80)
    logger.warning("")
    logger.warning("Текущая версия Python не поддерживает GPU в TensorFlow на Windows.")
    logger.warning("")
    logger.warning("РЕШЕНИЕ: Автоматическая настройка Python 3.12")
    logger.warning("")
    logger.warning("Выполните команду для автоматической настройки:")
    logger.warning("  python scripts/setup_python_gpu.py")
    logger.warning("")
    logger.warning("Этот скрипт:")
    logger.warning("  1. Проверит наличие Python 3.12 в системе")
    logger.warning("  2. Создаст виртуальное окружение .venv_gpu с Python 3.12")
    logger.warning("  3. Установит все зависимости включая TensorFlow с GPU")
    logger.warning("  4. Проверит работу GPU")
    logger.warning("")
    logger.warning("После настройки используйте:")
    if platform.system() == 'Windows':
        logger.warning("  .venv_gpu\\Scripts\\activate")
    else:
        logger.warning("  source .venv_gpu/bin/activate")
    logger.warning("  python ai.py")
    logger.warning("")
    logger.warning("=" * 80)
    logger.warning("")

def ensure_tensorflow_setup():
    """
    Главная функция: проверяет и при необходимости устанавливает TensorFlow
    Вызывается автоматически при импорте модуля
    """
    # Проверяем версию Python
    python_info = check_python_version()
    
    # Проверяем наличие GPU
    has_gpu = check_gpu_available()
    
    # Если есть GPU, но Python не поддерживает GPU - предлагаем даунгрейд
    if has_gpu and not python_info['gpu_supported']:
        suggest_python_downgrade()
        # Продолжаем с CPU версией, но предупреждаем пользователя
    
    # Проверяем установку TensorFlow
    tf_info = check_tensorflow_installation()
    
    if not tf_info['installed']:
        logger.info("TensorFlow не установлен. Начинаю автоматическую установку...")
        
        if has_gpu and python_info['gpu_supported']:
            logger.info("Обнаружен GPU, устанавливается TensorFlow с поддержкой GPU...")
        else:
            if has_gpu:
                logger.warning(f"GPU обнаружен, но {python_info['message']}")
            logger.info("Устанавливается TensorFlow (CPU версия)...")
        
        success, message = install_tensorflow_with_gpu()
        if success:
            logger.info(f"✅ {message}")
            # Перепроверяем установку
            tf_info = check_tensorflow_installation()
        else:
            logger.error(f"❌ {message}")
            return False
    
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
                    logger.warning("   GPU обнаружен в системе, но текущая версия Python не поддерживает GPU")
                    logger.warning(f"   {python_info['message']}")
                    if python_info['recommended']:
                        logger.warning(f"   Для работы с GPU рекомендуется использовать {python_info['recommended']}")
                        logger.warning("   Запустите: python scripts/setup_python_gpu.py")
                else:
                    logger.warning("   GPU обнаружен в системе, но TensorFlow не может его использовать")
                    logger.warning(f"   {python_info['message']}")
    
    return True

# Автоматически проверяем при импорте модуля
if __name__ != '__main__':
    # Вызываем только если это не прямой запуск скрипта
    try:
        ensure_tensorflow_setup()
    except Exception as e:
        logger.debug(f"Ошибка при автоматической проверке TensorFlow: {e}")
