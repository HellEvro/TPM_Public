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
    """Проверяет версию Python. Проект требует Python 3.12."""
    version = sys.version_info
    major, minor = version.major, version.minor
    
    if major == 3 and minor == 12:
        return {
            'supported': True,
            'gpu_supported': True,
            'message': 'Python 3.12 поддерживает GPU в TensorFlow',
            'recommended': None
        }
    
    # Все остальные версии — не поддерживаются, нужен 3.12
    return {
        'supported': False,
        'gpu_supported': False,
        'message': f'Python {major}.{minor} не поддерживается. Требуется Python 3.12. Выполните: python scripts/setup_python_gpu.py или установите Python 3.12.',
        'recommended': 'Python 3.12'
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
            logger.debug(f"Ошибка при поиске GPU: {e}")
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
    """Устанавливает TensorFlow (с GPU при Python 3.12 и наличии GPU)"""
    python_info = check_python_version()
    
    if not python_info['supported']:
        logger.warning("Требуется Python 3.12. Установка TensorFlow пропущена.")
        return False, python_info['message']
    
    # Если нет GPU или Python не поддерживает GPU - сразу устанавливаем CPU версию
    if not has_gpu or not python_info['gpu_supported']:
        if not has_gpu:
            logger.info("NVIDIA GPU не обнаружен. Устанавливается TensorFlow (CPU версия)...")
        else:
            logger.warning(f"⚠️ {python_info['message']}")
            logger.info("Устанавливается TensorFlow (CPU версия)...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow', '--no-warn-script-location'],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 минут таймаут
            )
            return True, "CPU версия установлена"
        except subprocess.TimeoutExpired:
            return False, "Таймаут при установке TensorFlow (более 5 минут)"
        except subprocess.CalledProcessError as e:
            err = e.stderr
            error_output = (err.decode('utf-8', errors='ignore') if isinstance(err, bytes) else (err or str(e)))
            logger.error(f"Детали ошибки установки: {error_output[:500]}")
            return False, f"Ошибка установки TensorFlow. Проверьте подключение к интернету и права доступа."
    
    # Только если есть GPU И Python поддерживает GPU - пытаемся установить tensorflow[and-cuda]
    logger.info("Обнаружен NVIDIA GPU. Попытка установки TensorFlow с поддержкой GPU...")
    try:
        # Пробуем установить без указания версии, чтобы pip сам выбрал совместимую
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow[and-cuda]', '--no-warn-script-location'],
            check=True,
            capture_output=True,
            text=True,
            timeout=900  # 15 минут таймаут для GPU версии
        )
        return True, "TensorFlow с GPU установлен"
    except subprocess.TimeoutExpired:
        logger.warning("Таймаут при установке tensorflow[and-cuda], пробуем базовую версию...")
    except subprocess.CalledProcessError as e:
        err = e.stderr
        error_output = (err.decode('utf-8', errors='ignore') if isinstance(err, bytes) else (err or str(e)))
        logger.warning(f"Не удалось установить tensorflow[and-cuda]: {error_output[:300]}")
        logger.warning("Устанавливается базовая версия TensorFlow (CPU)...")
    
    # Если не получилось, устанавливаем базовый TensorFlow
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow', '--no-warn-script-location'],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 минут таймаут
        )
        return True, "TensorFlow установлен (базовая версия)"
    except subprocess.TimeoutExpired:
        return False, "Таймаут при установке TensorFlow (более 5 минут)"
    except subprocess.CalledProcessError as e:
        err = e.stderr
        error_output = (err.decode('utf-8', errors='ignore') if isinstance(err, bytes) else (err or str(e)))
        logger.error(f"Детали ошибки установки: {error_output[:500]}")
        return False, f"Ошибка установки TensorFlow. Проверьте подключение к интернету и права доступа."

def suggest_python_downgrade():
    """Напоминание: проект требует Python 3.12"""
    global _gpu_warning_shown
    if _gpu_warning_shown:
        return
    _gpu_warning_shown = True
    logger.warning("=" * 80)
    logger.warning("InfoBot требует Python 3.12. Выполните: python scripts/setup_python_gpu.py")
    logger.warning("Создаст .venv_gpu с Python 3.12. ai.py автоматически использует его.")
    logger.warning("=" * 80)

def ensure_tensorflow_setup():
    """
    Главная функция: проверяет и при необходимости устанавливает TensorFlow
    Вызывается автоматически при импорте модуля
    """
    global _tensorflow_checked
    
    # Проверяем только один раз во всей программе
    if _tensorflow_checked:
        logger.debug("TensorFlow уже проверен, пропускаем повторную проверку")
        return True
    
    # Проверяем, что мы в главном процессе (для предотвращения дублирования в дочерних процессах)
    try:
        import multiprocessing
        is_main_process = multiprocessing.current_process().name == 'MainProcess'
        if not is_main_process:
            logger.debug("Дочерний процесс - пропускаем проверку TensorFlow")
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
            # Устанавливаем TensorFlow только если есть GPU
            # На ПК без GPU TensorFlow не нужен для работы системы
            if has_gpu:
                logger.info("TensorFlow не установлен. Начинаю автоматическую установку (обнаружен GPU)...")
                
                success, message = install_tensorflow_with_gpu(has_gpu=has_gpu)
                if success:
                    logger.info(f"✅ {message}")
                    # Перепроверяем установку
                    tf_info = check_tensorflow_installation()
                else:
                    logger.error(f"❌ {message}")
                    return False
            else:
                logger.info("ℹ️ TensorFlow не установлен, но GPU не обнаружен. Пропускаем установку TensorFlow.")
                logger.info("ℹ️ AI система будет работать без TensorFlow (LSTM и некоторые функции будут недоступны).")
                logger.info("ℹ️ Для использования GPU установите TensorFlow вручную на ПК с видеокартой.")
        
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
                        logger.info("   🔄 Пытаюсь автоматически переустановить TensorFlow с поддержкой GPU...")
                        logger.info("   ⏳ Это может занять несколько минут...")
                        # Сначала удаляем старую версию
                        try:
                            logger.info("   📦 Удаление старой версии TensorFlow...")
                            subprocess.run(
                                [sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu'],
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=60
                            )
                        except Exception as e:
                            logger.debug(f"   Предупреждение при удалении: {e}")
                        
                        # Пытаемся автоматически установить GPU версию
                        try:
                            logger.info("   📦 Установка TensorFlow с поддержкой GPU...")
                            result = subprocess.run(
                                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow[and-cuda]', '--no-warn-script-location'],
                                check=True,
                                capture_output=True,
                                text=True,
                                timeout=900  # 15 минут таймаут для GPU версии
                            )
                            logger.info("   ✅ TensorFlow с GPU успешно установлен! Перезапустите ai.py для применения изменений.")
                            # Перепроверяем установку
                            tf_info = check_tensorflow_installation()
                            if tf_info['cuda_built']:
                                logger.info("   ✅ TensorFlow теперь поддерживает CUDA")
                                if tf_info['gpus_found'] > 0:
                                    logger.info(f"   ✅ Найдено GPU устройств: {tf_info['gpus_found']}")
                        except subprocess.TimeoutExpired:
                            logger.warning("   ⚠️ Таймаут при установке tensorflow[and-cuda]")
                            logger.warning("   Установите вручную: pip uninstall -y tensorflow tensorflow-cpu && pip install tensorflow[and-cuda]")
                        except subprocess.CalledProcessError as e:
                            err = e.stderr
                            error_output = (err.decode('utf-8', errors='ignore') if isinstance(err, bytes) else (err or str(e)))
                            logger.warning(f"   ⚠️ Не удалось установить tensorflow[and-cuda]: {error_output[:300]}")
                            logger.warning("   Установите вручную: pip uninstall -y tensorflow tensorflow-cpu && pip install tensorflow[and-cuda]")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Ошибка при установке: {e}")
                            logger.warning("   Установите вручную: pip uninstall -y tensorflow tensorflow-cpu && pip install tensorflow[and-cuda]")
    except Exception as e:
        logger.warning(f"Ошибка при проверке TensorFlow: {e}")
        logger.info("Продолжаем работу...")
        return True
    
    return True

# НЕ вызываем автоматически при импорте - только по явному запросу
# Это предотвращает множественные вызовы из разных модулей
