"""
Асинхронное хранилище данных с батчингом и оптимизацией I/O операций

Оптимизации:
- Асинхронная запись файлов через ThreadPoolExecutor
- Батчинг операций записи для уменьшения количества I/O операций
- Очередь операций с автоматическим flush
- Оптимизация JSON сериализации
"""

import os
import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from threading import Lock
from collections import defaultdict

logger = logging.getLogger('AsyncStorage')

# ✅ РЕФАКТОРИНГ: Используем унифицированный ThreadPoolManager
try:
    from bot_engine.utils.thread_pool_manager import get_io_executor
except ImportError:
    # Fallback на локальную реализацию если модуль недоступен
    from concurrent.futures import ThreadPoolExecutor
    _io_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="async_storage_io")
    _io_executor_lock = Lock()
    
    def get_io_executor():
        """Получить глобальный пул потоков для I/O операций"""
        global _io_executor
        with _io_executor_lock:
            if _io_executor is None or _io_executor._shutdown:
                _io_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="async_storage_io")
        return _io_executor

# Очередь операций записи с батчингом
_write_queue: Dict[str, Dict[str, Any]] = {}
_write_queue_lock = Lock()
_write_timers: Dict[str, float] = {}
_batch_timeout = 2.0  # Секунды до автоматического flush
_max_batch_size = 10  # Максимальное количество операций в батче


async def save_json_file_async(filepath: str, data: Dict[str, Any], description: str = "данные", 
                               max_retries: int = 3, immediate: bool = False) -> bool:
    """
    Асинхронно сохраняет JSON файл с батчингом
    
    Args:
        filepath: Путь к файлу
        data: Данные для сохранения
        description: Описание данных для логов
        max_retries: Максимальное количество попыток
        immediate: Если True, сохраняет немедленно без батчинга
    
    Returns:
        bool: True если успешно сохранено
    """
    if immediate:
        # Немедленное сохранение без батчинга
        return await _save_file_immediate(filepath, data, description, max_retries)
    
    # Добавляем в очередь для батчинга
    with _write_queue_lock:
        _write_queue[filepath] = {
            'data': data,
            'description': description,
            'max_retries': max_retries,
            'timestamp': time.time()
        }
        _write_timers[filepath] = time.time()
    
    # Запускаем задачу flush если её еще нет
    asyncio.create_task(_flush_queue_if_needed())
    
    return True


async def _flush_queue_if_needed():
    """Проверяет и выполняет flush очереди если нужно"""
    await asyncio.sleep(0.1)  # Небольшая задержка для накопления операций
    
    with _write_queue_lock:
        if not _write_queue:
            return
        
        # Проверяем таймауты и размер батча
        current_time = time.time()
        files_to_flush = []
        
        for filepath, entry in _write_queue.items():
            age = current_time - entry['timestamp']
            if age >= _batch_timeout or len(_write_queue) >= _max_batch_size:
                files_to_flush.append(filepath)
        
        if not files_to_flush:
            return
        
        # Извлекаем файлы для flush
        files_data = {}
        for filepath in files_to_flush:
            files_data[filepath] = _write_queue.pop(filepath)
            _write_timers.pop(filepath, None)
    
    # Выполняем параллельную запись всех файлов
    if files_data:
        tasks = [
            _save_file_immediate(filepath, entry['data'], entry['description'], entry['max_retries'])
            for filepath, entry in files_data.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


# ✅ РЕФАКТОРИНГ: Используем унифицированный декоратор retry
def _save_file_sync_internal(filepath: str, data: Dict[str, Any], description: str) -> bool:
    """Внутренняя синхронная функция сохранения (без retry логики)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Атомарная запись через временный файл
    temp_file = filepath + '.tmp'
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Заменяем оригинальный файл
    if os.name == 'nt':  # Windows
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_file, filepath)
    else:  # Unix/Linux
        os.rename(temp_file, filepath)
    
    logger.debug(f"[ASYNC_STORAGE] {description} сохранены в {filepath}")
    return True


async def _save_file_immediate(filepath: str, data: Dict[str, Any], description: str, 
                                max_retries: int) -> bool:
    """Немедленное сохранение файла через ThreadPoolExecutor"""
    executor = get_io_executor()
    loop = asyncio.get_event_loop()
    
    # ✅ РЕФАКТОРИНГ: Используем унифицированный декоратор retry
    try:
        from bot_engine.utils.retry_utils import retry_with_backoff
        
        @retry_with_backoff(
            max_retries=max_retries,
            backoff_multiplier=2.0,
            initial_delay=0.1,
            exceptions=(OSError, PermissionError),
            on_failure=lambda e: logger.error(f"[ASYNC_STORAGE] Ошибка сохранения {description} после {max_retries} попыток: {e}")
        )
        def _save_with_retry():
            return _save_file_sync_internal(filepath, data, description)
        
        try:
            result = await loop.run_in_executor(executor, _save_with_retry)
            return result
        except Exception as e:
            logger.error(f"[ASYNC_STORAGE] Ошибка выполнения сохранения: {e}")
            return False
            
    except ImportError:
        # Fallback на старую реализацию если модуль недоступен
        def _save_sync():
            """Синхронная функция сохранения"""
            for attempt in range(max_retries):
                try:
                    return _save_file_sync_internal(filepath, data, description)
                except (OSError, PermissionError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.1 * (2 ** attempt)
                        logger.warning(f"[ASYNC_STORAGE] Попытка {attempt + 1} неудачна, повторяем через {wait_time}с: {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"[ASYNC_STORAGE] Ошибка сохранения {description} после {max_retries} попыток: {e}")
                        return False
                except Exception as e:
                    logger.error(f"[ASYNC_STORAGE] Неожиданная ошибка сохранения {description}: {e}")
                    return False
        
        try:
            result = await loop.run_in_executor(executor, _save_sync)
            return result
        except Exception as e:
            logger.error(f"[ASYNC_STORAGE] Ошибка выполнения сохранения: {e}")
            return False


async def flush_all_pending():
    """Принудительно сохраняет все ожидающие операции"""
    with _write_queue_lock:
        if not _write_queue:
            return True
        
        files_data = dict(_write_queue)
        _write_queue.clear()
        _write_timers.clear()
    
    if files_data:
        tasks = [
            _save_file_immediate(filepath, entry['data'], entry['description'], entry['max_retries'])
            for filepath, entry in files_data.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)
    
    return True


# Обертки для совместимости с существующим кодом
async def save_rsi_cache_async(coins_data: Dict, stats: Dict) -> bool:
    """Асинхронное сохранение RSI кэша"""
    from bot_engine.storage import RSI_CACHE_FILE
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'coins': coins_data,
        'stats': stats
    }
    return await save_json_file_async(RSI_CACHE_FILE, cache_data, "RSI кэш", immediate=True)


async def save_bots_state_async(bots_data: Dict, auto_bot_config: Dict) -> bool:
    """Асинхронное сохранение состояния ботов"""
    from bot_engine.storage import BOTS_STATE_FILE
    
    state_data = {
        'bots': bots_data,
        'auto_bot_config': auto_bot_config,
        'last_saved': datetime.now().isoformat(),
        'version': '1.0'
    }
    return await save_json_file_async(BOTS_STATE_FILE, state_data, "состояние ботов")


async def save_mature_coins_async(storage: Dict) -> bool:
    """Асинхронное сохранение зрелых монет"""
    from bot_engine.storage import MATURE_COINS_FILE
    return await save_json_file_async(MATURE_COINS_FILE, storage, "зрелые монеты")

