"""
Утилиты для работы с блокировками (locks) с поддержкой таймаутов

Унифицированная работа с блокировками во всем проекте
"""

import threading
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger('LockUtils')


@contextmanager
def acquire_lock_with_timeout(lock: threading.Lock, timeout: Optional[float] = None, 
                              description: str = "lock"):
    """
    Контекстный менеджер для безопасного получения блокировки с таймаутом
    
    Args:
        lock: threading.Lock для получения
        timeout: Таймаут в секундах (None = бесконечное ожидание)
        description: Описание блокировки для логирования
    
    Yields:
        True если блокировка получена, False если таймаут
    
    Example:
        with acquire_lock_with_timeout(my_lock, timeout=5.0, description="data lock") as acquired:
            if acquired:
                # Работа с данными
                pass
    """
    acquired = False
    try:
        if timeout is None:
            lock.acquire()
            acquired = True
        else:
            acquired = lock.acquire(timeout=timeout)
        
        if not acquired:
            logger.warning(f"[LOCK] Таймаут получения блокировки {description} ({timeout}с)")
            yield False
        else:
            yield True
    finally:
        if acquired:
            lock.release()


@contextmanager
def acquire_rlock_with_timeout(lock: threading.RLock, timeout: Optional[float] = None,
                               description: str = "lock"):
    """
    Контекстный менеджер для безопасного получения рекурсивной блокировки с таймаутом
    
    Args:
        lock: threading.RLock для получения
        timeout: Таймаут в секундах (None = бесконечное ожидание)
        description: Описание блокировки для логирования
    
    Yields:
        True если блокировка получена, False если таймаут
    
    Example:
        with acquire_rlock_with_timeout(my_rlock, timeout=5.0) as acquired:
            if acquired:
                # Работа с данными
                pass
    """
    acquired = False
    try:
        if timeout is None:
            lock.acquire()
            acquired = True
        else:
            acquired = lock.acquire(timeout=timeout)
        
        if not acquired:
            logger.warning(f"[LOCK] Таймаут получения рекурсивной блокировки {description} ({timeout}с)")
            yield False
        else:
            yield True
    finally:
        if acquired:
            lock.release()


def safe_lock_operation(lock: threading.Lock, operation, timeout: Optional[float] = None,
                       description: str = "operation", default_return=None):
    """
    Безопасное выполнение операции с блокировкой и таймаутом
    
    Args:
        lock: threading.Lock для получения
        operation: Функция для выполнения
        timeout: Таймаут в секундах
        description: Описание операции для логирования
        default_return: Значение по умолчанию если таймаут
    
    Returns:
        Результат операции или default_return при таймауте
    
    Example:
        result = safe_lock_operation(
            my_lock,
            lambda: modify_data(),
            timeout=5.0,
            description="data modification",
            default_return=False
        )
    """
    with acquire_lock_with_timeout(lock, timeout, description) as acquired:
        if acquired:
            return operation()
        else:
            return default_return

