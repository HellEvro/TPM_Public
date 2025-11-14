"""
Утилиты для обработки повторных попыток (retry) с экспоненциальным backoff

Унифицированная логика retry для всего проекта
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
import asyncio

logger = logging.getLogger('RetryUtils')


def retry_with_backoff(
    max_retries: int = 3,
    backoff_multiplier: float = 2.0,
    initial_delay: float = 0.1,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None
):
    """
    Декоратор для повторных попыток с экспоненциальным backoff
    
    Args:
        max_retries: Максимальное количество попыток
        backoff_multiplier: Множитель для экспоненциального backoff
        initial_delay: Начальная задержка в секундах
        exceptions: Кортеж исключений, при которых нужно повторять попытку
        on_retry: Функция вызываемая при каждой повторной попытке (attempt_num, exception)
        on_failure: Функция вызываемая при окончательной неудаче (exception)
    
    Returns:
        Декорированная функция
    
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def save_file(filepath, data):
            with open(filepath, 'w') as f:
                json.dump(data, f)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (backoff_multiplier ** attempt)
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        else:
                            logger.warning(
                                f"[RETRY] Попытка {attempt + 1}/{max_retries} неудачна для {func.__name__}, "
                                f"повторяем через {wait_time:.3f}с: {e}"
                            )
                        
                        time.sleep(wait_time)
                    else:
                        if on_failure:
                            on_failure(e)
                        else:
                            logger.error(
                                f"[RETRY] Все {max_retries} попыток провалились для {func.__name__}: {e}"
                            )
                        raise
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    backoff_multiplier: float = 2.0,
    initial_delay: float = 0.1,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None
):
    """
    Асинхронный декоратор для повторных попыток с экспоненциальным backoff
    
    Args:
        max_retries: Максимальное количество попыток
        backoff_multiplier: Множитель для экспоненциального backoff
        initial_delay: Начальная задержка в секундах
        exceptions: Кортеж исключений, при которых нужно повторять попытку
        on_retry: Функция вызываемая при каждой повторной попытке (attempt_num, exception)
        on_failure: Функция вызываемая при окончательной неудаче (exception)
    
    Returns:
        Декорированная async функция
    
    Example:
        @async_retry_with_backoff(max_retries=3, initial_delay=0.1)
        async def save_file_async(filepath, data):
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data))
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (backoff_multiplier ** attempt)
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        else:
                            logger.warning(
                                f"[RETRY] Попытка {attempt + 1}/{max_retries} неудачна для {func.__name__}, "
                                f"повторяем через {wait_time:.3f}с: {e}"
                            )
                        
                        await asyncio.sleep(wait_time)
                    else:
                        if on_failure:
                            on_failure(e)
                        else:
                            logger.error(
                                f"[RETRY] Все {max_retries} попыток провалились для {func.__name__}: {e}"
                            )
                        raise
        
        return wrapper
    return decorator


def retry_on_specific_errors(
    max_retries: int = 3,
    backoff_multiplier: float = 2.0,
    initial_delay: float = 0.1,
    error_keywords: Tuple[str, ...] = ('rate limit', 'timeout', 'connection'),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Декоратор для повторных попыток только при определенных ошибках
    
    Args:
        max_retries: Максимальное количество попыток
        backoff_multiplier: Множитель для экспоненциального backoff
        initial_delay: Начальная задержка в секундах
        error_keywords: Кортеж ключевых слов в сообщении об ошибке
        on_retry: Функция вызываемая при каждой повторной попытке
    
    Returns:
        Декорированная функция
    
    Example:
        @retry_on_specific_errors(error_keywords=('rate limit', 'timeout'))
        def api_request():
            return exchange.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    should_retry = any(keyword in error_str for keyword in error_keywords)
                    
                    if should_retry and attempt < max_retries - 1:
                        wait_time = initial_delay * (backoff_multiplier ** attempt)
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        else:
                            logger.warning(
                                f"[RETRY] Попытка {attempt + 1}/{max_retries} неудачна для {func.__name__} "
                                f"(ошибка: {error_str}), повторяем через {wait_time:.3f}с"
                            )
                        
                        time.sleep(wait_time)
                    else:
                        # Не retry ошибка или последняя попытка - пробрасываем исключение
                        raise
        
        return wrapper
    return decorator

