"""
Менеджер пулов потоков для унификации использования ThreadPoolExecutor

Централизованное управление пулами потоков во всем проекте
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger('ThreadPoolManager')


class ThreadPoolManager:
    """Менеджер для управления пулами потоков"""
    
    # Глобальные пулы потоков
    _io_executor: Optional[ThreadPoolExecutor] = None
    _calculation_executor: Optional[ThreadPoolExecutor] = None
    _api_executor: Optional[ThreadPoolExecutor] = None
    
    # Настройки по умолчанию
    DEFAULT_IO_WORKERS = 3
    DEFAULT_CALCULATION_WORKERS = 20
    DEFAULT_API_WORKERS = 10
    
    @classmethod
    def get_io_executor(cls, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Получить пул потоков для I/O операций
        
        Args:
            max_workers: Максимальное количество потоков (по умолчанию DEFAULT_IO_WORKERS)
        
        Returns:
            ThreadPoolExecutor для I/O операций
        """
        if cls._io_executor is None:
            workers = max_workers or cls.DEFAULT_IO_WORKERS
            cls._io_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="io")
            logger.info(f"[THREAD_POOL] Создан I/O executor с {workers} потоками")
        
        return cls._io_executor
    
    @classmethod
    def get_calculation_executor(cls, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Получить пул потоков для расчетов
        
        Args:
            max_workers: Максимальное количество потоков (по умолчанию DEFAULT_CALCULATION_WORKERS)
        
        Returns:
            ThreadPoolExecutor для расчетов
        """
        if cls._calculation_executor is None:
            workers = max_workers or cls.DEFAULT_CALCULATION_WORKERS
            cls._calculation_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="calc")
            logger.info(f"[THREAD_POOL] Создан Calculation executor с {workers} потоками")
        
        return cls._calculation_executor
    
    @classmethod
    def get_api_executor(cls, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Получить пул потоков для API запросов
        
        Args:
            max_workers: Максимальное количество потоков (по умолчанию DEFAULT_API_WORKERS)
        
        Returns:
            ThreadPoolExecutor для API запросов
        """
        if cls._api_executor is None:
            workers = max_workers or cls.DEFAULT_API_WORKERS
            cls._api_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="api")
            logger.info(f"[THREAD_POOL] Создан API executor с {workers} потоками")
        
        return cls._api_executor
    
    @classmethod
    def shutdown_all(cls, wait: bool = True):
        """Закрыть все пулы потоков"""
        if cls._io_executor:
            cls._io_executor.shutdown(wait=wait)
            cls._io_executor = None
            logger.info("[THREAD_POOL] I/O executor закрыт")
        
        if cls._calculation_executor:
            cls._calculation_executor.shutdown(wait=wait)
            cls._calculation_executor = None
            logger.info("[THREAD_POOL] Calculation executor закрыт")
        
        if cls._api_executor:
            cls._api_executor.shutdown(wait=wait)
            cls._api_executor = None
            logger.info("[THREAD_POOL] API executor закрыт")


def get_io_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """Удобная функция для получения I/O executor"""
    return ThreadPoolManager.get_io_executor(max_workers)


def get_calculation_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """Удобная функция для получения Calculation executor"""
    return ThreadPoolManager.get_calculation_executor(max_workers)


def get_api_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """Удобная функция для получения API executor"""
    return ThreadPoolManager.get_api_executor(max_workers)

