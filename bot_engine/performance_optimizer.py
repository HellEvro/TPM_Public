"""
Модуль для интеграции оптимизаций производительности

Объединяет:
- Асинхронное хранилище (async_storage.py)
- Оптимизированный клиент биржи (optimized_exchange_client.py)
- Оптимизированные расчеты (optimized_calculations.py)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger('PerformanceOptimizer')

# Импорты оптимизированных модулей
try:
    from bot_engine.async_storage import (
        save_json_file_async, save_rsi_cache_async, 
        save_bots_state_async, save_mature_coins_async, flush_all_pending
    )
    ASYNC_STORAGE_AVAILABLE = True
except ImportError:
    ASYNC_STORAGE_AVAILABLE = False
    logger.warning("[PERF_OPT] Асинхронное хранилище недоступно")

try:
    from bot_engine.optimized_exchange_client import OptimizedExchangeClient
    OPT_EXCHANGE_AVAILABLE = True
except ImportError:
    OPT_EXCHANGE_AVAILABLE = False
    logger.warning("[PERF_OPT] Оптимизированный клиент биржи недоступно")

try:
    from bot_engine.optimized_calculations import (
        calculate_rsi_batch, calculate_ema_batch,
        calculate_rsi_vectorized, calculate_ema_vectorized,
        process_coins_batch
    )
    OPT_CALC_AVAILABLE = True
except ImportError:
    OPT_CALC_AVAILABLE = False
    logger.warning("[PERF_OPT] Оптимизированные расчеты недоступны")


class PerformanceOptimizer:
    """Класс для управления оптимизациями производительности"""
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Включить ли оптимизации
        """
        self.enabled = enabled
        self.exchange_client = None
        self._stats = {
            'storage_operations': 0,
            'storage_time_saved': 0.0,
            'exchange_requests': 0,
            'exchange_time_saved': 0.0,
            'calculations': 0,
            'calculation_time_saved': 0.0
        }
    
    async def initialize_exchange_client(self, base_url: str, 
                                        max_connections: int = 100):
        """Инициализирует оптимизированный клиент биржи"""
        if not OPT_EXCHANGE_AVAILABLE or not self.enabled:
            return None
        
        try:
            self.exchange_client = OptimizedExchangeClient(
                base_url=base_url,
                max_connections=max_connections,
                max_connections_per_host=30
            )
            await self.exchange_client.__aenter__()
            logger.info("[PERF_OPT] Оптимизированный клиент биржи инициализирован")
            return self.exchange_client
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка инициализации клиента биржи: {e}")
            return None
    
    async def save_data_optimized(self, filepath: str, data: Dict[str, Any],
                                  description: str = "данные", immediate: bool = False) -> bool:
        """Оптимизированное сохранение данных"""
        if not ASYNC_STORAGE_AVAILABLE or not self.enabled:
            # Fallback на синхронное сохранение
            from bot_engine.storage import save_json_file
            return save_json_file(filepath, data, description)
        
        try:
            result = await save_json_file_async(filepath, data, description, immediate=immediate)
            if result:
                self._stats['storage_operations'] += 1
            return result
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка асинхронного сохранения: {e}")
            # Fallback на синхронное сохранение
            from bot_engine.storage import save_json_file
            return save_json_file(filepath, data, description)
    
    async def request_exchange_optimized(self, method: str, endpoint: str,
                                        params: Optional[Dict] = None) -> Optional[Dict]:
        """Оптимизированный запрос к бирже"""
        if not self.exchange_client or not self.enabled:
            return None
        
        try:
            result = await self.exchange_client.request(method, endpoint, params)
            if result:
                self._stats['exchange_requests'] += 1
            return result
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка оптимизированного запроса: {e}")
            return None
    
    async def request_exchange_batch(self, requests: List[Dict[str, Any]],
                                     max_concurrent: int = 20) -> List[Optional[Dict]]:
        """Пакетные запросы к бирже"""
        if not self.exchange_client or not self.enabled:
            return []
        
        try:
            results = await self.exchange_client.request_batch(requests, max_concurrent)
            self._stats['exchange_requests'] += len(requests)
            return results
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка пакетных запросов: {e}")
            return []
    
    def calculate_rsi_optimized(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Оптимизированный расчет RSI"""
        if not OPT_CALC_AVAILABLE or not self.enabled:
            from bot_engine.utils.rsi_utils import calculate_rsi
            return calculate_rsi(prices, period)
        
        try:
            result = calculate_rsi_vectorized(prices, period)
            if result is not None:
                self._stats['calculations'] += 1
            return result
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка оптимизированного расчета RSI: {e}")
            from bot_engine.utils.rsi_utils import calculate_rsi
            return calculate_rsi(prices, period)
    
    def calculate_rsi_batch_optimized(self, prices_list: List[List[float]],
                                     period: int = 14, max_workers: int = None) -> List[Optional[float]]:
        """Пакетный расчет RSI для множества монет"""
        if not OPT_CALC_AVAILABLE or not self.enabled:
            from bot_engine.utils.rsi_utils import calculate_rsi
            return [calculate_rsi(prices, period) for prices in prices_list]
        
        try:
            results = calculate_rsi_batch(prices_list, period, max_workers)
            self._stats['calculations'] += len(prices_list)
            return results
        except Exception as e:
            logger.error(f"[PERF_OPT] Ошибка пакетного расчета RSI: {e}")
            from bot_engine.utils.rsi_utils import calculate_rsi
            return [calculate_rsi(prices, period) for prices in prices_list]
    
    async def flush_all_storage(self):
        """Принудительно сохраняет все ожидающие операции"""
        if ASYNC_STORAGE_AVAILABLE and self.enabled:
            await flush_all_pending()
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.exchange_client:
            await self.exchange_client.__aexit__(None, None, None)
            self.exchange_client = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику оптимизаций"""
        stats = dict(self._stats)
        
        if self.exchange_client:
            exchange_stats = self.exchange_client.get_stats()
            stats['exchange'] = exchange_stats
        
        return stats


# Глобальный экземпляр оптимизатора
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(enabled: bool = True) -> PerformanceOptimizer:
    """Получить глобальный экземпляр оптимизатора"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(enabled=enabled)
    return _performance_optimizer

