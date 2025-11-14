"""
Оптимизированный клиент для сетевых запросов к бирже

Оптимизации:
- Connection pooling с переиспользованием соединений
- Асинхронные запросы через aiohttp
- Батчинг запросов для уменьшения количества соединений
- Кэширование ответов
- Rate limiting для защиты от блокировок
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger('OptimizedExchangeClient')


class OptimizedExchangeClient:
    """Оптимизированный клиент для запросов к бирже"""
    
    def __init__(self, base_url: str, max_connections: int = 100, 
                 max_connections_per_host: int = 30, timeout: int = 30):
        """
        Args:
            base_url: Базовый URL биржи
            max_connections: Максимальное количество соединений в пуле
            max_connections_per_host: Максимальное количество соединений на хост
            timeout: Таймаут запросов в секундах
        """
        self.base_url = base_url
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Connection pool
        self.connector = None
        self.session = None
        
        # Кэш ответов
        self._cache: Dict[str, tuple] = {}  # {key: (data, timestamp)}
        self._cache_ttl = 5.0  # Секунды
        
        # Rate limiting
        self._request_times: List[float] = []
        self._max_requests_per_second = 10
        self._rate_limit_lock = asyncio.Lock()
        
        # Статистика
        self._stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'failed_requests': 0,
            'total_time': 0.0
        }
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ttl_dns_cache=300,
            force_close=False,  # Переиспользование соединений
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'InfoBot/1.0',
                'Accept': 'application/json'
            }
        )
        
        logger.info(f"[OPT_EXCHANGE] Инициализирован клиент: {self.max_connections} соединений, "
                   f"{self.max_connections_per_host} на хост")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        logger.info("[OPT_EXCHANGE] Клиент закрыт")
    
    async def _rate_limit(self):
        """Rate limiting для защиты от блокировок"""
        async with self._rate_limit_lock:
            current_time = time.time()
            
            # Удаляем старые записи (старше 1 секунды)
            self._request_times = [t for t in self._request_times if current_time - t < 1.0]
            
            # Если превышен лимит - ждем
            if len(self._request_times) >= self._max_requests_per_second:
                sleep_time = 1.0 - (current_time - self._request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    current_time = time.time()
            
            self._request_times.append(current_time)
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Генерирует ключ кэша"""
        sorted_params = sorted(params.items())
        return f"{endpoint}:{sorted_params}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Получает данные из кэша если они актуальны"""
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                self._stats['cached_requests'] += 1
                return data
            else:
                del self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """Сохраняет данные в кэш"""
        self._cache[cache_key] = (data, time.time())
        
        # Очистка старого кэша (если больше 1000 записей)
        if len(self._cache) > 1000:
            current_time = time.time()
            self._cache = {
                k: v for k, v in self._cache.items()
                if current_time - v[1] < self._cache_ttl
            }
    
    async def request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                     use_cache: bool = True) -> Optional[Dict]:
        """
        Выполняет HTTP запрос к бирже
        
        Args:
            method: HTTP метод (GET, POST, etc.)
            endpoint: Endpoint API
            params: Параметры запроса
            use_cache: Использовать ли кэш
        
        Returns:
            Dict с данными ответа или None при ошибке
        """
        if not self.session:
            logger.error("[OPT_EXCHANGE] Сессия не инициализирована")
            return None
        
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params) if use_cache else None
        
        # Проверяем кэш
        if use_cache and cache_key:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Rate limiting
        await self._rate_limit()
        
        # Выполняем запрос
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, params=params) as response:
                self._stats['total_requests'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Сохраняем в кэш
                    if use_cache and cache_key:
                        self._set_cache(cache_key, data)
                    
                    elapsed = time.time() - start_time
                    self._stats['total_time'] += elapsed
                    
                    return data
                else:
                    logger.warning(f"[OPT_EXCHANGE] HTTP {response.status} для {endpoint}")
                    self._stats['failed_requests'] += 1
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"[OPT_EXCHANGE] Timeout для {endpoint}")
            self._stats['failed_requests'] += 1
            return None
        except Exception as e:
            logger.error(f"[OPT_EXCHANGE] Ошибка запроса {endpoint}: {e}")
            self._stats['failed_requests'] += 1
            return None
    
    async def request_batch(self, requests: List[Dict[str, Any]], 
                           max_concurrent: int = 20) -> List[Optional[Dict]]:
        """
        Выполняет пакет запросов параллельно
        
        Args:
            requests: Список запросов [{'method': 'GET', 'endpoint': '/api/...', 'params': {...}}, ...]
            max_concurrent: Максимальное количество одновременных запросов
        
        Returns:
            Список результатов в том же порядке
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _request_with_semaphore(req):
            async with semaphore:
                return await self.request(
                    req.get('method', 'GET'),
                    req.get('endpoint', ''),
                    req.get('params'),
                    req.get('use_cache', True)
                )
        
        tasks = [_request_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем исключения
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[OPT_EXCHANGE] Ошибка в батч-запросе {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        avg_time = (self._stats['total_time'] / self._stats['total_requests'] 
                   if self._stats['total_requests'] > 0 else 0)
        
        return {
            'total_requests': self._stats['total_requests'],
            'cached_requests': self._stats['cached_requests'],
            'failed_requests': self._stats['failed_requests'],
            'cache_hit_rate': (self._stats['cached_requests'] / self._stats['total_requests'] * 100
                             if self._stats['total_requests'] > 0 else 0),
            'avg_request_time': avg_time,
            'cache_size': len(self._cache)
        }

