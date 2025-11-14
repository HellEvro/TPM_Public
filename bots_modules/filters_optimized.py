"""
Оптимизированная версия filters.py с использованием новых модулей оптимизации

Это пример интеграции - показывает как использовать оптимизации в существующем коде
"""

import asyncio
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('BotsService')

# Импорт оптимизаций
try:
    from bot_engine.performance_optimizer import get_performance_optimizer
    from bot_engine.optimized_calculations import calculate_rsi_batch
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    logger.warning("[FILTERS_OPT] Оптимизации недоступны, используется стандартная версия")


async def load_all_coins_rsi_optimized(exchange_obj=None):
    """
    Оптимизированная версия load_all_coins_rsi с использованием:
    - Асинхронных запросов к бирже
    - Пакетных расчетов RSI
    - Оптимизированного клиента биржи
    """
    if not OPTIMIZATIONS_AVAILABLE:
        # Fallback на стандартную версию
        from bots_modules.filters import load_all_coins_rsi
        return load_all_coins_rsi()
    
    try:
        from bots_modules.imports_and_globals import (
            coins_rsi_data, rsi_data_lock, get_exchange
        )
        
        # Проверяем флаг обновления
        if coins_rsi_data.get('update_in_progress', False):
            logger.info("Обновление RSI уже выполняется...")
            return False
        
        coins_rsi_data['update_in_progress'] = True
        
        # Получаем биржу
        current_exchange = exchange_obj or get_exchange()
        if not current_exchange:
            logger.error("[RSI_OPT] ❌ Биржа не инициализирована")
            coins_rsi_data['update_in_progress'] = False
            return False
        
        # Получаем список пар
        pairs = current_exchange.get_all_pairs()
        if not pairs:
            logger.error("[RSI_OPT] ❌ Не удалось получить список пар")
            coins_rsi_data['update_in_progress'] = False
            return False
        
        logger.info(f"[RSI_OPT] 🚀 Начинаем оптимизированную загрузку RSI для {len(pairs)} монет")
        
        # Инициализируем оптимизатор
        optimizer = get_performance_optimizer(enabled=True)
        
        # Получаем базовый URL биржи (пример для Bybit)
        base_url = getattr(current_exchange, 'base_url', 'https://api.bybit.com')
        
        # Инициализируем оптимизированный клиент биржи
        exchange_client = await optimizer.initialize_exchange_client(
            base_url=base_url,
            max_connections=100
        )
        
        if not exchange_client:
            logger.warning("[RSI_OPT] ⚠️ Не удалось инициализировать оптимизированный клиент, используем стандартный")
            from bots_modules.filters import load_all_coins_rsi
            return load_all_coins_rsi()
        
        # Создаем пакет запросов для свечей
        requests = []
        for symbol in pairs:
            requests.append({
                'method': 'GET',
                'endpoint': '/v5/market/kline',
                'params': {
                    'symbol': symbol,
                    'interval': '6',
                    'limit': 200
                },
                'use_cache': False  # Не кэшируем свечи
            })
        
        # Выполняем пакетные запросы
        logger.info(f"[RSI_OPT] 📡 Выполняем {len(requests)} запросов параллельно...")
        results = await optimizer.request_exchange_batch(requests, max_concurrent=20)
        
        # Обрабатываем результаты и рассчитываем RSI пакетно
        temp_coins_data = {}
        prices_list = []
        symbols_list = []
        
        for i, result in enumerate(results):
            if not result or not result.get('result'):
                continue
            
            symbol = pairs[i]
            candles_data = result.get('result', {}).get('list', [])
            
            if not candles_data:
                continue
            
            # Извлекаем цены закрытия
            closes = [float(candle[4]) for candle in reversed(candles_data)]  # [4] = close price
            
            if len(closes) >= 15:  # Минимум для RSI
                prices_list.append(closes)
                symbols_list.append(symbol)
        
        # Пакетный расчет RSI
        logger.info(f"[RSI_OPT] 📊 Рассчитываем RSI для {len(prices_list)} монет...")
        rsi_values = optimizer.calculate_rsi_batch_optimized(prices_list, period=14)
        
        # Формируем результаты
        for i, (symbol, rsi) in enumerate(zip(symbols_list, rsi_values)):
            if rsi is not None:
                temp_coins_data[symbol] = {
                    'symbol': symbol,
                    'rsi6h': rsi,
                    # Добавить другие поля по необходимости
                }
        
        # Атомарно обновляем данные
        with rsi_data_lock:
            coins_rsi_data['coins'] = temp_coins_data
            coins_rsi_data['successful_coins'] = len(temp_coins_data)
            coins_rsi_data['failed_coins'] = len(pairs) - len(temp_coins_data)
            coins_rsi_data['last_update'] = datetime.now().isoformat()
            coins_rsi_data['update_in_progress'] = False
        
        logger.info(f"[RSI_OPT] ✅ Загружено {len(temp_coins_data)} монет")
        
        # Очистка
        await optimizer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"[RSI_OPT] ❌ Ошибка оптимизированной загрузки: {e}")
        import traceback
        logger.error(f"[RSI_OPT] Traceback: {traceback.format_exc()}")
        
        # Fallback на стандартную версию
        from bots_modules.filters import load_all_coins_rsi
        return load_all_coins_rsi()


# Пример использования в существующем коде:
# В continuous_data_loader.py можно заменить:
# from bots_modules.filters import load_all_coins_rsi
# на:
# from bots_modules.filters_optimized import load_all_coins_rsi_optimized as load_all_coins_rsi

