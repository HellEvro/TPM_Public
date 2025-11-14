"""
Оптимизированные расчеты для множества монет

Оптимизации:
- Векторные операции через NumPy (если доступен)
- Параллельные расчеты через ThreadPoolExecutor
- Батчинг операций для лучшей производительности
- Кэширование промежуточных результатов
"""

import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger('OptimizedCalculations')

# Пытаемся импортировать NumPy для векторных операций
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("[OPT_CALC] NumPy недоступен, используются стандартные операции")

# Импорт базовых функций
try:
    from bot_engine.utils.rsi_utils import calculate_rsi, calculate_rsi_history, calculate_ema
except ImportError:
    logger.warning("[OPT_CALC] Не удалось импортировать базовые функции расчета")


def calculate_rsi_batch(prices_list: List[List[float]], period: int = 14, 
                       max_workers: int = None) -> List[Optional[float]]:
    """
    Рассчитывает RSI для пакета монет параллельно
    
    Args:
        prices_list: Список списков цен для каждой монеты
        period: Период RSI
        max_workers: Максимальное количество потоков (None = автоматически)
    
    Returns:
        Список значений RSI в том же порядке
    """
    if not prices_list:
        return []
    
    max_workers = max_workers or min(len(prices_list), 20)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_rsi, prices, period): i
            for i, prices in enumerate(prices_list)
        }
        
        results = [None] * len(prices_list)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"[OPT_CALC] Ошибка расчета RSI для индекса {idx}: {e}")
                results[idx] = None
    
    return results


def calculate_rsi_batch_dict(prices_dict: Dict[str, List[float]], period: int = 14,
                            max_workers: int = None) -> Dict[str, Optional[float]]:
    """
    Рассчитывает RSI для словаря монет параллельно
    
    Args:
        prices_dict: Словарь {symbol: [prices]}
        period: Период RSI
        max_workers: Максимальное количество потоков
    
    Returns:
        Словарь {symbol: rsi_value}
    """
    if not prices_dict:
        return {}
    
    symbols = list(prices_dict.keys())
    prices_list = [prices_dict[symbol] for symbol in symbols]
    
    # Используем пакетный расчет
    rsi_list = calculate_rsi_batch(prices_list, period, max_workers)
    
    # Возвращаем словарь
    return {symbol: rsi for symbol, rsi in zip(symbols, rsi_list)}


def calculate_ema_batch(prices_list: List[List[float]], period: int,
                       return_list: bool = False, max_workers: int = None) -> List:
    """
    Рассчитывает EMA для пакета монет параллельно
    
    Args:
        prices_list: Список списков цен для каждой монеты
        period: Период EMA
        return_list: Возвращать ли список значений или одно значение
        max_workers: Максимальное количество потоков
    
    Returns:
        Список значений EMA в том же порядке
    """
    if not prices_list:
        return []
    
    max_workers = max_workers or min(len(prices_list), 20)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_ema, prices, period, return_list): i
            for i, prices in enumerate(prices_list)
        }
        
        results = [None] * len(prices_list)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"[OPT_CALC] Ошибка расчета EMA для индекса {idx}: {e}")
                results[idx] = None if not return_list else []
    
    return results


def calculate_rsi_vectorized(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Векторизованный расчет RSI через NumPy (быстрее для больших массивов)
    
    Args:
        prices: Список цен
        period: Период RSI
    
    Returns:
        Значение RSI или None
    """
    if not NUMPY_AVAILABLE:
        # Fallback на стандартную реализацию
        return calculate_rsi(prices, period)
    
    try:
        prices_array = np.array(prices, dtype=np.float64)
        
        if len(prices_array) < period + 1:
            return None
        
        # Рассчитываем изменения цен
        deltas = np.diff(prices_array)
        
        if len(deltas) < period:
            return None
        
        # Разделяем на прибыли и убытки
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Первоначальные средние значения
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Рассчитываем RSI используя сглаживание Wilder's
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Избегаем деления на ноль
        if avg_loss == 0:
            return 100.0
        
        # Рассчитываем RS и RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return round(float(rsi), 2)
        
    except Exception as e:
        logger.error(f"[OPT_CALC] Ошибка векторного расчета RSI: {e}")
        return calculate_rsi(prices, period)


def calculate_ema_vectorized(prices: List[float], period: int) -> Optional[float]:
    """
    Векторизованный расчет EMA через NumPy
    
    Args:
        prices: Список цен
        period: Период EMA
    
    Returns:
        Значение EMA или None
    """
    if not NUMPY_AVAILABLE:
        return calculate_ema(prices, period)
    
    try:
        prices_array = np.array(prices, dtype=np.float64)
        
        if len(prices_array) < period:
            return None
        
        # Первое значение EMA = SMA
        ema = np.mean(prices_array[:period])
        multiplier = 2.0 / (period + 1)
        
        # Рассчитываем EMA для остальных значений
        for price in prices_array[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
        
    except Exception as e:
        logger.error(f"[OPT_CALC] Ошибка векторного расчета EMA: {e}")
        return calculate_ema(prices, period)


def process_coins_batch(coins_data: List[Dict], calculation_func: callable,
                        max_workers: int = None, batch_size: int = 50) -> List[Dict]:
    """
    Обрабатывает пакет монет с расчетами параллельно
    
    Args:
        coins_data: Список данных монет
        calculation_func: Функция расчета (принимает coin_data, возвращает обновленный coin_data)
        max_workers: Максимальное количество потоков
        batch_size: Размер батча для обработки
    
    Returns:
        Список обработанных данных монет
    """
    if not coins_data:
        return []
    
    max_workers = max_workers or min(batch_size, 20)
    results = []
    
    # Обрабатываем батчами
    for i in range(0, len(coins_data), batch_size):
        batch = coins_data[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(calculation_func, coin): coin.get('symbol', f'coin_{i+j}')
                for j, coin in enumerate(batch)
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"[OPT_CALC] Ошибка обработки {symbol}: {e}")
    
    return results

