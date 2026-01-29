"""
Проверка зрелости монет для торговли
Монета считается зрелой если:
1. Имеет достаточную историю свечей
2. RSI достигал экстремальных значений (полный цикл)
"""

import logging
import time
import threading

logger = logging.getLogger('MaturityChecker')

# Дефолтные константы
MIN_CANDLES_FOR_MATURITY = 200
MIN_RSI_LOW = 35
MAX_RSI_HIGH = 65

# Хранилище зрелых монет
mature_coins_storage = {}
mature_coins_lock = threading.Lock()


def check_coin_maturity(symbol, candles, config, calculate_rsi_history_func=None):
    """
    Проверяет зрелость монеты для торговли
    
    Args:
        symbol: Символ монеты
        candles: Список свечей
        config: Конфигурация с параметрами зрелости
        calculate_rsi_history_func: Функция для расчета RSI истории
    """
    try:
        # Импортируем функцию если не передана
        if calculate_rsi_history_func is None:
            from .utils.rsi_utils import calculate_rsi_history
            calculate_rsi_history_func = calculate_rsi_history
        
        min_candles = config.get('min_candles_for_maturity', MIN_CANDLES_FOR_MATURITY)
        min_rsi_low = config.get('min_rsi_low', MIN_RSI_LOW)
        max_rsi_high = config.get('max_rsi_high', MAX_RSI_HIGH)
        
        if not candles or len(candles) < min_candles:
            return {
                'is_mature': False,
                'reason': f'Недостаточно свечей: {len(candles) if candles else 0}/{min_candles}',
                'details': {
                    'candles_count': len(candles) if candles else 0,
                    'min_required': min_candles
                }
            }
        
        # Берем только последние N свечей для анализа
        recent_candles = candles[-min_candles:] if len(candles) >= min_candles else candles
        
        # Извлекаем цены закрытия
        closes = [candle['close'] for candle in recent_candles]
        
        # Рассчитываем историю RSI
        rsi_history = calculate_rsi_history_func(closes, 14)
        if not rsi_history:
            return {
                'is_mature': False,
                'reason': 'Не удалось рассчитать историю RSI',
                'details': {}
            }
        
        # Анализируем диапазон RSI
        rsi_min = min(rsi_history)
        rsi_max = max(rsi_history)
        rsi_range = rsi_max - rsi_min
        
        # Проверяем критерии зрелости
        maturity_checks = {
            'sufficient_candles': len(candles) >= min_candles,
            'rsi_reached_low': rsi_min <= min_rsi_low,
            'rsi_reached_high': rsi_max >= max_rsi_high
        }
        
        # Монета зрелая, если достаточно свечей И RSI достигал низких И высоких значений
        is_mature = (maturity_checks['sufficient_candles'] and 
                    maturity_checks['rsi_reached_low'] and 
                    maturity_checks['rsi_reached_high'])
        
        details = {
            'candles_count': len(candles),
            'min_required': min_candles,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'rsi_range': rsi_range,
            'checks': maturity_checks
        }
        
        if not is_mature:
            failed_checks = [check for check, passed in maturity_checks.items() if not passed]
            reason = f'Не пройдены проверки: {", ".join(failed_checks)}'
        else:
            reason = 'Монета зрелая для торговли'
        
        return {
            'is_mature': is_mature,
            'reason': reason,
            'details': details
        }
        
    except Exception as e:
        logger.error(f"[MATURITY] Ошибка проверки зрелости {symbol}: {e}")
        return {
            'is_mature': False,
            'reason': f'Ошибка анализа: {str(e)}',
            'details': {}
        }


def is_coin_mature_stored(symbol):
    """Проверяет, есть ли монета в хранилище зрелых монет"""
    with mature_coins_lock:
        return symbol in mature_coins_storage


def add_mature_coin_to_storage(symbol, maturity_data, auto_save=True, save_func=None):
    """Добавляет монету в хранилище зрелых монет"""
    global mature_coins_storage
    
    with mature_coins_lock:
        if symbol in mature_coins_storage:
            mature_coins_storage[symbol]['last_verified'] = time.time()
            pass
            return
        
        mature_coins_storage[symbol] = {
            'timestamp': time.time(),
            'maturity_data': maturity_data,
            'last_verified': time.time()
        }
    
    if auto_save and save_func:
        save_func()
        logger.info(f"[MATURITY_STORAGE] Монета {symbol} добавлена в хранилище")
    else:
        pass


def remove_mature_coin_from_storage(symbol):
    """Удаляет монету из хранилища зрелых монет"""
    global mature_coins_storage
    if symbol in mature_coins_storage:
        del mature_coins_storage[symbol]
        pass


def update_mature_coin_verification(symbol):
    """Обновляет время последней проверки зрелости"""
    global mature_coins_storage
    if symbol in mature_coins_storage:
        mature_coins_storage[symbol]['last_verified'] = time.time()
        pass


def check_coin_maturity_with_storage(symbol, candles, config, save_func=None, calculate_rsi_history_func=None):
    """Проверяет зрелость с использованием хранилища"""
    # Проверяем хранилище
    if is_coin_mature_stored(symbol):
        pass
        update_mature_coin_verification(symbol)
        return {
            'is_mature': True,
            'reason': 'Монета зрелая (из хранилища)',
            'details': {'stored': True, 'from_storage': True}
        }
    
    # Если нет в хранилище, выполняем полную проверку
    maturity_result = check_coin_maturity(symbol, candles, config, calculate_rsi_history_func)
    
    # Если монета зрелая, добавляем в хранилище
    if maturity_result['is_mature']:
        add_mature_coin_to_storage(symbol, maturity_result, auto_save=False, save_func=save_func)
    
    return maturity_result


def check_coin_maturity_stored_or_verify(symbol, exchange_obj, ensure_exchange_func, config):
    """Проверяет зрелость из хранилища или выполняет проверку"""
    try:
        # Проверяем хранилище
        if is_coin_mature_stored(symbol):
            return True
        
        # Если нет в хранилище, выполняем проверку
        if not ensure_exchange_func():
            logger.warning(f"[MATURITY_CHECK] {symbol}: Биржа не инициализирована")
            return False
        
        chart_response = exchange_obj.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.warning(f"[MATURITY_CHECK] {symbol}: Не удалось получить свечи")
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if not candles:
            logger.warning(f"[MATURITY_CHECK] {symbol}: Нет свечей")
            return False
        
        maturity_result = check_coin_maturity_with_storage(symbol, candles, config)
        return maturity_result['is_mature']
        
    except Exception as e:
        logger.error(f"[MATURITY_CHECK] {symbol}: Ошибка проверки зрелости: {e}")
        return False


def get_mature_coins_storage():
    """Возвращает копию хранилища зрелых монет"""
    with mature_coins_lock:
        return mature_coins_storage.copy()


def set_mature_coins_storage(new_storage):
    """Устанавливает новое хранилище зрелых монет"""
    global mature_coins_storage
    with mature_coins_lock:
        mature_coins_storage = new_storage


def clear_mature_coins_storage():
    """Очищает хранилище зрелых монет"""
    global mature_coins_storage
    with mature_coins_lock:
        mature_coins_storage = {}
    logger.info("[MATURITY_STORAGE] Хранилище зрелых монет очищено")

