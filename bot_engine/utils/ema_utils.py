"""
EMA (Exponential Moving Average) расчеты и анализ тренда
"""

import logging

logger = logging.getLogger('EMA_Utils')

# Константы для анализа тренда
TREND_CONFIRMATION_BARS = 3


def calculate_ema(prices, period):
    """Рассчитывает EMA для массива цен"""
    if len(prices) < period:
        return None
    
    # Первое значение EMA = SMA
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)
    
    # Рассчитываем EMA для остальных значений
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def analyze_trend_6h(symbol, exchange_obj, get_optimal_ema_periods_func):
    """
    Анализирует тренд 6H с использованием оптимальных EMA периодов
    
    Args:
        symbol: Символ монеты
        exchange_obj: Объект биржи
        get_optimal_ema_periods_func: Функция для получения оптимальных EMA периодов
    """
    try:
        # Получаем оптимальные EMA периоды для монеты
        ema_periods = get_optimal_ema_periods_func(symbol)
        ema_short_period = ema_periods['ema_short']
        ema_long_period = ema_periods['ema_long']
        
        # Получаем свечи 6H для анализа тренда
        if not exchange_obj:
            logger.error(f"[TREND] ❌ Биржа не доступна для анализа тренда {symbol}")
            return None
            
        chart_response = exchange_obj.get_chart_data(symbol, '6h', '60d')
        
        if not chart_response or not chart_response.get('success'):
            return None
        
        candles = chart_response['data']['candles']
        min_candles = max(ema_long_period + 50, 210)
        if not candles or len(candles) < min_candles:
            return None
        
        # Извлекаем цены закрытия
        closes = [candle['close'] for candle in candles]
        
        # Рассчитываем оптимальные EMA
        ema_short = calculate_ema(closes, ema_short_period)
        ema_long = calculate_ema(closes, ema_long_period)
        
        if ema_short is None or ema_long is None:
            return None
        
        current_close = closes[-1]
        
        # Проверяем наклон длинной EMA
        if len(closes) >= ema_long_period + 1:
            prev_ema_long = calculate_ema(closes[:-1], ema_long_period)
            ema_long_slope = ema_long - prev_ema_long if prev_ema_long else 0
        else:
            ema_long_slope = 0
        
        # Проверяем минимум 3 закрытия подряд относительно длинной EMA
        recent_closes = closes[-TREND_CONFIRMATION_BARS:]
        all_above_ema_long = all(close > ema_long for close in recent_closes)
        all_below_ema_long = all(close < ema_long for close in recent_closes)
        
        # Определяем тренд
        trend = 'NEUTRAL'
        
        # UP: Close > EMA_long, EMA_short > EMA_long, наклон > 0, 3 закрытия > EMA_long
        if (current_close > ema_long and 
            ema_short > ema_long and 
            ema_long_slope > 0 and 
            all_above_ema_long):
            trend = 'UP'
        
        # DOWN: Close < EMA_long, EMA_short < EMA_long, наклон < 0, 3 закрытия < EMA_long
        elif (current_close < ema_long and 
              ema_short < ema_long and 
              ema_long_slope < 0 and 
              all_below_ema_long):
            trend = 'DOWN'
        
        return {
            'trend': trend,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period,
            'ema_long_slope': ema_long_slope,
            'current_close': current_close,
            'confirmations': TREND_CONFIRMATION_BARS,
            'accuracy': ema_periods['accuracy']
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка анализа тренда для {symbol}: {e}")
        return None

