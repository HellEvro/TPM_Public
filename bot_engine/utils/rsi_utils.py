"""
RSI (Relative Strength Index) расчеты
Wilder's RSI алгоритм
"""

def calculate_rsi(prices, period=14):
    """Рассчитывает RSI на основе массива цен (Wilder's RSI алгоритм)"""
    if len(prices) < period + 1:
        return None
    
    # Рассчитываем изменения цен
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])
    
    if len(changes) < period:
        return None
    
    # Разделяем на прибыли и убытки
    gains = []
    losses = []
    
    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0) 
            losses.append(-change)
    
    # Первоначальные средние значения (простое среднее для первого периода)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Рассчитываем RSI используя сглаживание Wilder's
    # (это тип экспоненциального сглаживания)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Избегаем деления на ноль
    if avg_loss == 0:
        return 100.0
    
    # Рассчитываем RS и RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return round(rsi, 2)


def calculate_rsi_history(prices, period=14):
    """Рассчитывает полную историю RSI для анализа зрелости монеты"""
    if len(prices) < period + 1:
        return None
    
    # Рассчитываем изменения цен
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])
    
    if len(changes) < period:
        return None
    
    # Разделяем на прибыли и убытки
    gains = []
    losses = []
    
    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0) 
            losses.append(-change)
    
    # Первоначальные средние значения
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Рассчитываем полную историю RSI
    rsi_history = []
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        rsi_history.append(round(rsi, 2))
    
    return rsi_history


def estimate_price_for_rsi(closes, target_rsi, period=14, side='LONG'):
    """
    Оценка цены, при которой RSI будет равен target_rsi (при следующей свече с закрытием P).
    Используется для расчёта лимитной цены входа/выхода по RSI.
    closes: список цен закрытия (последняя — текущая свеча), минимум period+1 элемент.
    side: 'LONG' — вход в лонг (нужен низкий RSI, цена ниже текущей) или выход из шорта;
          'SHORT' — вход в шорт (высокий RSI, цена выше) или выход из лонга.
    Возвращает цену или None при недостатке данных.
    """
    if not closes or len(closes) < period + 1:
        return None
    try:
        target_rsi = float(target_rsi)
    except (TypeError, ValueError):
        return None
    current = float(closes[-1])
    if current <= 0:
        return None
    # Диапазон поиска: ±50% от текущей цены
    low = current * 0.5
    high = current * 1.5
    for _ in range(60):
        mid = (low + high) / 2.0
        series = [float(c) for c in closes] + [mid]
        rsi = calculate_rsi(series, period)
        if rsi is None:
            return None
        if side.upper() == 'LONG':
            # Низкий RSI → цена падает. Если RSI выше цели — снижаем mid
            if rsi > target_rsi:
                high = mid
            else:
                low = mid
        else:
            if rsi < target_rsi:
                low = mid
            else:
                high = mid
        if abs(rsi - target_rsi) < 0.3:
            break
    return round((low + high) / 2.0, 6)

