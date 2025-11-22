"""
Модуль для вычисления технических индикаторов
"""

def calculate_rsi(prices, period=14):
    """Рассчитывает RSI на основе массива цен (Wilder's RSI алгоритм)"""
    if len(prices) < period + 1:
        return None
    
    # Рассчитываем изменения цен
    deltas = []
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i-1])
    
    # Разделяем на положительные и отрицательные изменения
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    # Рассчитываем средние значения (Wilder's smoothing)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Рассчитываем RSI для первого значения
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_ema(prices, period):
    """Вычисляет EMA для заданного периода"""
    if len(prices) < period:
        return []
    
    ema_values = []
    multiplier = 2 / (period + 1)
    
    # Первое значение EMA = SMA
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    # Остальные значения EMA
    for i in range(period, len(prices)):
        ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values

def calculate_sma(prices, period):
    """Вычисляет SMA для заданного периода"""
    if len(prices) < period:
        return []
    
    sma_values = []
    for i in range(period - 1, len(prices)):
        sma = sum(prices[i - period + 1:i + 1]) / period
        sma_values.append(sma)
    
    return sma_values
