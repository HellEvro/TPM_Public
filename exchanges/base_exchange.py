from abc import ABC, abstractmethod
import time
import signal
from functools import wraps

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")

def with_timeout(timeout_seconds=30):
    """Декоратор для добавления таймаута к методам"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Устанавливаем таймаут только для Unix систем
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Для Windows используем простую проверку времени
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"API call took {elapsed:.2f}s, exceeded timeout of {timeout_seconds}s")
                return result
        return wrapper
    return decorator

class BaseExchange(ABC):
    def __init__(self, api_key, api_secret, position_mode='Hedge', limit_order_offset=0.01):
        self.api_key = api_key
        self.api_secret = api_secret
        self.position_mode = position_mode
        self.limit_order_offset = limit_order_offset
        self.max_profit_values = {}
        self.max_loss_values = {}
        self.daily_pnl = {}
        self.api_timeout = 30  # Таймаут по умолчанию для API вызовов

    @abstractmethod
    def get_positions(self):
        """Получение активных позиций"""
        pass

    @abstractmethod
    def get_closed_pnl(self, sort_by='time'):
        """Получение закрытых позиций"""
        pass

    @abstractmethod
    def get_symbol_chart_data(self, symbol):
        """Получение данных для графика"""
        pass

    @abstractmethod
    def get_sma200_position(self, symbol):
        """Получение положения цены относительно SMA200"""
        pass

    @abstractmethod
    def get_ticker(self, symbol):
        """Получение текущих данных тикера"""
        pass

    @abstractmethod
    def close_position(self, symbol, size, side):
        """Закрытие позиции
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            size (float): Размер позиции для закрытия
            side (str): Сторона позиции ('Long' или 'Short')
            
        Returns:
            dict: Результат закрытия позиции с полями:
                - success (bool): Успешность операции
                - message (str): Сообщение о результате
                - order_id (str, optional): ID созданного ордера
                - close_price (float, optional): Цена закрытия
        """
        pass

    @abstractmethod
    def get_all_pairs(self):
        """Получение списка всех доступных бессрочных фьючерсов"""
        pass

    def get_chart_data(self, symbol, timeframe='1h', period='1w'):
        """Получение данных для графика"""
        raise NotImplementedError

    def get_indicators(self, symbol, timeframe='1h'):
        """Получение значений индикаторов"""
        raise NotImplementedError
    
    @abstractmethod
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        """Размещение ордера
        
        Args:
            symbol (str): Символ торговой пары
            side (str): Сторона ('BUY' или 'SELL', для ботов также 'LONG'/'SHORT')
            quantity (float): Количество
            order_type (str): Тип ордера ('market' или 'limit')
            price (float, optional): Цена для лимитного ордера
            
        Returns:
            dict: Результат размещения ордера
        """
        pass