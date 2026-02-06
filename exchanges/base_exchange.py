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
    def get_closed_pnl(self, sort_by='time', period='all', start_date=None, end_date=None):
        """Получение закрытых позиций
        
        Args:
            sort_by: Способ сортировки ('time' или 'pnl')
            period: Период фильтрации ('all', 'day', 'week', 'month', 'half_year', 'year', 'custom')
            start_date: Начальная дата для custom периода (формат: 'YYYY-MM-DD' или timestamp в мс)
            end_date: Конечная дата для custom периода (формат: 'YYYY-MM-DD' или timestamp в мс)
        """
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
    def close_position(self, symbol, size, side, order_type="Limit"):
        """Закрытие позиции
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            size (float): Размер позиции для закрытия
            side (str): Сторона позиции ('Long' или 'Short')
            order_type (str): Тип ордера ('Limit' или 'Market')
            
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
    
    def get_instrument_status(self, symbol):
        """
        Получает статус торговли для символа (опциональный метод)
        
        Returns:
            dict: {'status': str, 'is_tradeable': bool, 'is_delisting': bool}
            Или None если метод не реализован на конкретной бирже
        """
        # По умолчанию возвращаем None - каждая биржа может переопределить
        return None

    def get_chart_data(self, symbol, timeframe='1h', period='1w'):
        """Получение данных для графика"""
        raise NotImplementedError

    def get_indicators(self, symbol, timeframe='1h'):
        """Получение значений индикаторов"""
        raise NotImplementedError
    
    @abstractmethod
    def place_order(self, symbol, side, quantity, order_type='market', price=None, leverage=None, **kwargs):
        """Размещение ордера
        
        Args:
            symbol (str): Символ торговой пары
            side (str): Сторона ('BUY' или 'SELL', для ботов также 'LONG'/'SHORT')
            quantity (float): Количество
            order_type (str): Тип ордера ('market' или 'limit')
            price (float, optional): Цена для лимитного ордера
            leverage (int, optional): Кредитное плечо (например, 5 для x5). Если указано, будет установлено перед входом в позицию.
            **kwargs: Дополнительные параметры
            
        Returns:
            dict: Результат размещения ордера
        """
        pass
    
    def cancel_order(self, symbol, order_id):
        """
        Отменяет ордер (опциональный метод, может быть переопределен в конкретных биржах)
        
        Args:
            symbol (str): Символ торговой пары
            order_id (str): ID ордера для отмены
            
        Returns:
            dict: Результат отмены ордера с полями:
                - success (bool): Успешность операции
                - message (str): Сообщение о результате
        """
        # По умолчанию возвращаем ошибку - каждая биржа должна переопределить
        return {'success': False, 'message': 'cancel_order not implemented for this exchange'}
    
    def get_open_orders(self, symbol):
        """
        Получает список открытых ордеров для символа (опциональный метод)
        
        Args:
            symbol (str): Символ торговой пары
            
        Returns:
            list: Список открытых ордеров, каждый ордер - dict с полями:
                - order_id (str): ID ордера
                - orderId (str): Альтернативное поле для ID
                - id (str): Еще одно альтернативное поле
                - symbol (str): Символ
                - side (str): Сторона ('Buy', 'Sell', 'LONG', 'SHORT')
                - price (float): Цена ордера
                - quantity (float): Количество
                - status (str): Статус ордера
            Или пустой список, если метод не реализован
        """
        # По умолчанию возвращаем пустой список - каждая биржа может переопределить
        return []
    
    def set_leverage(self, symbol, leverage):
        """
        Устанавливает кредитное плечо для символа (опциональный метод)
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            leverage (int): Значение плеча (например, 5 для x5)
            
        Returns:
            dict: Результат установки плеча с полями:
                - success (bool): Успешность операции
                - message (str): Сообщение о результате
        """
        # По умолчанию возвращаем ошибку - каждая биржа должна переопределить
        return {'success': False, 'message': 'set_leverage not implemented for this exchange'}