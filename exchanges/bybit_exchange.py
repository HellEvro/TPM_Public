from pybit.unified_trading import HTTP
from .base_exchange import BaseExchange, with_timeout
from http.client import IncompleteRead, RemoteDisconnected
import requests.exceptions
import requests
import time
import math
from datetime import datetime, timedelta
import sys
try:
    from app.config import (  # type: ignore
        GROWTH_MULTIPLIER,
        HIGH_ROI_THRESHOLD,
        HIGH_LOSS_THRESHOLD
    )
except ImportError:  # pragma: no cover - fallback для статического анализа
    GROWTH_MULTIPLIER = 3.0
    HIGH_ROI_THRESHOLD = 100.0
    HIGH_LOSS_THRESHOLD = -40.0
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Глобальная настройка пула соединений для всех HTTP запросов
def setup_global_connection_pool():
    """Настраивает глобальный пул соединений для всех HTTP запросов"""
    try:
        import urllib3
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Увеличиваем лимиты пула соединений для urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Настраиваем глобальные лимиты пула соединений
        import urllib3.poolmanager
        urllib3.poolmanager.PoolManager.DEFAULT_POOLBLOCK = False
        urllib3.poolmanager.PoolManager.DEFAULT_POOLSIZE = 100
        urllib3.poolmanager.PoolManager.DEFAULT_MAXSIZE = 100
        
        # Создаем сессию с увеличенным пулом соединений
        session = requests.Session()
        
        # Настраиваем адаптер с большим пулом соединений
        adapter = HTTPAdapter(
            pool_connections=100,  # Увеличиваем количество пулов соединений
            pool_maxsize=200,      # Увеличиваем максимальное количество соединений в пуле
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        
        # Устанавливаем адаптер для HTTP и HTTPS
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Устанавливаем глобальную сессию для requests
        requests.Session = lambda: session
        
        # Убрано: logger.debug("✅ Глобальный пул соединений настроен: 100 пулов, 200 соединений на пул") - слишком шумно
        
    except Exception as e:
        logger.warning(f"⚠️ Не удалось настроить глобальный пул соединений: {e}")

# Настраиваем пул соединений при импорте модуля
setup_global_connection_pool()

# Устанавливаем кодировку для stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def clean_symbol(symbol):
    """Удаляет 'USDT' из названия символа"""
    return symbol.replace('USDT', '')

class BybitExchange(BaseExchange):
    def __init__(self, api_key, api_secret, test_server=False, position_mode='Hedge', limit_order_offset=0.1):
        super().__init__(api_key, api_secret)
        
        # Настраиваем пул соединений для requests и pybit
        self._setup_connection_pool()
        
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=test_server,
            timeout=30,
            recv_window=20000
        )
        self.position_mode = position_mode
        self.limit_order_offset = limit_order_offset  # Отсутп цены для лимитного ордера в процентах
        self.daily_pnl = {}
        self.last_reset_day = None
        self.max_profit_values = {}
        self.max_loss_values = {}
        
        # Управление задержкой между запросами для предотвращения rate limit
        self.base_request_delay = 0.5  # Базовая задержка между запросами (500ms) - увеличено для стабильности
        self.current_request_delay = 0.5  # Текущая задержка (может увеличиваться при rate limit)
        self.max_request_delay = 10.0  # Максимальная задержка для предотвращения таймаутов (увеличено)
        self.rate_limit_error_count = 0  # Счетчик ошибок rate limit для агрессивного увеличения задержки
        self.last_rate_limit_time = 0  # Время последней ошибки rate limit
        
        # Кэш для баланса кошелька (чтобы не спамить запросами при проблемах с сетью)
        self._wallet_balance_cache = None
        self._wallet_balance_cache_time = 0
        self._wallet_balance_cache_ttl = 30  # Кэш на 30 секунд при успешных запросах
        self._wallet_balance_cache_ttl_error = 300  # Кэш на 5 минут при сетевых ошибках
        self._last_network_error_time = 0
        self._network_error_count = 0
        
        # Кэш для режима позиции (чтобы не спамить запросами)
        self._position_mode_cache = None
        self._position_mode_cache_time = 0
        self._position_mode_cache_ttl = 300  # Кэш на 5 минут (режим позиции меняется редко)
    
    def _setup_connection_pool(self):
        """Настраивает пул соединений для requests и pybit"""
        try:
            import urllib3
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Увеличиваем лимиты пула соединений для urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Настраиваем глобальные лимиты пула соединений
            import urllib3.poolmanager
            urllib3.poolmanager.PoolManager.DEFAULT_POOLBLOCK = False
            urllib3.poolmanager.PoolManager.DEFAULT_POOLSIZE = 100
            urllib3.poolmanager.PoolManager.DEFAULT_MAXSIZE = 100
            
            # Создаем сессию с увеличенным пулом соединений
            session = requests.Session()
            
            # Настраиваем адаптер с большим пулом соединений
            adapter = HTTPAdapter(
                pool_connections=100,  # Увеличиваем количество пулов соединений
                pool_maxsize=200,      # Увеличиваем максимальное количество соединений в пуле
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.3,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            
            # Устанавливаем адаптер для HTTP и HTTPS
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Устанавливаем глобальную сессию для requests
            requests.Session = lambda: session
            
            logging.info("✅ Пул соединений настроен: 100 пулов, 200 соединений на пул")
            
        except Exception as e:
            logging.warning(f"⚠️ Не удалось настроить пул соединений: {e}")

    def reset_request_delay(self):
        """Сбрасывает текущую задержку запросов к базовому значению"""
        if self.current_request_delay != self.base_request_delay:
            logger.info(f"🔄 Сброс задержки запросов: {self.current_request_delay:.3f}с → {self.base_request_delay:.3f}с")
            self.current_request_delay = self.base_request_delay
            # Сбрасываем счетчик ошибок только после успешных запросов
            if time.time() - self.last_rate_limit_time > 30:
                self.rate_limit_error_count = 0

    def increase_request_delay(self, multiplier=2.0, reason='Rate limit'):
        """Увеличивает задержку запросов с учетом максимального порога"""
        current_time = time.time()
        
        # Если прошло больше 60 секунд с последней ошибки - сбрасываем счетчик
        if current_time - self.last_rate_limit_time > 60:
            self.rate_limit_error_count = 0
        
        self.rate_limit_error_count += 1
        self.last_rate_limit_time = current_time
        
        # Более агрессивное увеличение при множественных ошибках
        if self.rate_limit_error_count >= 3:
            multiplier = 3.0  # Увеличиваем множитель при частых ошибках
        elif self.rate_limit_error_count >= 5:
            multiplier = 5.0  # Еще более агрессивное увеличение
        
        old_delay = self.current_request_delay
        new_delay = min(self.current_request_delay * multiplier, self.max_request_delay)
        self.current_request_delay = new_delay

        if new_delay > old_delay:
            logger.warning(f"⚠️ {reason}. Увеличиваем задержку: {old_delay:.3f}с → {new_delay:.3f}с (ошибок подряд: {self.rate_limit_error_count})")
        else:
            logger.warning(f"⚠️ {reason}. Задержка уже максимальная: {new_delay:.3f}с (ошибок подряд: {self.rate_limit_error_count})")

        return new_delay
    
    def reset_daily_pnl(self, positions):
        """Сброс значений PnL в 00:00"""
        self.daily_pnl = {}
        for position in positions:
            symbol = clean_symbol(position['symbol'])
            self.daily_pnl[symbol] = float(position['unrealisedPnl'])
        self.last_reset_day = datetime.now().date()

    def get_positions(self):
        try:
            retries = 3
            retry_delay = 5
            
            for attempt in range(retries):
                try:
                    all_positions = []
                    cursor = None
                    rapid_growth_positions = []
                    
                    while True:
                        params = {
                            "category": "linear",
                            "settleCoin": "USDT",
                            "limit": 100
                        }
                        if cursor:
                            params["cursor"] = cursor
                        
                        try:
                            response = self.client.get_positions(**params)
                            positions = response['result']['list']
                            
                            active_positions = [p for p in positions if abs(float(p['size'])) > 0]
                            all_positions.extend(active_positions)
                            
                            cursor = response['result'].get('nextPageCursor')
                            if not cursor:
                                break
                                
                        except (ConnectionError, IncompleteRead, RemoteDisconnected, requests.exceptions.ConnectionError) as e:
                            logger.error("Connection error on attempt {}: {}".format(attempt + 1, str(e)))
                            if attempt < retries - 1:
                                time.sleep(retry_delay)
                                continue
                            raise
                    
                    if not all_positions:
                        # Нет активных позиций - это нормально, не логируем
                        return [], []

                    if self.last_reset_day is None or datetime.now().date() != self.last_reset_day:
                        self.reset_daily_pnl(all_positions)
                    
                    processed_positions = []
                    for position in all_positions:
                        symbol = clean_symbol(position['symbol'])
                        current_pnl = float(position['unrealisedPnl'])
                        position_size = abs(float(position['size']))
                        avg_price = float(position.get('avgPrice', 0) or 0)
                        leverage = float(position.get('leverage', 1) or 1)
                        
                        # ROI рассчитывается от ИЗНАЧАЛЬНОЙ маржи (залога), которую вложили при входе
                        # В Bybit API v5:
                        # - positionValue = стоимость позиции в USDT (размер * текущая цена)
                        # - positionIM = текущая изолированная маржа (может меняться из-за изменения цены)
                        # - leverage = плечо
                        # 
                        # ИЗНАЧАЛЬНАЯ маржа = positionValue / leverage (стоимость позиции / плечо)
                        # Это маржа, которую вложили при открытии позиции
                        
                        # Рассчитываем изначальную маржу
                        # В Bybit API positionValue может содержать либо стоимость позиции, либо маржу
                        # Сначала рассчитываем маржу из размера и цены входа
                        
                        position_value_calc = avg_price * position_size  # Стоимость позиции из размера и цены
                        
                        # ИЗНАЧАЛЬНАЯ маржа = стоимость позиции / плечо
                        if leverage > 0:
                            margin = position_value_calc / leverage
                        else:
                            margin = position_value_calc
                        
                        # Проверяем positionValue из API
                        position_value = float(position.get('positionValue', 0))
                        if position_value > 0:
                            # Если positionValue в разумных пределах для маржи (1-1000 USDT),
                            # используем его напрямую как маржу (в Bybit API positionValue часто уже содержит маржу)
                            if 1.0 <= position_value <= 1000.0:
                                margin = position_value
                            # Если positionValue значительно больше (вероятно стоимость позиции), делим на leverage
                            elif position_value > 1000.0 and leverage > 0:
                                margin = position_value / leverage
                        
                        # Если маржа все еще 0 или очень маленькая, логируем для отладки
                        if margin == 0 or margin < 0.01:
                            logger.warning(f"[BYBIT ROI DEBUG] {symbol}: margin={margin}, positionValue={position.get('positionValue')}, leverage={leverage}, size={position_size}, avgPrice={avg_price}, pnl={current_pnl}")
                            logger.warning(f"[BYBIT ROI DEBUG] Все поля позиции: {list(position.keys())}")
                            margin = 1.0  # Минимальная маржа для избежания деления на ноль
                        
                        roi = (current_pnl / margin * 100) if margin > 0 else 0
                        
                        # Логирование ROI убрано (слишком много логов)
                        # if current_pnl != 0:
                        #     logger.info(f"[BYBIT ROI] {symbol}: PnL={current_pnl:.4f} USDT, margin={margin:.4f} USDT, ROI={roi:.2f}%, positionValue={position.get('positionValue')}, leverage={leverage}, calculated={position_value / leverage if position_value > 0 and leverage > 0 else 'N/A'}")
                        
                        if current_pnl > 0:
                            if symbol not in self.max_profit_values or current_pnl > self.max_profit_values[symbol]:
                                self.max_profit_values[symbol] = current_pnl
                        else:
                            if symbol not in self.max_loss_values or current_pnl < self.max_loss_values[symbol]:
                                self.max_loss_values[symbol] = current_pnl
                        
                        mark_price = float(position.get('markPrice', 0) or 0)

                        position_info = {
                            'symbol': symbol,
                            'pnl': current_pnl,
                            'max_profit': self.max_profit_values.get(symbol, 0),
                            'max_loss': self.max_loss_values.get(symbol, 0),
                            'roi': roi,
                            'high_roi': roi > HIGH_ROI_THRESHOLD,
                            'high_loss': current_pnl < HIGH_LOSS_THRESHOLD,
                            'side': 'Long' if position['side'] == 'Buy' else 'Short',
                            'size': position_size,
                            'take_profit': position.get('takeProfit', ''),
                            'stop_loss': position.get('stopLoss', ''),
                            'mark_price': mark_price,
                            'avg_price': avg_price,
                            'entry_price': avg_price,
                            'current_price': mark_price,
                            'realized_pnl': float(position.get('cumRealisedPnl', 0)),
                            'leverage': float(position.get('leverage', 1))
                        }
                        
                        processed_positions.append(position_info)
                        
                        if symbol in self.daily_pnl:
                            start_pnl = self.daily_pnl[symbol]
                            if start_pnl > 0 and current_pnl > 0:
                                growth_ratio = current_pnl / start_pnl
                                if growth_ratio >= GROWTH_MULTIPLIER:
                                    rapid_growth_positions.append({
                                        'symbol': symbol,
                                        'start_pnl': start_pnl,
                                        'current_pnl': current_pnl,
                                        'growth_ratio': growth_ratio
                                    })
                        else:
                            self.daily_pnl[symbol] = current_pnl
                    
                    return processed_positions, rapid_growth_positions
                    
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning("Attempt {} failed: {}, retrying in {} seconds...".format(attempt + 1, str(e), retry_delay))
                        time.sleep(retry_delay)
                        continue
                    raise
                    
        except Exception as e:
            # Логируем ошибку через logger, не через print
            # print("Error getting positions: {}".format(str(e)))
            return [], []

    def get_closed_pnl(self, sort_by='time', period='all', start_date=None, end_date=None):
        """Получает историю закрытых позиций с PNL
        
        Args:
            sort_by: Способ сортировки ('time' или 'pnl')
            period: Период фильтрации ('all', 'day', 'week', 'month', 'half_year', 'year', 'custom')
            start_date: Начальная дата для custom периода (формат: 'YYYY-MM-DD' или timestamp в мс)
            end_date: Конечная дата для custom периода (формат: 'YYYY-MM-DD' или timestamp в мс)
        """
        try:
            all_closed_pnl = []
            
            def safe_float(value, default=None):
                try:
                    if value in (None, ''):
                        return default
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            def timestamp_to_iso(ts_ms):
                try:
                    if ts_ms is None:
                        return None
                    ts_ms = int(ts_ms)
                    return datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    return None
            
            def build_pnl_record(pos: Dict[str, Any]) -> Dict[str, Any]:
                close_ts = int(pos.get('updatedTime', time.time() * 1000))
                created_ts = pos.get('createdTime')
                created_ts = int(created_ts) if created_ts else None
                
                entry_price = safe_float(pos.get('avgEntryPrice'), 0.0) or 0.0
                exit_price = safe_float(pos.get('avgExitPrice'), 0.0) or 0.0
                qty = safe_float(pos.get('qty'))
                if qty is None or qty == 0:
                    qty = safe_float(pos.get('closedSize'))
                if qty is None or qty == 0:
                    qty = safe_float(pos.get('size'), 0.0)
                qty = abs(qty or 0.0)
                
                position_value = safe_float(pos.get('cumEntryValue'))
                if position_value is None and entry_price and qty:
                    position_value = abs(entry_price * qty)
                
                leverage = safe_float(pos.get('leverage'))
                
                return {
                    'symbol': clean_symbol(pos['symbol']),
                    'symbol_raw': pos.get('symbol'),
                    'qty': qty,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'closed_pnl': safe_float(pos.get('closedPnl'), 0.0) or 0.0,
                    'close_time': timestamp_to_iso(close_ts),
                    'close_timestamp': close_ts,
                    'created_time': timestamp_to_iso(created_ts),
                    'created_timestamp': created_ts,
                    'exchange': 'bybit',
                    'side': pos.get('side'),
                    'order_type': pos.get('orderType'),
                    'position_value': position_value,
                    'position_value_entry': safe_float(pos.get('cumEntryValue')),
                    'position_value_exit': safe_float(pos.get('cumExitValue')),
                    'leverage': leverage,
                    'raw_record': {
                        'qty': pos.get('qty'),
                        'closedSize': pos.get('closedSize'),
                        'positionSize': pos.get('size')
                    }
                }
            
            # Получаем текущее время
            end_time = int(time.time() * 1000)
            end_dt = datetime.fromtimestamp(end_time / 1000)
            
            # Определяем диапазон дат в зависимости от периода
            # Максимальный период для Bybit API - 2 года (730 дней)
            MAX_PERIOD_MS = 730 * 24 * 60 * 60 * 1000
            
            if period == 'custom' and start_date and end_date:
                # Парсим даты для custom периода
                try:
                    if isinstance(start_date, str):
                        if '-' in start_date:  # Формат 'YYYY-MM-DD'
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            period_start = int(start_dt.timestamp() * 1000)
                        else:  # Timestamp
                            period_start = int(start_date)
                    else:
                        period_start = int(start_date)
                    
                    if isinstance(end_date, str):
                        if '-' in end_date:  # Формат 'YYYY-MM-DD'
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            # Добавляем 23:59:59 к конечной дате
                            end_dt = end_dt.replace(hour=23, minute=59, second=59)
                            period_end = int(end_dt.timestamp() * 1000)
                        else:  # Timestamp
                            period_end = int(end_date)
                    else:
                        period_end = int(end_date)
                    
                    # Проверяем, что период не превышает 2 года
                    if period_end - period_start > MAX_PERIOD_MS:
                        logger.warning(f"Custom period exceeds 2 years limit. Limiting to last 2 years from end_date")
                        period_start = period_end - MAX_PERIOD_MS
                    
                    # Проверяем, что начальная дата не старше 2 лет от текущего времени
                    if end_time - period_start > MAX_PERIOD_MS:
                        logger.warning(f"Start date is older than 2 years. Limiting to last 2 years from now")
                        period_start = end_time - MAX_PERIOD_MS
                        if period_end > end_time:
                            period_end = end_time
                            
                except Exception as e:
                    logger.error(f"Error parsing custom dates: {e}")
                    # Используем период 1.5 года при ошибке парсинга
                    period_start = end_time - (547 * 24 * 60 * 60 * 1000)
                    period_end = end_time
            elif period == 'day':
                # Начало текущего дня (00:00:00)
                day_start = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                period_start = int(day_start.timestamp() * 1000)
                period_end = end_time
            elif period == 'week':
                # Начало текущей недели (понедельник 00:00:00)
                days_since_monday = end_dt.weekday()  # 0 = понедельник, 6 = воскресенье
                week_start = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                week_start = week_start - timedelta(days=days_since_monday)
                period_start = int(week_start.timestamp() * 1000)
                period_end = end_time
            elif period == 'month':
                # Начало текущего месяца (1-е число 00:00:00)
                month_start = end_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_start = int(month_start.timestamp() * 1000)
                period_end = end_time
            elif period == 'half_year':
                # Начало текущего полугодия (январь или июль, 1-е число 00:00:00)
                if end_dt.month <= 6:
                    half_year_start = end_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                else:
                    half_year_start = end_dt.replace(month=7, day=1, hour=0, minute=0, second=0, microsecond=0)
                period_start = int(half_year_start.timestamp() * 1000)
                period_end = end_time
            elif period == 'year':
                # Начало текущего года (1 января 00:00:00)
                year_start = end_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                period_start = int(year_start.timestamp() * 1000)
                period_end = end_time
            else:  # period == 'all'
                # Для 'all' получаем данные за последние 1.5 года (максимум для Bybit API - 2 года)
                # Используем 1.5 года (547 дней) чтобы быть в безопасности
                period_start = end_time - (547 * 24 * 60 * 60 * 1000)
                period_end = end_time
            
            # Разбиваем запрос на периоды по 7 дней для избежания лимитов API
            if period == 'all' or (period_end - period_start) > (7 * 24 * 60 * 60 * 1000):
                # Для больших периодов разбиваем на части
                current_end = period_end
                while current_end > period_start:
                    current_start = max(current_end - (7 * 24 * 60 * 60 * 1000), period_start)
                    
                    try:
                        cursor = None
                        while True:
                            params = {
                                "category": "linear",
                                "settleCoin": "USDT",
                                "limit": 100,
                                "startTime": str(int(current_start)),
                                "endTime": str(int(current_end))
                            }
                            if cursor:
                                params["cursor"] = cursor
                            
                            response = self.client.get_closed_pnl(**params)
                            
                            if not response:
                                break
                            
                            # Обрабатываем ошибку API о лимите в 2 года
                            if response.get('retCode') != 0:
                                ret_msg = response.get('retMsg', '')
                                # Если ошибка о лимите в 2 года, пропускаем этот период
                                if '2 years' in ret_msg or 'ErrCode: 10001' in ret_msg:
                                    logger.warning(f"Bybit API: Cannot query data older than 2 years. Skipping period {current_start}-{current_end}")
                                    break
                                else:
                                    # Для других ошибок также прерываем
                                    break
                            
                            positions = response['result'].get('list', [])
                            if not positions:
                                break
                            
                            for pos in positions:
                                all_closed_pnl.append(build_pnl_record(pos))
                            
                            cursor = response['result'].get('nextPageCursor')
                            if not cursor:
                                break
                                
                    except Exception as e:
                        logger.error(f"Error fetching closed PNL for period: {e}")
                        break
                    
                    current_end = current_start
            else:
                # Для коротких периодов делаем один запрос
                try:
                    cursor = None
                    while True:
                        params = {
                            "category": "linear",
                            "settleCoin": "USDT",
                            "limit": 100,
                            "startTime": str(int(period_start)),
                            "endTime": str(int(period_end))
                        }
                        if cursor:
                            params["cursor"] = cursor
                        
                        response = self.client.get_closed_pnl(**params)
                        
                        if not response:
                            break
                        
                        # Обрабатываем ошибку API о лимите в 2 года
                        if response.get('retCode') != 0:
                            ret_msg = response.get('retMsg', '')
                            # Если ошибка о лимите в 2 года, пропускаем этот период
                            if '2 years' in ret_msg or 'ErrCode: 10001' in ret_msg:
                                logger.warning(f"Bybit API: Cannot query data older than 2 years. Period {period_start}-{period_end} is too old")
                                break
                            else:
                                # Для других ошибок также прерываем
                                break
                        
                        positions = response['result'].get('list', [])
                        if not positions:
                            break
                        
                        for pos in positions:
                            all_closed_pnl.append(build_pnl_record(pos))
                        
                        cursor = response['result'].get('nextPageCursor')
                        if not cursor:
                            break
                            
                except Exception as e:
                    logger.error(f"Error fetching closed PNL: {e}")
            
            # Фильтруем по датам (на случай если API вернул данные вне диапазона)
            if period != 'all':
                filtered_pnl = []
                for pnl in all_closed_pnl:
                    close_ts = pnl.get('close_timestamp', 0)
                    if period_start <= close_ts <= period_end:
                        filtered_pnl.append(pnl)
                all_closed_pnl = filtered_pnl
            
            # Сортировка
            if sort_by == 'pnl':
                all_closed_pnl.sort(key=lambda x: abs(float(x['closed_pnl'])), reverse=True)
            else:  # По умолчанию сортируем по времени
                all_closed_pnl.sort(key=lambda x: x.get('close_timestamp', 0), reverse=True)
            
            return all_closed_pnl
            
        except Exception as e:
            logger.error(f"Error in get_closed_pnl: {e}")
            return []

    def get_symbol_chart_data(self, symbol):
        """Получает исторические данные для графика"""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=f"{symbol}USDT",
                interval="5",  # 5 минут
                limit=24  # 2 часа данных
            )
            if response['retCode'] == 0:
                return [float(k[4]) for k in response['result']['list']]  # Берем цены закрытия
            return []
        except Exception as e:
            logger.error(f"Error getting chart data for {symbol}: {e}")
            return []

    def get_sma200_position(self, symbol):
        """Определяет положение цены относительно SMA200"""
        retries = 3
        retry_delay = 5
        
        for attempt in range(retries):
            try:
                response = self.client.get_kline(
                    category="linear",
                    symbol=f"{symbol}USDT",
                    interval="D",
                    limit=200
                )
                
                if response['retCode'] == 0:
                    closes = [float(k[4]) for k in response['result']['list']]
                    if len(closes) >= 200:
                        sma200 = sum(closes[:200]) / 200
                        current_price = float(closes[0])
                        result = current_price > sma200
                        return result
                return None
                
            except (ConnectionError, IncompleteRead, RemoteDisconnected, requests.exceptions.ConnectionError) as e:
                logger.error(f"Error getting SMA200 for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Error getting SMA200 for {symbol}: {e}")
                return None

    @with_timeout(15)  # 15 секунд таймаут для получения тикера
    def get_ticker(self, symbol):
        """Получение текущих данных тикера"""
        retries = 3
        base_delay = 0.1
        last_error = None
        
        for attempt in range(1, retries + 1):
            try:
                # Добавляем задержку для предотвращения rate limiting + экспоненциальную паузу между ретраями
                time.sleep(base_delay * attempt)
                
                response = self.client.get_tickers(
                    category="linear",
                    symbol=f"{symbol}USDT"
                )
                
                if response['retCode'] == 0 and response['result']['list']:
                    ticker = response['result']['list'][0]
                    return {
                        'symbol': symbol,
                        'last': float(ticker['lastPrice']),
                        'bid': float(ticker['bid1Price']),
                        'ask': float(ticker['ask1Price']),
                        'timestamp': response['time']
                    }
                
                logger.debug(f"[BYBIT] ⚠️ Пустой ответ тикера для {symbol}: {response}")
            except Exception as e:
                last_error = e
                logger.warning(f"[BYBIT] ⚠️ Попытка {attempt}/{retries} получить тикер {symbol} не удалась: {e}")
        
        if last_error:
            logger.error(f"[BYBIT] ❌ Не удалось получить тикер {symbol}: {last_error}")
        return None

    def get_instruments_info(self, symbol):
        """Получает информацию об торговых правилах для символа"""
        try:
            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                instrument = response['result']['list'][0]
                result = {
                    'minOrderQty': instrument['lotSizeFilter']['minOrderQty'],
                    'qtyStep': instrument['lotSizeFilter']['qtyStep'],
                    'tickSize': instrument['priceFilter']['tickSize'],
                    'status': instrument.get('status', 'Unknown')  # ✅ Добавляем статус инструмента
                }
                # ✅ Проверяем наличие minNotionalValue (минимальная сумма ордера в USDT!)
                if 'lotSizeFilter' in instrument and 'minNotionalValue' in instrument['lotSizeFilter']:
                    result['minNotionalValue'] = float(instrument['lotSizeFilter']['minNotionalValue'])
                return result
            else:
                logger.warning(f"[BYBIT] ❌ Не удалось получить информацию об инструменте {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"[BYBIT] ❌ Ошибка получения информации об инструменте {symbol}: {e}")
            return {}
    
    def get_instrument_status(self, symbol):
        """
        Получает статус торговли для символа
        
        Возможные статусы Bybit:
        - Trading: Активная торговля
        - PreLaunch: Предварительный запуск (торговля невозможна)
        - Delivering: В процессе поставки
        - Closed: Закрыто (делистинг)
        
        Returns:
            dict: {'status': str, 'is_tradeable': bool, 'is_delisting': bool}
        """
        try:
            # Добавляем небольшую задержку для предотвращения rate limiting
            time.sleep(0.02)  # 20ms задержка для проверки статуса инструмента
            
            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                instrument = response['result']['list'][0]
                status = instrument.get('status', 'Unknown')
                
                return {
                    'status': status,
                    'is_tradeable': status == 'Trading',
                    'is_delisting': status in ['Closed', 'Delivering'],
                    'symbol': symbol
                }
            else:
                logger.warning(f"⚠️ Не удалось получить статус инструмента {symbol}")
                return {
                    'status': 'Unknown',
                    'is_tradeable': False,
                    'is_delisting': False,
                    'symbol': symbol
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса инструмента {symbol}: {e}")
            return {
                'status': 'Error',
                'is_tradeable': False,
                'is_delisting': False,
                'symbol': symbol,
                'error': str(e)
            }

    def close_position(self, symbol, size, side, order_type="Limit"):
        try:
            logger.info(f"[BYBIT] Закрытие позиции {symbol}, объём: {size}, сторона: {side}, тип: {order_type}")
            
            # Проверяем существование активной позиции
            try:
                response = self.client.get_positions(
                    category="linear",
                    symbol=f"{symbol}USDT"
                )
                
                if not response or response.get('retCode') != 0:
                    return {
                        'success': False,
                        'message': 'Ошибка при проверке позиций'
                    }
                
                positions = response['result']['list']
                active_position = None
                
                # ✅ КРИТИЧНО: Логируем все позиции для отладки
                logger.debug(f"[BYBIT] DEBUG: Получено позиций: {len(positions)}")
                for pos in positions:
                    logger.debug(f"[BYBIT] DEBUG: Позиция: symbol={pos.get('symbol')}, side={pos.get('side')}, size={pos.get('size')}")
                
                # Ищем позицию с нужной стороной
                # ✅ Нормализуем side (принимаем и 'Long', и 'LONG')
                normalized_side = side if side in ['Long', 'Short'] else ('Long' if side.upper() == 'LONG' else 'Short' if side.upper() == 'SHORT' else side)
                
                for pos in positions:
                    pos_side = 'Long' if pos['side'] == 'Buy' else 'Short'
                    pos_size = abs(float(pos['size']))
                    logger.debug(f"[BYBIT] DEBUG: Проверка: pos_side={pos_side}, normalized_side={normalized_side}, pos_size={pos_size}")
                    
                    if pos_size > 0 and pos_side == normalized_side:
                        active_position = pos
                        logger.debug(f"[BYBIT] DEBUG: ✅ Найдена позиция: {active_position.get('symbol')}, size={active_position.get('size')}")
                        break
                
                if not active_position:
                    # ✅ Детальное логирование для отладки
                    logger.debug(f"[BYBIT] DEBUG: ❌ Позиция не найдена! Искали: side={normalized_side} (было {side}), symbol={symbol}USDT")
                    return {
                        'success': False,
                        'message': f'Нет активной {side} позиции для {symbol}'
                    }
                
                # Позиция найдена - не логируем для уменьшения спама
                # print(f"[BYBIT] Found active position: {active_position}")
                
            except Exception as e:
                logger.error(f"[BYBIT] Ошибка при проверке позиций: {str(e)}")
                return {
                    'success': False,
                    'message': f'Ошибка при проверке позиций: {str(e)}'
                }
            
            # Получаем текущую цену
            ticker = self.get_ticker(symbol)
            if not ticker:
                return {
                    'success': False,
                    'message': 'Не удалось получить текущую рыночную цену'
                }
            
            # Определяем сторону для закрытия (противоположную текущей позиции)
            close_side = "Sell" if side == "Long" else "Buy"
            
            # ✅ КРИТИЧНО: Определяем positionIdx в зависимости от режима позиции
            # В One-Way Mode: position_idx = 0 (для обеих сторон)
            # В Hedge Mode: position_idx = 1 (LONG), position_idx = 2 (SHORT)
            position_mode = self._get_position_mode(symbol)
            if position_mode == 'One-Way':
                position_idx = 0
                logger.debug(f"[BYBIT] {symbol}: One-Way mode, используем position_idx=0 для закрытия")
            else:
                # Hedge mode
                position_idx = 1 if side == "Long" or side.upper() == "LONG" else 2
                logger.debug(f"[BYBIT] {symbol}: Hedge mode, используем position_idx={position_idx} для закрытия {side}")
            
            # Базовые параметры ордера
            order_params = {
                "category": "linear",
                "symbol": f"{symbol}USDT",
                "side": close_side,
                "orderType": order_type.upper(),  # Важно: используем верхний регистр
                "qty": str(size),
                "reduceOnly": True,
                "positionIdx": position_idx
            }

            # Добавляем цену для лимитных ордеров
            if order_type.upper() == "LIMIT":  # Проверяем в верхнем регистре
                price_multiplier = (100 - self.limit_order_offset) / 100 if close_side == "Buy" else (100 + self.limit_order_offset) / 100
                limit_price = ticker['ask'] * price_multiplier if close_side == "Buy" else ticker['bid'] * price_multiplier
                
                # ✅ КРИТИЧНО: Используем 6 знаков после запятой для поддержки дешевых монет (например, MEW ~0.005)
                # round(0.005, 2) = 0.00 ❌ → round(0.005, 6) = 0.005 ✅
                order_params["price"] = str(round(limit_price, 6))
                order_params["timeInForce"] = "GTC"
                logger.debug(f"[BYBIT] Calculated limit price: {limit_price} → rounded: {round(limit_price, 6)}")
            
            logger.debug(f"[BYBIT] Sending order with params: {order_params}")
            response = self.client.place_order(**order_params)
            logger.debug(f"[BYBIT] Order response: {response}")
            
            if response['retCode'] == 0:
                close_price = float(order_params.get('price', ticker['last']))
                return {
                    'success': True,
                    'order_id': response['result']['orderId'],
                    'message': f'{order_type} ордер успешно размещён',
                    'close_price': close_price
                }
            else:
                return {
                    'success': False,
                    'message': f"Не удалось разместить {order_type} ордер: {response['retMsg']}"
                }
                
        except Exception as e:
            logger.error(f"[BYBIT] Ошибка при закрытии позиции: {str(e)}")
            import traceback
            logger.error(f"[BYBIT] Трейсбек: {traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Ошибка при закрытии позиции: {str(e)}"
            }

    def get_all_pairs(self):
        """Получение списка всех доступных ессрочных фьючерсов"""
        try:
            logger.info("Запрос списка всех торговых пар...")
            
            response = self.client.get_instruments_info(
                category="linear",
                limit=1000,  # Увеличиваем лимит чтобы получить ВСЕ инструменты
                status="Trading"  # Только активные для торговли
            )
            
            if response and response.get('retCode') == 0 and response['result']['list']:
                all_instruments = response['result']['list']
                logger.info(f"Получено {len(all_instruments)} инструментов")
                
                # Фильтруем только бессрочные контракты (USDT)
                usdt_pairs = [
                    item for item in all_instruments
                    if item['symbol'].endswith('USDT')
                ]
                logger.info(f"Найдено {len(usdt_pairs)} USDT пар")
                
                # Дополнительная фильтрация по статусу
                trading_pairs = [
                    item for item in usdt_pairs 
                    if item.get('status') == 'Trading'
                ]
                logger.info(f"В торговле: {len(trading_pairs)} пар")
                
                pairs = [
                    clean_symbol(item['symbol'])
                    for item in trading_pairs
                ]
                
                # Логируем только общее количество пар
                logger.info(f"✅ Загружено {len(pairs)} торговых пар")
                
                return sorted(pairs)
            else:
                logger.error(f"Ошибка API: {response.get('retMsg', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting pairs: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    @with_timeout(30)  # 30 секунд таймаут для получения данных графика
    def get_chart_data(self, symbol, timeframe='1h', period='1w'):
        """Получение данных для графика
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', 'all')
            period (str): Период ('1d', '1w', '1M')
            
        Returns:
            dict: Данные для построения графика
        """
        # Добавляем задержку для предотвращения rate limiting (динамическая задержка)
        time.sleep(self.current_request_delay)
        
        try:
            # Специальная обработка для таймфрейма "all"
            if timeframe == 'all':
                # Последовательно пробуем разные интервалы
                intervals = [
                    ('1', '1m'),
                    ('5', '5m'),
                    ('15', '15m'),
                    ('30', '30m'),
                    ('60', '1h'),
                    ('240', '4h'),
                    ('360', '6h'),
                    ('D', '1d'),
                    ('W', '1w')
                ]
                
                selected_interval = None
                selected_klines = None
                
                for interval, interval_name in intervals:
                    try:
                        logger.debug(f"[BYBIT] Пробуем интервал {interval_name}")
                        # Убираем USDT если он уже есть в символе
                        clean_sym = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                        
                        # Повторные попытки при rate limit
                        max_retries = 3
                        retry_count = 0
                        response = None
                        
                        while retry_count < max_retries:
                            try:
                                response = self.client.get_kline(
                                    category="linear",
                                    symbol=f"{clean_sym}USDT",
                                    interval=interval,
                                    limit=1000
                                )
                                
                                # Обработка rate limiting в ответе
                                if response.get('retCode') == 10006:
                                    # Увеличиваем задержку, но не превышаем максимум
                                    delay = self.increase_request_delay(
                                        reason=f"Rate limit для {symbol} ({interval_name})"
                                    )
                                    
                                    # Добавляем дополнительную задержку при rate limit (минимум 2 секунды)
                                    additional_delay = max(2.0, delay * 0.5)
                                    total_delay = delay + additional_delay
                                    
                                    logger.error(f"❌ [BOTS] Too many visits. Exceeded the API Rate Limit. (ErrCode: 10006). Hit the API rate limit on https://api.bybit.com/v5/market/kline?category=linear&interval={interval}&limit=1000&symbol={clean_sym}USDT. Sleeping then trying again.")
                                    time.sleep(total_delay)
                                    retry_count += 1
                                    
                                    if retry_count < max_retries:
                                        logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} ({interval_name}) после паузы {total_delay:.1f}с...")
                                        continue
                                    else:
                                        logger.error(f"❌ Превышено максимальное количество попыток для {symbol} ({interval_name})")
                                        break
                                # Обработка ошибки timestamp (10002)
                                elif response.get('retCode') == 10002:
                                    # Увеличиваем recv_window и повторяем запрос
                                    current_recv_window = getattr(self.client, 'recv_window', 20000)
                                    new_recv_window = min(current_recv_window + 2500, 60000)
                                    self.client.recv_window = new_recv_window
                                    logger.error(f"❌ [BOTS] invalid request, please check your server timestamp or recv_window param. req_timestamp[{int(time.time() * 1000)}],server_timestamp[{response.get('time', int(time.time() * 1000))}],recv_window[{new_recv_window}] (ErrCode: 10002). Added 2.5 seconds to recv_window. Retrying...")
                                    time.sleep(1.0)
                                    retry_count += 1
                                    
                                    if retry_count < max_retries:
                                        logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} ({interval_name}) с увеличенным recv_window...")
                                        continue
                                    else:
                                        logger.error(f"❌ Превышено максимальное количество попыток для {symbol} ({interval_name})")
                                        break
                                else:
                                    # Успешный ответ - выходим из цикла повторных попыток
                                    break
                            except Exception as api_error:
                                # Перехватываем исключения от pybit (rate limit выбрасывает исключение)
                                error_str = str(api_error).lower()
                                if 'rate limit' in error_str or 'too many' in error_str or '10006' in error_str or 'x-bapi-limit-reset-timestamp' in error_str:
                                    # Увеличиваем задержку, но не превышаем максимум
                                    delay = self.increase_request_delay(
                                        reason=f"Rate limit (исключение) для {symbol} ({interval_name})"
                                    )
                                    
                                    # Добавляем дополнительную задержку при rate limit (минимум 2 секунды)
                                    additional_delay = max(2.0, delay * 0.5)
                                    total_delay = delay + additional_delay
                                    
                                    logger.error(f"❌ [BOTS] Too many visits. Exceeded the API Rate Limit. (ErrCode: 10006). Hit the API rate limit on https://api.bybit.com/v5/market/kline?category=linear&interval={interval}&limit=1000&symbol={clean_sym}USDT. Sleeping then trying again.")
                                    time.sleep(total_delay)
                                    retry_count += 1
                                    
                                    if retry_count < max_retries:
                                        logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} ({interval_name}) после исключения и паузы {total_delay:.1f}с...")
                                        continue
                                    else:
                                        logger.error(f"❌ Превышено максимальное количество попыток для {symbol} ({interval_name})")
                                        break
                                elif '10002' in error_str or 'timestamp' in error_str or 'recv_window' in error_str:
                                    # Обработка ошибки timestamp
                                    current_recv_window = getattr(self.client, 'recv_window', 20000)
                                    new_recv_window = min(current_recv_window + 2500, 60000)
                                    self.client.recv_window = new_recv_window
                                    logger.error(f"❌ [BOTS] invalid request, please check your server timestamp or recv_window param. (ErrCode: 10002). Added 2.5 seconds to recv_window. Retrying...")
                                    time.sleep(1.0)
                                    retry_count += 1
                                    
                                    if retry_count < max_retries:
                                        logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} ({interval_name}) с увеличенным recv_window...")
                                        continue
                                    else:
                                        logger.error(f"❌ Превышено максимальное количество попыток для {symbol} ({interval_name})")
                                        break
                                else:
                                    # Другая ошибка - пробрасываем дальше
                                    raise
                        
                        # Если все попытки исчерпаны - пропускаем этот интервал
                        if response and (response.get('retCode') == 10006 or response.get('retCode') == 10002):
                            continue
                        if response is None:
                            # Если response None после всех попыток - пропускаем интервал
                            continue
                        
                        if response['retCode'] == 0:
                            klines = response['result']['list']
                            if len(klines) <= 500:
                                selected_interval = interval
                                selected_klines = klines
                                logger.debug(f"[BYBIT] Выбран интервал {interval_name} ({len(klines)} свечей)")
                                break
                            
                            # Если это последний интервал, используем его независимо от количества свечей
                            if interval == 'W':
                                selected_interval = interval
                                selected_klines = klines
                                logger.debug(f"[BYBIT] Использован последний интервал {interval_name} ({len(klines)} свечей)")
                    except Exception as e:
                        logger.error(f"[BYBIT] Ошибка при получении данных для интервала {interval_name}: {e}")
                        continue
                
                if selected_interval and selected_klines:
                    candles = []
                    for k in selected_klines:
                        candle = {
                            'time': int(k[0]),
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5])
                        }
                        candles.append(candle)
                    
                    # Сортируем свечи от старых к новым
                    candles.sort(key=lambda x: x['time'])
                    
                    self.reset_request_delay()
                    return {
                        'success': True,
                        'data': {
                            'candles': candles
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': "Не удалось получить данные ни для одного интервала"
                    }
            else:
                # Стандартная обработка для конкретного таймфрейма
                timeframe_map = {
                    '1m': '1',
                    '5m': '5',
                    '15m': '15',
                    '30m': '30',
                    '1h': '60',
                    '4h': '240',
                    '6h': '360',
                    '1d': 'D',
                    '1w': 'W'
                }
                
                interval = timeframe_map.get(timeframe)
                if not interval:
                    logger.warning(f"[BYBIT] Неподдерживаемый таймфрейм: {timeframe}")
                    return {
                        'success': False,
                        'error': f'Неподдерживаемый таймфрейм: {timeframe}'
                    }
                
                # Убираем USDT если он уже есть в символе
                clean_sym = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                
                # Повторные попытки при rate limit
                max_retries = 3
                retry_count = 0
                response = None
                
                while retry_count < max_retries:
                    try:
                        response = self.client.get_kline(
                            category="linear",
                            symbol=f"{clean_sym}USDT",
                            interval=interval,
                            limit=1000
                        )
                        
                        # Обработка rate limiting в ответе
                        if response.get('retCode') == 10006:
                            # Увеличиваем задержку, но не превышаем максимум
                            delay = self.increase_request_delay(
                                reason=f"Rate limit для {symbol}"
                            )
                            
                            # Добавляем дополнительную задержку при rate limit (минимум 2 секунды)
                            additional_delay = max(2.0, delay * 0.5)
                            total_delay = delay + additional_delay
                            
                            logger.error(f"❌ [BOTS] Too many visits. Exceeded the API Rate Limit. (ErrCode: 10006). Hit the API rate limit on https://api.bybit.com/v5/market/kline?category=linear&interval={interval}&limit=1000&symbol={clean_sym}USDT. Sleeping then trying again.")
                            time.sleep(total_delay)
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} после паузы {total_delay:.1f}с...")
                                continue
                            else:
                                logger.error(f"❌ Превышено максимальное количество попыток для {symbol}")
                                return {
                                    'success': False,
                                    'error': 'Rate limit exceeded, maximum retries reached'
                                }
                        # Обработка ошибки timestamp (10002)
                        elif response.get('retCode') == 10002:
                            # Увеличиваем recv_window и повторяем запрос
                            current_recv_window = getattr(self.client, 'recv_window', 20000)
                            new_recv_window = min(current_recv_window + 2500, 60000)  # Максимум 60 секунд
                            self.client.recv_window = new_recv_window
                            logger.error(f"❌ [BOTS] invalid request, please check your server timestamp or recv_window param. req_timestamp[{int(time.time() * 1000)}],server_timestamp[{response.get('time', int(time.time() * 1000))}],recv_window[{new_recv_window}] (ErrCode: 10002). Added 2.5 seconds to recv_window. Retrying...")
                            time.sleep(1.0)  # Небольшая задержка перед повтором
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} с увеличенным recv_window...")
                                continue
                            else:
                                logger.error(f"❌ Превышено максимальное количество попыток для {symbol}")
                                return {
                                    'success': False,
                                    'error': 'Timestamp error, maximum retries reached'
                                }
                        else:
                            # Успешный ответ или другая ошибка - выходим из цикла
                            break
                    except Exception as api_error:
                        # Перехватываем исключения от pybit (rate limit выбрасывает исключение)
                        error_str = str(api_error).lower()
                        if 'rate limit' in error_str or 'too many' in error_str or '10006' in error_str or 'x-bapi-limit-reset-timestamp' in error_str:
                            # Увеличиваем задержку, но не превышаем максимум
                            delay = self.increase_request_delay(
                                reason=f"Rate limit (исключение) для {symbol}"
                            )
                            
                            # Добавляем дополнительную задержку при rate limit (минимум 2 секунды)
                            additional_delay = max(2.0, delay * 0.5)
                            total_delay = delay + additional_delay
                            
                            logger.error(f"❌ [BOTS] Too many visits. Exceeded the API Rate Limit. (ErrCode: 10006). Hit the API rate limit on https://api.bybit.com/v5/market/kline?category=linear&interval={interval}&limit=1000&symbol={clean_sym}USDT. Sleeping then trying again.")
                            time.sleep(total_delay)
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} после исключения и паузы {total_delay:.1f}с...")
                                continue
                            else:
                                logger.error(f"❌ Превышено максимальное количество попыток для {symbol}")
                                return {
                                    'success': False,
                                    'error': 'Rate limit exceeded, maximum retries reached'
                                }
                        elif '10002' in error_str or 'timestamp' in error_str or 'recv_window' in error_str:
                            # Обработка ошибки timestamp
                            current_recv_window = getattr(self.client, 'recv_window', 20000)
                            new_recv_window = min(current_recv_window + 2500, 60000)
                            self.client.recv_window = new_recv_window
                            logger.error(f"❌ [BOTS] invalid request, please check your server timestamp or recv_window param. (ErrCode: 10002). Added 2.5 seconds to recv_window. Retrying...")
                            time.sleep(1.0)
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                logger.info(f"🔄 Повторная попытка {retry_count}/{max_retries} для {symbol} с увеличенным recv_window...")
                                continue
                            else:
                                logger.error(f"❌ Превышено максимальное количество попыток для {symbol}")
                                return {
                                    'success': False,
                                    'error': 'Timestamp error, maximum retries reached'
                                }
                        else:
                            # Другая ошибка - пробрасываем дальше
                            raise
                
                # Если все попытки исчерпаны
                if response and (response.get('retCode') == 10006 or response.get('retCode') == 10002):
                    error_msg = 'Rate limit exceeded, please try again later' if response.get('retCode') == 10006 else 'Timestamp error, please try again later'
                    return {
                        'success': False,
                        'error': error_msg
                    }
                if response is None:
                    return {
                        'success': False,
                        'error': 'Rate limit exceeded, please try again later'
                    }
                
                if response['retCode'] == 0:
                    candles = []
                    for k in response['result']['list']:
                        candle = {
                            'time': int(k[0]),
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5])
                        }
                        candles.append(candle)
                    
                    # Сортируем свечи от старых к новым
                    candles.sort(key=lambda x: x['time'])
                    
                    self.reset_request_delay()
                    return {
                        'success': True,
                        'data': {
                            'candles': candles
                        }
                    }
                
                return {
                    'success': False,
                    'error': f"Ошибка API: {response.get('retMsg', 'Неизвестная ошибка')}"
                }
            
        except Exception as e:
            logger.error(f"[BYBIT] Ошибка получения данных графика: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_indicators(self, symbol, timeframe='1h'):
        """Получение значений индикаторов
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм
            
        Returns:
            dict: Значения индикаторов
        """
        try:
            logger.debug(f"[BYBIT] Запрос индикаторов для {symbol}, таймфрейм: {timeframe}")
            
            # Конвертируем таймфрейм в формат Bybit
            timeframe_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '4h': '240',
                '1d': 'D',
                '1w': 'W'
            }
            
            interval = timeframe_map.get(timeframe)
            if not interval:
                logger.warning(f"[BYBIT] Неподдерживаемый таймфрейм: {timeframe}")
                return {
                    'success': False,
                    'error': f'Неподдерживаемый таймфрейм: {timeframe}'
                }

            # Получаем последние 100 свечей для расчета индикаторов
            # Убираем USDT если он уже есть в символе
            clean_sym = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            response = self.client.get_kline(
                category="linear",
                symbol=f"{clean_sym}USDT",
                interval=interval,
                limit=100
            )

            if not response or response.get('retCode') != 0:
                return {
                    'success': False,
                    'error': 'Не удалось получить данные свечей'
                }

            klines = response.get('result', {}).get('list', [])
            if not klines:
                return {
                    'success': False,
                    'error': 'Нет данных свечей'
                }

            # Преобразуем данные в массивы для расчетов
            closes = np.array([float(k[4]) for k in klines])  # Цены закрытия
            highs = np.array([float(k[2]) for k in klines])   # Максимумы
            lows = np.array([float(k[3]) for k in klines])    # Минимумы
            volumes = np.array([float(k[5]) for k in klines])  # Объемы
            timestamps = [int(k[0]) for k in klines]          # Временные метки

            # 1. Расчет RSI
            rsi = self._calculate_rsi(closes)
            current_rsi = rsi[-1]
            
            # Определение состояния RSI
            rsi_status = "Нейтральный"
            if current_rsi >= 70:
                rsi_status = "Перекуплен"
            elif current_rsi <= 30:
                rsi_status = "Перепродан"

            # 2. Расчет тренда
            trend_info = self._calculate_trend(closes)
            
            # 3. Расчет объемов
            volume_info = self._calculate_volume_metrics(volumes)

            # 4. Расчет уровней поддержки и сопротивления
            support_resistance = self._calculate_support_resistance(highs, lows, closes)

            # 5. Расчет точек входа/выхода
            entry_exit = self._calculate_entry_exit_points(
                closes[-1], 
                support_resistance['support'], 
                support_resistance['resistance'],
                trend_info['direction']
            )

            # 6. Расчет торгового канала
            channel = self._calculate_trading_channel(highs, lows)

            # Формируем рекомендацию
            recommendation = self._generate_recommendation(
                current_rsi,
                trend_info['direction'],
                closes[-1],
                support_resistance,
                volume_info['volume_trend']
            )

            return {
                'success': True,
                'data': {
                    'time': {
                        'timestamp': timestamps[-1],
                        'datetime': datetime.fromtimestamp(timestamps[-1]/1000).strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'price': {
                        'current': closes[-1],
                        'high_24h': max(highs[-24:]) if len(highs) >= 24 else highs[-1],
                        'low_24h': min(lows[-24:]) if len(lows) >= 24 else lows[-1]
                    },
                    'rsi': {
                        'value': round(current_rsi, 2),
                        'status': rsi_status
                    },
                    'trend': {
                        'direction': trend_info['direction'],
                        'strength': trend_info['strength']
                    },
                    'volume': {
                        'current_24h': volume_info['current_24h'],
                        'change_percent': volume_info['change_percent'],
                        'trend': volume_info['volume_trend']
                    },
                    'levels': {
                        'support': support_resistance['support'],
                        'resistance': support_resistance['resistance']
                    },
                    'entry_exit': {
                        'entry_point': entry_exit['entry_point'],
                        'stop_loss': entry_exit['stop_loss'],
                        'target': entry_exit['target']
                    },
                    'channel': {
                        'upper': channel['upper'],
                        'lower': channel['lower'],
                        'position': channel['position']
                    },
                    'recommendation': recommendation
                }
            }

        except Exception as e:
            logger.error(f"[BYBIT] Ошибка при расчете индикаторов: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_rsi(self, closes, period=14):
        """Расчет RSI"""
        deltas = np.diff(closes)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(closes)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(closes)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi

    def _calculate_trend(self, closes):
        """Расчет тренда и его силы"""
        # Используем 20-периодную SMA для определения тренда
        sma20 = np.mean(closes[-20:])
        current_price = closes[-1]
        
        # Определяем направление тренда
        if current_price > sma20 * 1.02:  # Цена выше SMA на 2%
            direction = "Восходящий"
        elif current_price < sma20 * 0.98:  # Цена ниже SMA на 2%
            direction = "Нисходящий"
        else:
            direction = "Боковой"

        # Рассчитываем силу тренда на основе отклонения от SMA
        deviation = abs((current_price - sma20) / sma20 * 100)
        if deviation < 2:
            strength = "Слабый"
        elif deviation < 5:
            strength = "Умеренный"
        else:
            strength = "Сильный"

        return {
            'direction': direction,
            'strength': strength
        }

    def _calculate_volume_metrics(self, volumes):
        """Расчет метрик объема"""
        current_24h = sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes)
        prev_24h = sum(volumes[-48:-24]) if len(volumes) >= 48 else sum(volumes)
        
        # Изменение объема
        if prev_24h > 0:
            change_percent = ((current_24h - prev_24h) / prev_24h) * 100
        else:
            change_percent = 0

        # Определяем тренд объема
        if change_percent > 10:
            volume_trend = "Растущий"
        elif change_percent < -10:
            volume_trend = "Падающий"
        else:
            volume_trend = "Стабильный"

        return {
            'current_24h': current_24h,
            'change_percent': round(change_percent, 2),
            'volume_trend': volume_trend
        }

    def _calculate_support_resistance(self, highs, lows, closes):
        """Расчет уровней поддержки и сопротивления"""
        # Используем метод кластеризации цен
        all_prices = np.concatenate([highs, lows, closes])
        price_clusters = {}

        # Группируем цены в кластеры с погрешностью 0.5%
        for price in all_prices:
            found_cluster = False
            for cluster_price in list(price_clusters.keys()):
                if abs(price - cluster_price) / cluster_price < 0.005:
                    price_clusters[cluster_price] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                price_clusters[price] = 1

        # Сортируем кластеры по количеству точек
        sorted_clusters = sorted(price_clusters.items(), key=lambda x: x[1], reverse=True)
        
        current_price = closes[-1]
        support = current_price
        resistance = current_price

        # Находим ближайшие уровни поддержки и сопротивления
        for price, _ in sorted_clusters:
            if price < current_price and price > support:
                support = price
            elif price > current_price and price < resistance:
                resistance = price

        return {
            'support': support,
            'resistance': resistance
        }

    def _calculate_entry_exit_points(self, current_price, support, resistance, trend):
        """Расчет точек входа, выхода и стоп-лосса"""
        # Расчет точки входа
        if trend == "Восходящий":
            entry_point = support + (resistance - support) * 0.382  # Уровень Фибоначчи
        else:
            entry_point = resistance - (resistance - support) * 0.382

        # Расчет стоп-лосса (2% от точки входа)
        stop_loss = entry_point * 0.98 if trend == "Восходящий" else entry_point * 1.02

        # Расчет целевой цены (соотношение риск/прибыль 1:2)
        risk = abs(entry_point - stop_loss)
        target = entry_point + (risk * 2) if trend == "Восходящий" else entry_point - (risk * 2)

        return {
            'entry_point': round(entry_point, 8),
            'stop_loss': round(stop_loss, 8),
            'target': round(target, 8)
        }

    def _calculate_trading_channel(self, highs, lows):
        """Расчет торгового канала"""
        # Используем последние 20 свечей для канала
        period = 20
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]

        upper = np.max(recent_highs)
        lower = np.min(recent_lows)
        current = (highs[-1] + lows[-1]) / 2

        # Определяем положение текущей цены в канале
        channel_height = upper - lower
        if channel_height > 0:
            position_percent = ((current - lower) / channel_height) * 100
            if position_percent < 25:
                position = "Нижняя часть канала"
            elif position_percent > 75:
                position = "Верхняя часть канала"
            else:
                position = "Середина канала"
        else:
            position = "Неопределено"

        return {
            'upper': upper,
            'lower': lower,
            'position': position
        }

    def _generate_recommendation(self, rsi, trend_direction, current_price, support_resistance, volume_trend):
        """Генерация торговой рекомендации"""
        if rsi >= 70 and trend_direction == "Восходящий" and volume_trend == "Падающий":
            return "Возможна коррекция - рекомендуется фиксация прибыли"
        elif rsi <= 30 and trend_direction == "Нисходящий" and volume_trend == "Растущий":
            return "Возможен отскок - рекомендуется поиск точки входа"
        elif trend_direction == "Восходящий" and current_price < support_resistance['resistance']:
            return "Восходящий тренд - рассмотреть покупку на откате"
        elif trend_direction == "Нисходящий" and current_price > support_resistance['support']:
            return "Нисходящий тренд - рассмотреть продажу на росте"
        else:
            return "Нейтральная ситуация - рекомендуется наблюдение"

    def get_wallet_balance(self):
        """Получает общий баланс кошелька и реализованный PNL (с кэшированием)"""
        import time
        current_time = time.time()
        
        # Проверяем кэш (используем более длинный TTL при сетевых ошибках)
        cache_ttl = self._wallet_balance_cache_ttl_error if self._network_error_count > 3 else self._wallet_balance_cache_ttl
        if (self._wallet_balance_cache is not None and 
            current_time - self._wallet_balance_cache_time < cache_ttl):
            return self._wallet_balance_cache
        
        try:
            # Получаем баланс кошелька
            wallet_response = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            
            if wallet_response['retCode'] != 0:
                raise Exception(f"Failed to get wallet balance: {wallet_response['retMsg']}")
                
            wallet_data = wallet_response['result']['list'][0]
            
            # Получаем значения из правильных полей
            total_balance = float(wallet_data['totalWalletBalance'])  # Общий баланс
            available_balance = float(wallet_data['totalAvailableBalance'])  # Доступный баланс
            
            # Получаем реализованный PNL из данных кошелька
            coin_data = wallet_data['coin'][0]  # Берем данные для USDT
            realized_pnl = float(coin_data['cumRealisedPnl'])  # Используем накопленный реализованный PNL
            
            result = {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'realized_pnl': realized_pnl
            }
            
            # Сохраняем в кэш при успешном запросе
            self._wallet_balance_cache = result
            self._wallet_balance_cache_time = current_time
            self._network_error_count = 0  # Сбрасываем счетчик ошибок
            
            return result
            
        except Exception as e:
            error_str = str(e)
            is_network_error = any(keyword in error_str.lower() for keyword in [
                'getaddrinfo failed', 'name resolution', 'connection', 
                'dns', 'network', 'timeout', 'resolve'
            ])
            
            # Логируем только раз в минуту при сетевых ошибках
            if is_network_error:
                self._network_error_count += 1
                if current_time - self._last_network_error_time > 60:  # Логируем не чаще раза в минуту
                    logger.warning(f"Network error getting wallet balance (count: {self._network_error_count}): {error_str[:100]}")
                    self._last_network_error_time = current_time
            else:
                logger.error(f"Error getting wallet balance: {error_str}")
            
            # Возвращаем кэш, если есть, иначе нули
            if self._wallet_balance_cache is not None:
                return self._wallet_balance_cache
            
            return {
                'total_balance': 0.0,
                'available_balance': 0.0,
                'realized_pnl': 0.0
            }
    
    def get_unified_account_info(self):
        """Получает полную информацию о едином торговом счете"""
        try:
            # Получаем баланс единого торгового счета
            wallet_response = self.client.get_wallet_balance(accountType="UNIFIED")
            
            if wallet_response["retCode"] != 0:
                return {
                    "success": False,
                    "error": f"API Error: {wallet_response['retMsg']}"
                }
            
            account_data = wallet_response["result"]["list"][0]
            
            account_info = {
                "total_equity": float(account_data.get("totalEquity", 0)),
                "total_wallet_balance": float(account_data.get("totalWalletBalance", 0)),
                "total_available_balance": float(account_data.get("totalAvailableBalance", 0)),
                "total_unrealized_pnl": float(account_data.get("totalPerpUPL", 0)),
                "total_margin_balance": float(account_data.get("totalMarginBalance", 0)),
                "account_type": "UNIFIED"
            }
            
            # Получаем ВСЕ открытые позиции используя ту же логику что и в get_positions()
            active_positions = 0
            total_position_value = 0.0
            cursor = None
            
            try:
                while True:
                    params = {
                        "category": "linear",
                        "settleCoin": "USDT",
                        "limit": 100
                    }
                    if cursor:
                        params["cursor"] = cursor
                    
                    response = self.client.get_positions(**params)
                    
                    if response["retCode"] != 0:
                        break
                    
                    positions = response['result']['list']
                    
                    # Считаем активные позиции на этой странице
                    for position in positions:
                        position_size = float(position.get("size", 0))
                        if abs(position_size) > 0:  # Любые открытые позиции
                            active_positions += 1
                            total_position_value += abs(float(position.get("positionValue", 0)))
                    
                    # Проверяем следующую страницу
                    cursor = response['result'].get('nextPageCursor')
                    if not cursor:
                        break
                account_info["active_positions"] = active_positions
                account_info["total_position_value"] = total_position_value
                
            except Exception as pos_error:
                account_info["active_positions"] = 0
                account_info["total_position_value"] = 0.0
            
            account_info["success"] = True
            return account_info
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception: {str(e)}"
            }

    def _get_position_mode(self, symbol):
        """
        Получает текущий режим позиции для символа с биржи.
        
        Returns:
            str: 'One-Way' или 'Hedge', или None если не удалось определить
        """
        # Проверяем кэш
        current_time = time.time()
        if (self._position_mode_cache is not None and 
            current_time - self._position_mode_cache_time < self._position_mode_cache_ttl):
            return self._position_mode_cache
        
        try:
            # Сначала пробуем использовать метод get_position_mode если он доступен (самый надежный способ)
            try:
                if hasattr(self.client, 'get_position_mode'):
                    mode_response = self.client.get_position_mode(category="linear", symbol=f"{symbol}USDT")
                    if mode_response.get('retCode') == 0:
                        result = mode_response.get('result', {})
                        # Bybit API возвращает mode как число: 0 = One-Way, 1 = Hedge
                        mode_value = result.get('mode')
                        if mode_value == 0:
                            mode = 'One-Way'
                        elif mode_value == 1:
                            mode = 'Hedge'
                        else:
                            # Fallback на строковое значение
                            mode = result.get('mode', 'One-Way')
                            if isinstance(mode, str):
                                mode = 'Hedge' if 'Hedge' in mode or 'hedge' in mode.lower() else 'One-Way'
                            else:
                                mode = 'Hedge' if mode else 'One-Way'
                        
                        # Сохраняем в кэш
                        self._position_mode_cache = mode
                        self._position_mode_cache_time = current_time
                        logger.debug(f"[BYBIT_BOT] Режим позиции для {symbol} (через API): {mode}")
                        return mode
            except Exception as e:
                logger.debug(f"[BYBIT_BOT] Метод get_position_mode недоступен или вернул ошибку: {e}")
            
            # Fallback: проверяем через позиции
            try:
                # Пробуем получить позицию для символа
                pos_response = self.client.get_positions(
                    category="linear",
                    symbol=f"{symbol}USDT"
                )
                
                if pos_response.get('retCode') == 0 and pos_response.get('result', {}).get('list'):
                    pos_list = pos_response['result']['list']
                    if pos_list:
                        # Если в ответе есть positionIdx и он не 0, значит Hedge mode
                        # Если positionIdx отсутствует или равен 0, значит One-Way mode
                        position_idx = pos_list[0].get('positionIdx')
                        if position_idx is not None and position_idx != 0:
                            # В Hedge mode positionIdx может быть 1 (LONG) или 2 (SHORT)
                            mode = 'Hedge'
                        else:
                            # В One-Way mode positionIdx должен быть 0 или отсутствовать
                            mode = 'One-Way'
                        
                        # Сохраняем в кэш
                        self._position_mode_cache = mode
                        self._position_mode_cache_time = current_time
                        logger.debug(f"[BYBIT_BOT] Режим позиции для {symbol} (через позиции): {mode}")
                        return mode
                    
                    # Если позиций нет для этого символа, пробуем проверить другие символы
                    # (режим позиции обычно одинаковый для всех символов в категории)
                    all_pos_response = self.client.get_positions(category="linear", limit=10)
                    if all_pos_response.get('retCode') == 0 and all_pos_response.get('result', {}).get('list'):
                        all_pos_list = all_pos_response['result']['list']
                        for pos in all_pos_list:
                            if abs(float(pos.get('size', 0))) > 0:  # Только активные позиции
                                position_idx = pos.get('positionIdx')
                                if position_idx is not None and position_idx != 0:
                                    mode = 'Hedge'
                                    # Сохраняем в кэш
                                    self._position_mode_cache = mode
                                    self._position_mode_cache_time = current_time
                                    logger.debug(f"[BYBIT_BOT] Режим позиции (определен через другие позиции): {mode}")
                                    return mode
                                else:
                                    mode = 'One-Way'
                                    # Сохраняем в кэш
                                    self._position_mode_cache = mode
                                    self._position_mode_cache_time = current_time
                                    logger.debug(f"[BYBIT_BOT] Режим позиции (определен через другие позиции): {mode}")
                                    return mode
            except Exception as e:
                logger.debug(f"[BYBIT_BOT] Не удалось определить режим позиции через позиции: {e}")
            
            # Если не удалось определить, используем значение из конфига как fallback
            mode = self.position_mode if hasattr(self, 'position_mode') else 'Hedge'
            logger.warning(f"[BYBIT_BOT] ⚠️ Не удалось определить режим позиции для {symbol}, используем fallback: {mode}")
            return mode
            
        except Exception as e:
            logger.warning(f"[BYBIT_BOT] ⚠️ Ошибка при получении режима позиции для {symbol}: {e}")
            # Fallback на значение из конфига
            mode = self.position_mode if hasattr(self, 'position_mode') else 'Hedge'
            return mode

    @with_timeout(15)  # 15 секунд таймаут для размещения ордера
    def place_order(self, symbol, side, quantity, order_type='market', price=None,
                    take_profit=None, stop_loss=None, max_loss_percent=None, quantity_is_usdt=True,
                    skip_min_notional_enforcement=False):
        """Размещение ордера для бота
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            side (str): Сторона ('BUY', 'SELL', 'LONG', 'SHORT')
            quantity (float): Количество в USDT
            order_type (str): Тип ордера ('market' или 'limit')
            price (float, optional): Цена для лимитного ордера
            take_profit (float, optional): Цена Take Profit
            stop_loss (float, optional): Цена Stop Loss
            max_loss_percent (float, optional): Максимальный убыток в % (если не указана цена стоп-лосса)
            skip_min_notional_enforcement (bool): Если True, выводить специальное предупреждение при увеличении до minNotionalValue
                                                  (используется для лимитных ордеров из набора позиций)
                                                  ВАЖНО: ордер все равно будет увеличен до минимума, иначе биржа отклонит его!
            
        Returns:
            dict: Результат размещения ордера
        """
        try:
            unit_label = "USDT" if quantity_is_usdt else "coins"
            logger.info(f"[BYBIT_BOT] Размещение ордера: {symbol} {side} {quantity} {unit_label} ({order_type})")
            
            # ✅ КРИТИЧНО: Получаем АКТУАЛЬНУЮ цену с биржи ПЕРЕД расчетом ордера!
            # Цена нужна всегда, чтобы правильно рассчитать количество монет и округление
            current_price = None
            try:
                ticker = self.client.get_tickers(category="linear", symbol=f"{symbol}USDT")
                if ticker.get('retCode') == 0 and ticker.get('result', {}).get('list'):
                    current_price = float(ticker['result']['list'][0].get('lastPrice', 0))
                    if current_price and current_price > 0:
                        logger.debug(f"[BYBIT_BOT] 📊 Текущая цена {symbol}: {current_price}")
                    else:
                        raise ValueError("Получена некорректная цена (0 или отрицательная)")
                else:
                    raise ValueError(f"Ошибка API: {ticker.get('retMsg', 'Unknown error')}")
            except Exception as e:
                error_msg = f"❌ Не удалось получить актуальную цену с биржи для {symbol}: {e}"
                logger.error(f"[BYBIT_BOT] {error_msg}")
                return {
                    'success': False,
                    'message': error_msg
                }
            
            # Проверяем что цена получена и валидна
            if not current_price or current_price <= 0:
                error_msg = f"❌ Некорректная цена {symbol}: {current_price}"
                logger.error(f"[BYBIT_BOT] {error_msg}")
                return {
                    'success': False,
                    'message': error_msg
                }
                         
            # ⚠️ ПЛЕЧО НЕ УСТАНАВЛИВАЕТСЯ ЧЕРЕЗ API!
            # Плечо должно быть установлено ВРУЧНУЮ в настройках аккаунта на бирже
                         
            # Конвертируем side для ботов
            if side.upper() == 'LONG':
                bybit_side = 'Buy'
                position_side = 'LONG'
            elif side.upper() == 'SHORT':
                bybit_side = 'Sell'
                position_side = 'SHORT'
            elif side.upper() == 'BUY':
                bybit_side = 'Buy'
                position_side = 'LONG'
            elif side.upper() == 'SELL':
                bybit_side = 'Sell'
                position_side = 'SHORT'
            else:
                return {
                    'success': False,
                    'message': f'Неизвестная сторона ордера: {side}'
                }
            
            # ✅ КРИТИЧНО: Определяем position_idx в зависимости от режима позиции
            # В One-Way Mode: position_idx = 0 (для обеих сторон)
            # В Hedge Mode: position_idx = 1 (LONG), position_idx = 2 (SHORT)
            position_mode = self._get_position_mode(symbol)
            if position_mode == 'One-Way':
                position_idx = 0
                logger.debug(f"[BYBIT_BOT] {symbol}: One-Way mode, используем position_idx=0")
            else:
                # Hedge mode
                if position_side == 'LONG':
                    position_idx = 1
                else:  # SHORT
                    position_idx = 2
                logger.debug(f"[BYBIT_BOT] {symbol}: Hedge mode, используем position_idx={position_idx} для {position_side}")
            
            # ⚡ Для LINEAR фьючерсов используем marketUnit='quoteCoin' для указания суммы в USDT
            # ✅ marketUnit='quoteCoin' работает ТОЛЬКО для MARKET ордеров, НО Bybit проверяет кратность монет!
            
            # ✅ Получаем ПОЛНУЮ информацию об инструменте для ВСЕХ проверок
            instruments_info = None
            min_notional_value = None
            qty_step = None
            min_order_qty = None
            
            try:
                instruments_info = self.get_instruments_info(f"{symbol}USDT")
                if instruments_info:
                    min_notional_value = instruments_info.get('minNotionalValue')
                    qty_step = instruments_info.get('qtyStep')
                    min_order_qty = instruments_info.get('minOrderQty')
                    
                    if min_notional_value:
                        min_notional_value = float(min_notional_value)
                    if qty_step:
                        qty_step = float(qty_step)
                    if min_order_qty:
                        min_order_qty = float(min_order_qty)
                        
                    logger.debug(f"[BYBIT_BOT] 📊 {symbol}: minNotionalValue={min_notional_value} USDT, minOrderQty={min_order_qty}, qtyStep={qty_step}")
            except Exception as e:
                logger.warning(f"[BYBIT_BOT] ⚠️ Не удалось получить информацию об инструменте: {e}")
            
            # ✅ Получаем ТЕКУЩЕЕ плечо для монеты из настроек биржи
            current_leverage = None
            try:
                pos_response = self.client.get_positions(category="linear", symbol=f"{symbol}USDT")
                if pos_response.get('retCode') == 0 and pos_response.get('result', {}).get('list'):
                    # get_positions всегда возвращает leverage даже для пустых позиций!
                    # Берем leverage из первой позиции в списке (она может быть пустой)
                    pos_list = pos_response['result']['list']
                    if pos_list:
                        current_leverage = float(pos_list[0].get('leverage', 10))
                        logger.debug(f"[BYBIT_BOT] 📊 {symbol}: Плечо с биржи: {current_leverage}x")
            except Exception as e:
                logger.warning(f"[BYBIT_BOT] ⚠️ Не удалось получить текущее плечо: {e}")
            
            # Если не удалось - используем дефолтное 10x (НО ЭТО НЕ ДОЛЖНО БЫТЬ!)
            if not current_leverage:
                current_leverage = 10.0
                logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: FALLBACK - используем дефолтное плечо: {current_leverage}x")
            
            qty_in_coins = None
            requested_qty_usdt = None
            if quantity_is_usdt:
                requested_qty_usdt = float(quantity)
                logger.debug(f"[BYBIT_BOT] 🎯 {symbol}: Запрошенная сумма из конфига: {requested_qty_usdt} USDT")
            else:
                qty_in_coins = float(quantity)
                requested_qty_usdt = qty_in_coins * current_price
                logger.debug(f"[BYBIT_BOT] 🎯 {symbol}: Запрошено {qty_in_coins} монет (~{requested_qty_usdt:.4f} USDT)")
            
            # Рассчитываем количество МОНЕТ с учетом кратности qtyStep и minOrderQty
            # Затем передаем монеты в Bybit - он САМ применит плечо!
            if qty_step and current_price and min_order_qty:
                # ✅ ШАГ 1: Считаем сколько МОНЕТ нужно
                # ⚠️ КРИТИЧНО: Для лимитных ордеров используем цену ЛИМИТНОГО ордера для расчета количества монет!
                # Для рыночных ордеров - текущую цену
                # Это важно: чем ниже лимитная цена, тем БОЛЬШЕ монет нужно купить на те же 5 USDT!
                price_for_calculation = price if (order_type.lower() == 'limit' and price) else current_price
                if order_type.lower() == 'limit' and price:
                    logger.debug(f"[BYBIT_BOT] 💡 {symbol}: ЛИМИТНЫЙ ордер - расчет по ЛИМИТНОЙ цене {price_for_calculation:.6f} (текущая: {current_price:.6f})")
                    logger.debug(f"[BYBIT_BOT] 💡 {symbol}: На {requested_qty_usdt} USDT по цене {price_for_calculation:.6f} нужно БОЛЬШЕ монет, чем по текущей {current_price:.6f}")
                requested_coins = requested_qty_usdt / price_for_calculation if quantity_is_usdt else qty_in_coins
                logger.debug(f"[BYBIT_BOT] 🔍 {symbol}: Исходное количество монет: {requested_coins:.2f} (рассчитано по цене {price_for_calculation:.6f}, запрошено {requested_qty_usdt} USDT)")
                
                # ✅ ШАГ 2: Округляем монеты вверх до qtyStep
                rounded_coins = math.ceil(requested_coins / qty_step) * qty_step
                logger.debug(f"[BYBIT_BOT] 🔍 {symbol}: Округлили {requested_coins:.2f} до {rounded_coins} монет (кратно {qty_step})")
                
                # ✅ ШАГ 3: Проверяем minOrderQty - если меньше, берем minOrderQty
                min_coins_for_qty = math.ceil(min_order_qty / qty_step) * qty_step
                if rounded_coins < min_coins_for_qty:
                    rounded_coins = min_coins_for_qty
                    logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Меньше minOrderQty={min_order_qty}, увеличили до {rounded_coins} монет")
                
                # ✅ ШАГ 4: Проверяем minNotionalValue (по номинальной стоимости!)
                # ⚠️ КРИТИЧНО: Биржа Bybit ОТКЛОНЯЕТ ордера меньше minNotionalValue!
                # ⚠️ КРИТИЧНО: Для лимитных ордеров биржа проверяет по цене ЛИМИТНОГО ордера, а не текущей!
                # Для рыночных ордеров - по текущей цене
                price_for_notional_check = price if (order_type.lower() == 'limit' and price) else current_price
                nominal_usdt = rounded_coins * price_for_notional_check
                min_usdt_from_notional = min_notional_value if min_notional_value else 5.0
                
                if nominal_usdt < min_usdt_from_notional:
                    # Если получилось меньше minNotional - ВСЕГДА увеличиваем монеты
                    # Иначе биржа отклонит ордер с ошибкой "Order does not meet minimum order value"
                    # Используем цену для проверки (лимитная цена для лимитных ордеров, текущая для рыночных)
                    min_coins_for_notional = math.ceil(min_usdt_from_notional / price_for_notional_check / qty_step) * qty_step
                    rounded_coins = min_coins_for_notional
                    if skip_min_notional_enforcement:
                        # Для лимитных ордеров из набора - предупреждаем, что увеличили до минимума
                        logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Запрошенная сумма {nominal_usdt:.2f} USDT < minNotionalValue={min_usdt_from_notional} USDT "
                                     f"(по цене {price_for_notional_check:.6f}). "
                                     f"Увеличиваем до минимума {rounded_coins} монет (~{rounded_coins * price_for_notional_check:.2f} USDT), "
                                     f"иначе биржа отклонит ордер (лимитный ордер из набора позиций)")
                    else:
                        # Для обычных ордеров - стандартное предупреждение
                        logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Меньше minNotionalValue={min_usdt_from_notional}, увеличили до {rounded_coins} монет")
                
                qty_in_coins = rounded_coins
                logger.debug(f"[BYBIT_BOT] 💰 {symbol}: ФИНАЛЬНО: {qty_in_coins} монет @ {current_price:.8f} (кратно {qty_step})")
            else:
                # Fallback если нет данных об инструменте
                # Просто пересчитываем USDT в монеты
                qty_in_coins = requested_qty_usdt / current_price if current_price else 0
                logger.debug(f"[BYBIT_BOT] 💰 {symbol}: Fallback: {qty_in_coins:.2f} монет")
            
            # ✅ Передаем количество МОНЕТ без marketUnit='quoteCoin'!
            # Bybit САМ применит плечо при размещении ордера!
            # КРИТИЧНО: Форматируем с точностью до 8 знаков после запятой (стандарт крипты)
            qty_coins_str = f"{qty_in_coins:.8f}".rstrip('0').rstrip('.')
            
            order_params = {
                "category": "linear",
                "symbol": f"{symbol}USDT",
                "side": bybit_side,
                "orderType": order_type.title(),
                "qty": qty_coins_str,  # ✅ Количество в МОНЕТАХ!
                "positionIdx": position_idx
            }
            
            logger.debug(f"[BYBIT_BOT] 🎯 {symbol}: order_params={order_params}")
            logger.debug(f"[BYBIT_BOT] 🔍 {symbol}: ДЕТАЛИ: qty='{qty_coins_str}' монет, orderType='{order_type.title()}'")
            
            # ⚠️ НЕ добавляем leverage в order_params - Bybit не поддерживает это при размещении ордера!
            # Плечо должно быть установлено ВРУЧНУЮ в настройках аккаунта на бирже
            
            # Добавляем цену для лимитных ордеров
            if order_type.lower() == 'limit':
                if price is None and current_price:
                    # Используем текущую цену с небольшим отступом
                    if bybit_side == 'Buy':
                        price = current_price * 0.999  # Покупаем чуть ниже рынка
                    else:
                        price = current_price * 1.001  # Продаем чуть выше рынка
                
                if price:
                    order_params["price"] = str(round(price, 6))  # 6 знаков для поддержки дешевых монет
                    order_params["timeInForce"] = "GTC"
            
            # 🎯 Добавляем Take Profit если указан
            if take_profit is not None and take_profit > 0:
                # Bybit API: takeProfit принимает абсолютную цену (НЕ процент!)
                order_params["takeProfit"] = str(round(take_profit, 6))
                logger.debug(f"[BYBIT_BOT] 🎯 Take Profit установлен: {take_profit:.6f} (цена)")
            
            # 🛑 Добавляем Stop Loss если указан
            if stop_loss is not None and stop_loss > 0:
                # Bybit API: stopLoss принимает абсолютную цену (НЕ процент!)
                order_params["stopLoss"] = str(round(stop_loss, 6))
                logger.debug(f"[BYBIT_BOT] 🛑 Stop Loss установлен: {stop_loss:.6f} (цена)")
            
            logger.debug(f"[BYBIT_BOT] Параметры ордера: {order_params}")
            
            # Размещаем ордер
            logger.debug(f"[BYBIT_BOT] 🔍 {symbol}: ОТПРАВЛЯЕМ ЗАПРОС в Bybit API...")
            try:
                response = self.client.place_order(**order_params)
                logger.debug(f"[BYBIT_BOT] ✅ {symbol}: ПОЛУЧЕН ОТВЕТ от Bybit API: retCode={response.get('retCode')}, retMsg={response.get('retMsg')}")
                logger.debug(f"[BYBIT_BOT] 📊 {symbol}: Полный ответ: {response}")
            except Exception as api_error:
                # Pybit бросает исключение при retCode != 0, но ответ может быть в ошибке!
                logger.error(f"[BYBIT_BOT] ❌ {symbol}: Pybit exception: {api_error}")
                # Пытаемся извлечь ответ из исключения
                error_str = str(api_error)
                import re
                # Извлекаем retCode и retMsg из строки ошибки
                if "retCode" in error_str and "retMsg" in error_str:
                    logger.debug(f"[BYBIT_BOT] 📊 {symbol}: Ошибка содержит информацию об ответе: {error_str}")
                raise api_error  # Пробрасываем дальше
            
            if response['retCode'] == 0:
                # Вычисляем количество в USDT для возврата
                qty_usdt_actual = (qty_in_coins * current_price) if (qty_in_coins and current_price and current_price > 0) else requested_qty_usdt
                logger.info(f"[BYBIT_BOT] ✅ Ордер успешно размещён: {qty_in_coins} монет = {qty_usdt_actual:.4f} USDT @ {current_price}")
                
                return {
                    'success': True,
                    'order_id': response['result']['orderId'],
                    'message': f'{order_type.title()} ордер успешно размещён',
                    'price': price or current_price or 0,
                    'quantity': qty_in_coins,  # Возвращаем количество в монетах
                    'quantity_usdt': qty_usdt_actual  # Возвращаем фактическую сумму в USDT
                }
            else:
                # Извлекаем код ошибки из ответа
                error_code = response.get('retCode', '')
                error_msg = response.get('retMsg', 'unknown error')
                return {
                    'success': False,
                    'message': f"Ошибка размещения ордера: {error_msg}",
                    'error_code': str(error_code)  # Добавляем код ошибки для проверки делистинга
                }
                
        except Exception as e:
            error_str = str(e)
            logger.error(f"[BYBIT_BOT] Ошибка размещения ордера: {error_str}")
            import traceback
            logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
            # Извлекаем код ошибки из строки исключения (если есть)
            error_code = ''
            if 'ErrCode:' in error_str:
                import re
                match = re.search(r'ErrCode:\s*(\d+)', error_str)
                if match:
                    error_code = match.group(1)
            return {
                'success': False,
                'message': f"Ошибка размещения ордера: {error_str}",
                'error_code': error_code  # Добавляем код ошибки для проверки делистинга
            }
    
    @with_timeout(15)  # 15 секунд таймаут для обновления TP
    def update_take_profit(self, symbol, take_profit_price, position_side=None):
        """
        Обновляет Take Profit для существующей позиции
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            take_profit_price (float): Новая цена Take Profit
            position_side (str, optional): Направление позиции ('LONG' или 'SHORT')
            
        Returns:
            dict: Результат обновления TP
        """
        try:
            logger.info(f"[BYBIT_BOT] Обновление Take Profit: {symbol} → {take_profit_price:.6f} (side: {position_side})")
            
            # ✅ КРИТИЧНО: Определяем positionIdx в зависимости от режима позиции
            # В One-Way Mode: position_idx = 0 (для обеих сторон)
            # В Hedge Mode: position_idx = 1 (LONG), position_idx = 2 (SHORT)
            position_mode = self._get_position_mode(symbol)
            if position_mode == 'One-Way':
                position_idx = 0
                logger.debug(f"[BYBIT_BOT] {symbol}: One-Way mode, используем position_idx=0 для TP")
            else:
                # Hedge mode
                if position_side:
                    position_idx = 1 if position_side.upper() == 'LONG' else 2
                else:
                    # Если side не указан, пытаемся определить из позиции
                    position_idx = 0  # Fallback
                    logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Hedge mode, но side не указан, используем position_idx=0")
                logger.debug(f"[BYBIT_BOT] {symbol}: Hedge mode, используем position_idx={position_idx} для TP")
            
            # Параметры для обновления TP (используем Trading Stop API)
            tp_params = {
                "category": "linear",
                "symbol": f"{symbol}USDT",
                "takeProfit": str(round(take_profit_price, 6)),
                "positionIdx": position_idx
            }
            
            logger.info(f"[BYBIT_BOT] Параметры TP: {tp_params}")
            
            # Обновляем TP через API - используем метод set_trading_stop
            try:
                response = self.client.set_trading_stop(**tp_params)
                logger.debug(f"[BYBIT_BOT] Ответ API TP: {response}")
                
                if response['retCode'] == 0:
                    return {
                        'success': True,
                        'message': f'Take Profit обновлен: {take_profit_price:.6f}',
                        'take_profit': take_profit_price
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Ошибка обновления TP: {response['retMsg']}"
                    }
            except Exception as e:
                # Проверяем код ошибки 34040 (not modified) - это нормально, TP уже установлен
                error_str = str(e)
                if "34040" in error_str or "not modified" in error_str:
                    logger.info(f"[BYBIT_BOT] ✅ TP уже установлен на {take_profit_price:.6f}")
                    return {
                        'success': True,
                        'message': f'Take Profit уже установлен: {take_profit_price:.6f}',
                        'take_profit': take_profit_price
                    }
                
                # Для других ошибок - логируем и возвращаем ошибку
                logger.error(f"[BYBIT_BOT] Ошибка обновления Take Profit: {e}")
                import traceback
                logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
                return {
                    'success': False,
                    'message': f"Ошибка обновления TP: {error_str}"
                }
            except AttributeError:
                # Если метод set_trading_stop не существует, пробуем альтернативный способ
                logger.warning(f"[BYBIT_BOT] ⚠️ Метод set_trading_stop не найден, используем альтернативный способ")
                # Пока просто логируем - TP будет установлен при открытии позиции
                return {
                    'success': False,
                    'message': f"Метод set_trading_stop не поддерживается"
                }
                
        except Exception as e:
            logger.error(f"[BYBIT_BOT] Ошибка обновления Take Profit: {str(e)}")
            import traceback
            logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Ошибка обновления TP: {str(e)}"
            }
    
    def place_stop_loss(self, symbol, side, entry_price, loss_percent):
        """
        Совместимость со старым API: рассчитывает цену SL и делегирует на update_stop_loss
        """
        try:
            if not entry_price or entry_price <= 0 or not loss_percent:
                return {'success': False, 'message': 'Некорректные параметры SL'}
            
            side = (side or '').upper()
            loss_percent = float(loss_percent)
            
            if side == 'LONG':
                stop_price = entry_price * (1 - loss_percent / 100.0)
            elif side == 'SHORT':
                stop_price = entry_price * (1 + loss_percent / 100.0)
            else:
                return {'success': False, 'message': f'Неизвестная сторона позиции: {side}'}
            
            stop_price = round(float(stop_price), 6)
            return self.update_stop_loss(symbol, stop_price, side)
        except Exception as exc:
            logger.error(f"[BYBIT_BOT] Ошибка place_stop_loss: {exc}")
            return {'success': False, 'message': str(exc)}

    @with_timeout(15)  # 15 секунд таймаут для обновления SL
    def update_stop_loss(self, symbol, stop_loss_price, position_side=None):
        """
        Обновляет Stop Loss для существующей позиции (программный трейлинг)
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            stop_loss_price (float): Новая цена Stop Loss
            position_side (str, optional): Направление позиции ('LONG' или 'SHORT')
            
        Returns:
            dict: Результат обновления SL
        """
        try:
            logger.info(f"[BYBIT_BOT] Обновление Stop Loss: {symbol} → {stop_loss_price:.6f} (side: {position_side})")
            
            # ✅ КРИТИЧНО: Определяем positionIdx в зависимости от режима позиции
            # В One-Way Mode: position_idx = 0 (для обеих сторон)
            # В Hedge Mode: position_idx = 1 (LONG), position_idx = 2 (SHORT)
            position_mode = self._get_position_mode(symbol)
            if position_mode == 'One-Way':
                position_idx = 0
                logger.debug(f"[BYBIT_BOT] {symbol}: One-Way mode, используем position_idx=0 для SL")
            else:
                # Hedge mode
                if position_side:
                    position_idx = 1 if position_side.upper() == 'LONG' else 2
                else:
                    # Если side не указан, пытаемся определить из позиции
                    position_idx = 0  # Fallback
                    logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Hedge mode, но side не указан, используем position_idx=0")
                logger.debug(f"[BYBIT_BOT] {symbol}: Hedge mode, используем position_idx={position_idx} для SL")
            
            # Параметры для обновления SL (используем Trading Stop API)
            sl_params = {
                "category": "linear",
                "symbol": f"{symbol}USDT",
                "stopLoss": str(round(stop_loss_price, 6)),
                "positionIdx": position_idx
            }
            
            logger.info(f"[BYBIT_BOT] Параметры SL: {sl_params}")
            
            # Обновляем SL через API - используем метод set_trading_stop
            try:
                response = self.client.set_trading_stop(**sl_params)
                logger.debug(f"[BYBIT_BOT] Ответ API SL: {response}")
                
                if response['retCode'] == 0:
                    return {
                        'success': True,
                        'message': f'Stop Loss обновлен: {stop_loss_price:.6f}',
                        'stop_loss': stop_loss_price
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Ошибка обновления SL: {response['retMsg']}"
                    }
            except Exception as e:
                # Проверяем код ошибки 34040 (not modified) - это нормально, SL уже установлен
                error_str = str(e)
                if "34040" in error_str or "not modified" in error_str:
                    logger.info(f"[BYBIT_BOT] ✅ SL уже установлен на {stop_loss_price:.6f}")
                    return {
                        'success': True,
                        'message': f'Stop Loss уже установлен: {stop_loss_price:.6f}',
                        'stop_loss': stop_loss_price
                    }
                
                # Для других ошибок - логируем и возвращаем ошибку
                logger.error(f"[BYBIT_BOT] Ошибка обновления Stop Loss: {e}")
                import traceback
                logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
                return {
                    'success': False,
                    'message': f"Ошибка обновления SL: {error_str}"
                }
            except AttributeError:
                # Если метод set_trading_stop не существует
                logger.warning(f"[BYBIT_BOT] ⚠️ Метод set_trading_stop не найден")
                return {
                    'success': False,
                    'message': f"Метод set_trading_stop не поддерживается"
                }
                
        except Exception as e:
            logger.error(f"[BYBIT_BOT] Ошибка обновления Stop Loss: {str(e)}")
            import traceback
            logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Ошибка обновления SL: {str(e)}"
            }
    
    @with_timeout(15)  # 15 секунд таймаут для установки SL по ROI
    def update_stop_loss_by_roi(self, symbol, roi_percent, position_side=None):
        """
        Устанавливает Stop Loss по ROI (% потери от маржи)
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            roi_percent (float): ROI в % (например, -15.0 для потери 15%)
            position_side (str, optional): Направление позиции ('LONG' или 'SHORT')
            
        Returns:
            dict: Результат установки SL
        """
        try:
            logger.info(f"[BYBIT_BOT] Установка Stop Loss по ROI: {symbol} → {roi_percent}% (side: {position_side})")
            
            # ✅ КРИТИЧНО: Определяем positionIdx в зависимости от режима позиции
            # В One-Way Mode: position_idx = 0 (для обеих сторон)
            # В Hedge Mode: position_idx = 1 (LONG), position_idx = 2 (SHORT)
            position_mode = self._get_position_mode(symbol)
            if position_mode == 'One-Way':
                position_idx = 0
                logger.debug(f"[BYBIT_BOT] {symbol}: One-Way mode, используем position_idx=0 для SL по ROI")
            else:
                # Hedge mode
                if position_side:
                    position_idx = 1 if position_side.upper() == 'LONG' else 2
                else:
                    # Если side не указан, пытаемся определить из позиции
                    position_idx = 0  # Fallback
                    logger.warning(f"[BYBIT_BOT] ⚠️ {symbol}: Hedge mode, но side не указан, используем position_idx=0")
                logger.debug(f"[BYBIT_BOT] {symbol}: Hedge mode, используем position_idx={position_idx} для SL по ROI")
            
            # Параметры для установки SL по ROI
            # Bybit API: slSize - размер стопа в % (отрицательный для стоп-лосса)
            sl_params = {
                "category": "linear",
                "symbol": f"{symbol}USDT",
                "slTriggerBy": "LastPrice",  # Триггер по последней цене
                "slSize": str(roi_percent),  # ROI в % (например, "-15.0")
                "positionIdx": position_idx
            }
            
            logger.info(f"[BYBIT_BOT] Параметры SL по ROI: {sl_params}")
            
            # Устанавливаем SL через API - используем метод set_trading_stop
            try:
                response = self.client.set_trading_stop(**sl_params)
                logger.debug(f"[BYBIT_BOT] Ответ API SL по ROI: {response}")
                
                if response['retCode'] == 0:
                    return {
                        'success': True,
                        'message': f'Stop Loss установлен по ROI: {roi_percent}%',
                        'roi_percent': roi_percent
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Ошибка установки SL: {response['retMsg']}"
                    }
            except Exception as e:
                # Проверяем код ошибки 34040 (not modified) - это нормально, SL уже установлен
                error_str = str(e)
                if "34040" in error_str or "not modified" in error_str:
                    logger.info(f"[BYBIT_BOT] ✅ SL уже установлен на {roi_percent}%")
                    return {
                        'success': True,
                        'message': f'Stop Loss уже установлен по ROI: {roi_percent}%',
                        'roi_percent': roi_percent
                    }
                
                # Для других ошибок - логируем и возвращаем ошибку
                logger.error(f"[BYBIT_BOT] Ошибка установки Stop Loss по ROI: {e}")
                import traceback
                logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
                return {
                    'success': False,
                    'message': f"Ошибка установки SL: {error_str}"
                }
            except AttributeError:
                # Если метод set_trading_stop не существует
                logger.warning(f"[BYBIT_BOT] ⚠️ Метод set_trading_stop не найден")
                return {
                    'success': False,
                    'message': f"Метод set_trading_stop не поддерживается"
                }
                
        except Exception as e:
            logger.error(f"[BYBIT_BOT] Ошибка установки Stop Loss по ROI: {str(e)}")
            import traceback
            logger.error(f"[BYBIT_BOT] Трейсбек: {traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Ошибка установки SL: {str(e)}"
            }
    
    def get_open_orders(self, symbol):
        """
        Получает список открытых ордеров для символа
        
        Args:
            symbol (str): Символ торговой пары (без USDT)
            
        Returns:
            list: Список открытых ордеров
        """
        try:
            # Используем API Bybit для получения открытых ордеров
            response = self.client.get_open_orders(
                category="linear",
                symbol=f"{symbol}USDT",
                limit=50
            )
            
            if response.get('retCode') == 0:
                orders = response.get('result', {}).get('list', [])
                # Преобразуем формат для единообразия
                formatted_orders = []
                for order in orders:
                    order_type = order.get('orderType', '').lower()  # 'Limit' или 'Market'
                    formatted_orders.append({
                        'order_id': order.get('orderId', ''),
                        'orderId': order.get('orderId', ''),
                        'id': order.get('orderId', ''),
                        'symbol': order.get('symbol', '').replace('USDT', ''),
                        'side': order.get('side', ''),
                        'order_type': order_type,  # Добавляем тип ордера
                        'price': float(order.get('price', 0)),
                        'quantity': float(order.get('qty', 0)),
                        'status': order.get('orderStatus', '')
                    })
                return formatted_orders
            else:
                logger.warning(f"[BYBIT_BOT] ⚠️ Не удалось получить открытые ордера для {symbol}: {response.get('retMsg', 'unknown error')}")
                return []
        
        except Exception as e:
            logger.error(f"[BYBIT_BOT] ❌ Ошибка получения открытых ордеров для {symbol}: {e}")
            return []
    
    def set_leverage(self, symbol, leverage):
        """
        Устанавливает кредитное плечо для символа
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC')
            leverage (int): Значение плеча (например, 5 для x5)
            
        Returns:
            dict: Результат установки плеча с полями:
                - success (bool): Успешность операции
                - message (str): Сообщение о результате
        """
        try:
            # Проверяем валидность плеча
            leverage = int(leverage)
            if leverage < 1 or leverage > 125:
                return {
                    'success': False,
                    'message': f'Недопустимое значение плеча: {leverage}. Допустимый диапазон: 1-125'
                }
            
            # Получаем текущее плечо
            current_leverage = None
            try:
                pos_response = self.client.get_positions(category="linear", symbol=f"{symbol}USDT")
                if pos_response.get('retCode') == 0 and pos_response.get('result', {}).get('list'):
                    pos_list = pos_response['result']['list']
                    if pos_list:
                        current_leverage = float(pos_list[0].get('leverage', 10))
            except Exception as e:
                logger.warning(f"[BYBIT_BOT] ⚠️ Не удалось получить текущее плечо: {e}")
            
            # Если плечо уже установлено на нужное значение, пропускаем
            if current_leverage and int(current_leverage) == leverage:
                logger.debug(f"[BYBIT_BOT] ✅ {symbol}: Плечо уже установлено на {leverage}x")
                return {
                    'success': True,
                    'message': f'Плечо уже установлено на {leverage}x'
                }
            
            # Устанавливаем плечо через API Bybit
            response = self.client.set_leverage(
                category="linear",
                symbol=f"{symbol}USDT",
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response.get('retCode') == 0:
                logger.info(f"[BYBIT_BOT] ✅ {symbol}: Плечо установлено на {leverage}x")
                return {
                    'success': True,
                    'message': f'Плечо успешно установлено на {leverage}x'
                }
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error(f"[BYBIT_BOT] ❌ {symbol}: Ошибка установки плеча: {error_msg}")
                return {
                    'success': False,
                    'message': f'Ошибка установки плеча: {error_msg}'
                }
                
        except Exception as e:
            logger.error(f"[BYBIT_BOT] ❌ {symbol}: Ошибка установки плеча: {e}")
            return {
                'success': False,
                'message': f'Ошибка установки плеча: {str(e)}'
            }