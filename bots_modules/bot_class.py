"""Класс торгового бота NewTradingBot

Основной класс для управления торговым ботом с поддержкой:
- Автоматического открытия/закрытия позиций
- Проверки фильтров (RSI time filter, trend, maturity)
- Защитных механизмов (trailing stop, break-even)
"""

import logging
from datetime import datetime
import time
import threading

logger = logging.getLogger('BotsService')

# Импортируем глобальные переменные
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
        BOT_STATUS, get_exchange, system_initialized
    )
except ImportError:
    # Fallback если импорт не удался
    bots_data_lock = threading.Lock()
    bots_data = {}
    rsi_data_lock = threading.Lock()
    coins_rsi_data = {}
    BOT_STATUS = {
        'IDLE': 'idle',
        'RUNNING': 'running',
        'IN_POSITION_LONG': 'in_position_long',
        'IN_POSITION_SHORT': 'in_position_short',
        'WAITING': 'waiting',
        'STOPPED': 'stopped',
        'ERROR': 'error',
        'PAUSED': 'paused'
    }
    def get_exchange():
        return None
    system_initialized = False

# Импорт функций фильтров (будут доступны после импорта)
try:
    from bots_modules.filters import check_rsi_time_filter
except:
    def check_rsi_time_filter(*args, **kwargs):
        return {'allowed': True, 'reason': 'Filter not loaded'}

class NewTradingBot:
    """Новый торговый бот согласно требованиям"""
    
    def __init__(self, symbol, config=None, exchange=None):
        self.symbol = symbol
        self.config = config or {}
        self.exchange = exchange
        
        logger.info(f"[NEW_BOT_{symbol}] 🤖 Инициализация нового торгового бота")
        
        # Параметры сделки из конфига
        self.volume_mode = self.config.get('volume_mode', 'usdt')
        self.volume_value = self.config.get('volume_value', 10.0)
        
        # Состояние бота
        self.status = self.config.get('status', BOT_STATUS['IDLE'])
        self.entry_price = self.config.get('entry_price', None)
        self.position_side = self.config.get('position_side', None)
        self.unrealized_pnl = self.config.get('unrealized_pnl', 0.0)
        self.created_at = self.config.get('created_at', datetime.now().isoformat())
        self.last_signal_time = self.config.get('last_signal_time', None)
        
        # Защитные механизмы
        self.max_profit_achieved = self.config.get('max_profit_achieved', 0.0)
        self.trailing_stop_price = self.config.get('trailing_stop_price', None)
        self.break_even_activated = bool(self.config.get('break_even_activated', False))
        
        # Время входа в позицию
        position_start_str = self.config.get('position_start_time', None)
        if position_start_str:
            try:
                self.position_start_time = datetime.fromisoformat(position_start_str)
            except:
                self.position_start_time = None
        else:
            self.position_start_time = None
        
        # Отслеживание позиций
        self.order_id = self.config.get('order_id', None)
        self.entry_timestamp = self.config.get('entry_timestamp', None)
        self.opened_by_autobot = self.config.get('opened_by_autobot', False)
        
        logger.info(f"[NEW_BOT_{symbol}] ✅ Бот инициализирован (статус: {self.status})")
        
    def update_status(self, new_status, entry_price=None, position_side=None):
        """Обновляет статус бота"""
        old_status = self.status
        self.status = new_status
        
        if entry_price is not None:
            self.entry_price = entry_price
        if position_side is not None:
            self.position_side = position_side
            
        # Инициализируем защитные механизмы при входе в позицию
        if new_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
            self.position_start_time = datetime.now()
            self.max_profit_achieved = 0.0
            self.trailing_stop_price = None
            self.break_even_activated = False
            
        logger.info(f"[NEW_BOT_{self.symbol}] 📊 Статус изменен: {old_status} → {new_status}")
    
    def should_open_long(self, rsi, trend, candles):
        """Проверяет, нужно ли открывать LONG позицию"""
        try:
            # Получаем настройки из конфига
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)
                avoid_down_trend = auto_config.get('avoid_down_trend', True)
                rsi_time_filter_enabled = auto_config.get('rsi_time_filter_enabled', True)
                rsi_time_filter_candles = auto_config.get('rsi_time_filter_candles', 8)
                rsi_time_filter_lower = auto_config.get('rsi_time_filter_lower', 35)
            
            # 1. Проверка RSI
            if rsi > rsi_long_threshold:
                logger.debug(f"[NEW_BOT_{self.symbol}] ❌ RSI {rsi:.1f} > {rsi_long_threshold} - не открываем LONG")
                return False
            
            # 2. Проверка тренда
            if avoid_down_trend and trend == 'DOWN':
                logger.debug(f"[NEW_BOT_{self.symbol}] ❌ DOWN тренд - не открываем LONG")
                return False
            
            # 3. RSI временной фильтр
            if rsi_time_filter_enabled:
                time_filter_result = self.check_rsi_time_filter_for_long(candles, rsi, rsi_time_filter_candles, rsi_time_filter_lower)
                if not time_filter_result['allowed']:
                    logger.info(f"[NEW_BOT_{self.symbol}] ❌ RSI Time Filter блокирует LONG: {time_filter_result['reason']}")
                    return False
            
            logger.info(f"[NEW_BOT_{self.symbol}] ✅ Все проверки пройдены - открываем LONG (RSI: {rsi:.1f}, Trend: {trend})")
            return True
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки LONG: {e}")
            return False
    
    def should_open_short(self, rsi, trend, candles):
        """Проверяет, нужно ли открывать SHORT позицию"""
        try:
            # Получаем настройки из конфига
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)
                avoid_up_trend = auto_config.get('avoid_up_trend', True)
                rsi_time_filter_enabled = auto_config.get('rsi_time_filter_enabled', True)
                rsi_time_filter_candles = auto_config.get('rsi_time_filter_candles', 8)
                rsi_time_filter_upper = auto_config.get('rsi_time_filter_upper', 65)
            
            # 1. Проверка RSI
            if rsi < rsi_short_threshold:
                logger.debug(f"[NEW_BOT_{self.symbol}] ❌ RSI {rsi:.1f} < {rsi_short_threshold} - не открываем SHORT")
                return False
            
            # 2. Проверка тренда
            if avoid_up_trend and trend == 'UP':
                logger.debug(f"[NEW_BOT_{self.symbol}] ❌ UP тренд - не открываем SHORT")
                return False
            
            # 3. RSI временной фильтр
            if rsi_time_filter_enabled:
                time_filter_result = self.check_rsi_time_filter_for_short(candles, rsi, rsi_time_filter_candles, rsi_time_filter_upper)
                if not time_filter_result['allowed']:
                    logger.info(f"[NEW_BOT_{self.symbol}] ❌ RSI Time Filter блокирует SHORT: {time_filter_result['reason']}")
                    return False
            
            logger.info(f"[NEW_BOT_{self.symbol}] ✅ Все проверки пройдены - открываем SHORT (RSI: {rsi:.1f}, Trend: {trend})")
            return True
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки SHORT: {e}")
            return False
    
    def check_rsi_time_filter_for_long(self, candles, rsi, filter_candles, filter_lower):
        """Проверяет RSI временной фильтр для LONG (использует сложную логику)"""
        try:
            # Используем старую сложную логику временного фильтра
            return check_rsi_time_filter(candles, rsi, 'ENTER_LONG')
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка RSI Time Filter для LONG: {e}")
            return {'allowed': False, 'reason': f'Ошибка анализа: {str(e)}'}
    
    def check_rsi_time_filter_for_short(self, candles, rsi, filter_candles, filter_upper):
        """Проверяет RSI временной фильтр для SHORT (использует сложную логику)"""
        try:
            # Используем старую сложную логику временного фильтра
            return check_rsi_time_filter(candles, rsi, 'ENTER_SHORT')
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка RSI Time Filter для SHORT: {e}")
            return {'allowed': False, 'reason': f'Ошибка анализа: {str(e)}'}
    
    def should_close_long(self, rsi, current_price):
        """Проверяет, нужно ли закрывать LONG позицию"""
        try:
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                rsi_long_exit = auto_config.get('rsi_long_exit', 65)
            
            if rsi >= rsi_long_exit:
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Закрываем LONG: RSI {rsi:.1f} >= {rsi_long_exit}")
                return True, 'RSI_EXIT'
            
            return False, None
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки закрытия LONG: {e}")
            return False, None
    
    def should_close_short(self, rsi, current_price):
        """Проверяет, нужно ли закрывать SHORT позицию"""
        try:
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                rsi_short_exit = auto_config.get('rsi_short_exit', 35)
            
            if rsi <= rsi_short_exit:
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Закрываем SHORT: RSI {rsi:.1f} <= {rsi_short_exit}")
                return True, 'RSI_EXIT'
            
            return False, None
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки закрытия SHORT: {e}")
            return False, None
    
    def update(self, force_analysis=False, external_signal=None, external_trend=None):
        """Основной метод обновления бота"""
        try:
            if not self.exchange:
                logger.warning(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return {'success': False, 'error': 'Exchange not initialized'}
            
            # Получаем текущие данные
            current_price = None
            current_rsi = None
            current_trend = external_trend
            
            # Получаем RSI данные
            try:
                # Проверяем, определен ли rsi_data_lock
                if 'rsi_data_lock' in globals():
                    with rsi_data_lock:
                        coin_data = coins_rsi_data['coins'].get(self.symbol)
                        if coin_data:
                            current_rsi = coin_data.get('rsi6h')
                            current_price = coin_data.get('price')
                            if not current_trend:
                                current_trend = coin_data.get('trend6h', 'NEUTRAL')
                else:
                    # Fallback если lock не определен
                    coin_data = coins_rsi_data['coins'].get(self.symbol)
                    if coin_data:
                        current_rsi = coin_data.get('rsi6h')
                        current_price = coin_data.get('price')
                        if not current_trend:
                            current_trend = coin_data.get('trend6h', 'NEUTRAL')
            except Exception as e:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка получения RSI данных: {e}")
                # Fallback если lock не определен
                coin_data = coins_rsi_data['coins'].get(self.symbol)
                if coin_data:
                    current_rsi = coin_data.get('rsi6h')
                    current_price = coin_data.get('price')
                    if not current_trend:
                        current_trend = coin_data.get('trend6h', 'NEUTRAL')
            
            if current_rsi is None or current_price is None:
                logger.warning(f"[NEW_BOT_{self.symbol}] ❌ Нет RSI данных")
                return {'success': False, 'error': 'No RSI data'}
            
            # Получаем свечи для анализа
            chart_response = self.exchange.get_chart_data(self.symbol, '6h', '30d')
            if not chart_response or not chart_response.get('success'):
                logger.warning(f"[NEW_BOT_{self.symbol}] ❌ Не удалось получить свечи")
                return {'success': False, 'error': 'No candles data'}
            
            candles = chart_response.get('data', {}).get('candles', [])
            if not candles:
                logger.warning(f"[NEW_BOT_{self.symbol}] ❌ Нет свечей")
                return {'success': False, 'error': 'Empty candles'}
            
            # Обрабатываем в зависимости от статуса
            if self.status == BOT_STATUS['IDLE']:
                return self._handle_idle_state(current_rsi, current_trend, candles, current_price)
            elif self.status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                return self._handle_position_state(current_rsi, current_trend, candles, current_price)
            else:
                logger.debug(f"[NEW_BOT_{self.symbol}] ⏳ Статус {self.status} - ждем")
                return {'success': True, 'status': self.status}
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_idle_state(self, rsi, trend, candles, price):
        """Обрабатывает состояние IDLE (ожидание сигнала)"""
        try:
            # Проверяем, включен ли автобот
            with bots_data_lock:
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            if not auto_bot_enabled:
                logger.debug(f"[NEW_BOT_{self.symbol}] ⏹️ Автобот выключен - не открываем позицию")
                return {'success': True, 'status': self.status}
            
            # Проверяем возможность открытия LONG
            if self.should_open_long(rsi, trend, candles):
                logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Открываем LONG позицию (RSI: {rsi:.1f})")
                if self._open_position_on_exchange('LONG', price):
                    self.update_status(BOT_STATUS['IN_POSITION_LONG'], price, 'LONG')
                    return {'success': True, 'action': 'OPEN_LONG', 'status': self.status}
            else:
                    logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть LONG позицию")
                    return {'success': False, 'error': 'Failed to open LONG position'}
            
            # Проверяем возможность открытия SHORT
            if self.should_open_short(rsi, trend, candles):
                logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Открываем SHORT позицию (RSI: {rsi:.1f})")
                if self._open_position_on_exchange('SHORT', price):
                    self.update_status(BOT_STATUS['IN_POSITION_SHORT'], price, 'SHORT')
                    return {'success': True, 'action': 'OPEN_SHORT', 'status': self.status}
                else:
                    logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть SHORT позицию")
                    return {'success': False, 'error': 'Failed to open SHORT position'}
            
            logger.debug(f"[NEW_BOT_{self.symbol}] ⏳ Ждем сигнал (RSI: {rsi:.1f}, Trend: {trend})")
            return {'success': True, 'status': self.status}
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка в idle состоянии: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_position_state(self, rsi, trend, candles, price):
        """Обрабатывает состояние в позиции"""
        try:
            if not self.entry_price:
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Нет цены входа - обновляем из биржи")
                self._sync_position_with_exchange()
            
            # 1. Проверяем защитные механизмы
            protection_result = self.check_protection_mechanisms(price)
            if protection_result['should_close']:
                logger.info(f"[NEW_BOT_{self.symbol}] 🛡️ Закрываем позицию: {protection_result['reason']}")
                self._close_position_on_exchange(protection_result['reason'])
                return {'success': True, 'action': f"CLOSE_{self.position_side}", 'reason': protection_result['reason']}
            
            # 2. Проверяем условия закрытия по RSI
            if self.position_side == 'LONG':
                should_close, reason = self.should_close_long(rsi, price)
                if should_close:
                    logger.info(f"[NEW_BOT_{self.symbol}] 🔴 Закрываем LONG позицию: {reason}")
                    self._close_position_on_exchange(reason)
                    return {'success': True, 'action': 'CLOSE_LONG', 'reason': reason}
            
            elif self.position_side == 'SHORT':
                should_close, reason = self.should_close_short(rsi, price)
                if should_close:
                    logger.info(f"[NEW_BOT_{self.symbol}] 🔴 Закрываем SHORT позицию: {reason}")
                    self._close_position_on_exchange(reason)
                    return {'success': True, 'action': 'CLOSE_SHORT', 'reason': reason}
            
            # 3. Обновляем защитные механизмы
            self._update_protection_mechanisms(price)
            
            logger.debug(f"[NEW_BOT_{self.symbol}] 📊 В позиции {self.position_side} (RSI: {rsi:.1f}, Цена: {price})")
            return {'success': True, 'status': self.status, 'position_side': self.position_side}
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка в позиции: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_protection_mechanisms(self, current_price):
        """Проверяет все защитные механизмы"""
        try:
            if not self.entry_price or not current_price:
                return {'should_close': False, 'reason': None}
            
            # Получаем настройки из конфига
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                stop_loss_percent = auto_config.get('stop_loss_percent', 15.0)
                trailing_activation_percent = auto_config.get('trailing_activation_percent', 300.0)
                trailing_distance_percent = auto_config.get('trailing_distance_percent', 150.0)
                break_even_trigger_percent = auto_config.get('break_even_trigger_percent', 100.0)
            
            # Вычисляем текущую прибыль в процентах
            if self.position_side == 'LONG':
                profit_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            else:  # SHORT
                profit_percent = ((self.entry_price - current_price) / self.entry_price) * 100
            
            # 1. Проверка стоп-лосса
            if profit_percent <= -stop_loss_percent:
                logger.warning(f"[NEW_BOT_{self.symbol}] 💀 Стоп-лосс! Убыток: {profit_percent:.2f}%")
                return {'should_close': True, 'reason': f'STOP_LOSS_{profit_percent:.2f}%'}
            
            # 2. Обновляем максимальную прибыль
            if profit_percent > self.max_profit_achieved:
                self.max_profit_achieved = profit_percent
                logger.debug(f"[NEW_BOT_{self.symbol}] 📈 Новая максимальная прибыль: {profit_percent:.2f}%")
            
            # 3. Проверка безубыточности
            if not self.break_even_activated and profit_percent >= break_even_trigger_percent:
                self.break_even_activated = True
                logger.info(f"[NEW_BOT_{self.symbol}] 🛡️ Активирована защита безубыточности при {profit_percent:.2f}%")
            
            if self.break_even_activated and profit_percent <= 0:
                logger.info(f"[NEW_BOT_{self.symbol}] 🛡️ Закрываем по безубыточности (было {self.max_profit_achieved:.2f}%, сейчас {profit_percent:.2f}%)")
                return {'should_close': True, 'reason': f'BREAK_EVEN_MAX_{self.max_profit_achieved:.2f}%'}
            
            # 4. Проверка trailing stop
            if self.max_profit_achieved >= trailing_activation_percent:
                # Рассчитываем trailing stop цену
                if self.position_side == 'LONG':
                    # Для LONG trailing stop ниже максимальной цены
                    max_price = self.entry_price * (1 + self.max_profit_achieved / 100)
                    trailing_stop = max_price * (1 - trailing_distance_percent / 100)
                    
                    if current_price <= trailing_stop:
                        logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Trailing Stop! Макс: {self.max_profit_achieved:.2f}%, Текущ: {profit_percent:.2f}%")
                        return {'should_close': True, 'reason': f'TRAILING_STOP_MAX_{self.max_profit_achieved:.2f}%'}
                else:  # SHORT
                    # Для SHORT trailing stop выше минимальной цены
                    min_price = self.entry_price * (1 - self.max_profit_achieved / 100)
                    trailing_stop = min_price * (1 + trailing_distance_percent / 100)
                    
                    if current_price >= trailing_stop:
                        logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Trailing Stop! Макс: {self.max_profit_achieved:.2f}%, Текущ: {profit_percent:.2f}%")
                        return {'should_close': True, 'reason': f'TRAILING_STOP_MAX_{self.max_profit_achieved:.2f}%'}
            
            return {'should_close': False, 'reason': None}
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки защитных механизмов: {e}")
            return {'should_close': False, 'reason': None}
    
    def _update_protection_mechanisms(self, current_price):
        """Обновляет защитные механизмы"""
        try:
            if not self.entry_price or not current_price:
                return
            
            # Вычисляем текущую прибыль
            if self.position_side == 'LONG':
                profit_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            else:  # SHORT
                profit_percent = ((self.entry_price - current_price) / self.entry_price) * 100
            
            # Обновляем максимальную прибыль
            if profit_percent > self.max_profit_achieved:
                self.max_profit_achieved = profit_percent
                logger.debug(f"[NEW_BOT_{self.symbol}] 📈 Обновлена максимальная прибыль: {profit_percent:.2f}%")
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления защитных механизмов: {e}")
    
    def _sync_position_with_exchange(self):
        """Синхронизирует данные бота с позицией на бирже"""
        try:
            if not self.exchange:
                return
            
            exchange_positions = self.exchange.get_positions()
            if isinstance(exchange_positions, tuple):
                positions_list = exchange_positions[0] if exchange_positions else []
            else:
                positions_list = exchange_positions if exchange_positions else []
            
            for pos in positions_list:
                if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                    self.entry_price = float(pos.get('entry_price', 0))
                    self.position_side = pos.get('side', 'UNKNOWN')
                    self.unrealized_pnl = float(pos.get('unrealized_pnl', 0))
                    logger.info(f"[NEW_BOT_{self.symbol}] 🔄 Синхронизировано с биржей: {self.position_side} @ {self.entry_price}")
                    break
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка синхронизации с биржей: {e}")
    
    def enter_position(self, direction):
        """
        Публичный метод для входа в позицию
        
        Args:
            direction (str): Направление сделки ('LONG' или 'SHORT')
            
        Returns:
            bool: True если вход успешен, False иначе
        """
        try:
            # Получаем текущую цену
            ticker = self.exchange.get_ticker(self.symbol) if self.exchange else None
            price = ticker['last'] if ticker and 'last' in ticker else 0
            
            logger.info(f"[NEW_BOT_{self.symbol}] 📈 Входим в {direction} позицию @ {price}")
            
            # Открываем позицию
            if self._open_position_on_exchange(direction, price):
                # Обновляем статус
                status_key = 'IN_POSITION_LONG' if direction == 'LONG' else 'IN_POSITION_SHORT'
                self.update_status(BOT_STATUS[status_key], price, direction)
                
                # Сохраняем состояние
                with bots_data_lock:
                    bots_data['bots'][self.symbol] = self.to_dict()
                
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Вход в {direction} позицию успешен")
                return True
            else:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось войти в {direction} позицию")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка входа в позицию: {e}")
            return False
    
    def _open_position_on_exchange(self, side, price):
        """Открывает позицию на бирже"""
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return False
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Открываем позицию {side} @ {price}")
            
            # Открываем позицию на бирже
            # Рассчитываем количество монет на основе volume_value в USDT
            qty_in_coins = self.volume_value / price if price > 0 else 0
            
            # 🎯 Рассчитываем Take Profit на основе RSI
            take_profit_price = None
            try:
                # Получаем текущий RSI
                if 'rsi_data_lock' in globals():
                    with rsi_data_lock:
                        coin_data = coins_rsi_data['coins'].get(self.symbol)
                        current_rsi = coin_data.get('rsi6h') if coin_data else None
                else:
                    coin_data = coins_rsi_data['coins'].get(self.symbol)
                    current_rsi = coin_data.get('rsi6h') if coin_data else None
                
                logger.info(f"[NEW_BOT_{self.symbol}] 🔍 TP SETUP: current_rsi={current_rsi}, side={side}, price={price}")
                
                if current_rsi:
                    take_profit_price = self.calculate_dynamic_take_profit(side, price, current_rsi)
                    if take_profit_price:
                        logger.info(f"[NEW_BOT_{self.symbol}] ✅ TP рассчитан и будет установлен: {price:.6f} → {take_profit_price:.6f}")
                    else:
                        logger.warning(f"[NEW_BOT_{self.symbol}] ❌ TP НЕ рассчитан - функция вернула None")
                else:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ❌ TP НЕ рассчитан - нет RSI данных")
            except Exception as tp_error:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка расчета TP: {tp_error}")
                import traceback
                traceback.print_exc()
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🚀 ОТПРАВЛЯЕМ ОРДЕР: symbol={self.symbol}, side={side}, quantity={qty_in_coins}, take_profit={take_profit_price}")
            
            order_result = self.exchange.place_order(
                symbol=self.symbol,
                side=side,
                quantity=qty_in_coins,  # Количество в монетах
                order_type='market',
                take_profit=take_profit_price  # ✅ Передаем TP
            )
            
            if order_result and order_result.get('success'):
                self.order_id = order_result.get('order_id')
                self.entry_timestamp = datetime.now().isoformat()
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Позиция {side} открыта: Order ID {self.order_id}")
                return True
            else:
                error = order_result.get('error', 'Unknown error') if order_result else 'No response'
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть позицию: {error}")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка открытия позиции: {e}")
            return False
    
    def _close_position_on_exchange(self, reason):
        """Закрывает позицию на бирже"""
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return False
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🔴 Закрываем позицию {self.position_side} (причина: {reason})")
            
            # Закрываем позицию на бирже
            close_result = self.exchange.close_position(
                symbol=self.symbol,
                side=self.position_side
            )
            
            if close_result and close_result.get('success'):
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Позиция закрыта успешно")
                self.update_status(BOT_STATUS['IDLE'])
                return True
            else:
                error = close_result.get('error', 'Unknown error') if close_result else 'No response'
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось закрыть позицию: {error}")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка закрытия позиции: {e}")
            return False
    
    def emergency_close_delisting(self):
        """Экстренное закрытие позиции при делистинге - рыночным ордером по любой цене"""
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована для экстренного закрытия")
                return False
            
            if self.status not in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Бот не в позиции, экстренное закрытие не требуется")
                return True
            
            logger.warning(f"[NEW_BOT_{self.symbol}] 🚨 ЭКСТРЕННОЕ ЗАКРЫТИЕ: ДЕЛИСТИНГ ОБНАРУЖЕН! Закрываем {self.position_side} рыночным ордером")
            
            # Определяем сторону для закрытия (противоположную позиции)
            close_side = 'Sell' if self.position_side == 'Long' else 'Buy'
            
            # Экстренное закрытие рыночным ордером
            emergency_result = self.exchange.close_position(
                symbol=self.symbol,
                side=self.position_side,
                order_type='Market',  # Принудительно рыночный ордер
                emergency=True  # Флаг экстренного закрытия
            )
            
            if emergency_result and emergency_result.get('success'):
                logger.warning(f"[NEW_BOT_{self.symbol}] ✅ ЭКСТРЕННОЕ ЗАКРЫТИЕ УСПЕШНО: Позиция закрыта рыночным ордером")
                self.update_status(BOT_STATUS['IDLE'])
                
                # Дополнительно обнуляем все данные позиции
                self.position_side = None
                self.entry_price = None
                self.unrealized_pnl = 0.0
                self.max_profit_achieved = 0.0
                self.trailing_stop_price = None
                self.break_even_activated = False
                
                return True
            else:
                error = emergency_result.get('error', 'Unknown error') if emergency_result else 'No response'
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ ЭКСТРЕННОЕ ЗАКРЫТИЕ НЕУДАЧНО: {error}")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА ЭКСТРЕННОГО ЗАКРЫТИЯ: {e}")
            return False
    
    def calculate_dynamic_take_profit(self, direction, current_price, current_rsi):
        """
        Рассчитывает динамический Take Profit на основе RSI настроек
        
        Args:
            direction (str): Направление позиции ('LONG' или 'SHORT')
            current_price (float): Текущая цена
            current_rsi (float): Текущий RSI
            
        Returns:
            float: Цена Take Profit или None если не нужен
        """
        try:
            # ОТЛАДКА: Логируем входные параметры
            logger.info(f"[NEW_BOT_{self.symbol}] 🔍 TP CALC DEBUG: direction={direction}, price={current_price}, rsi={current_rsi}")
            
            # Получаем настройки RSI из конфигурации
            rsi_exit_long = self.config.get('rsi_exit_long', 55)
            rsi_exit_short = self.config.get('rsi_exit_short', 45)
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🔍 TP CONFIG: rsi_exit_long={rsi_exit_long}, rsi_exit_short={rsi_exit_short}")
            
            if direction == 'LONG':
                # Для LONG: TP когда RSI достигнет rsi_exit_long
                if current_rsi >= rsi_exit_long:
                    logger.info(f"[NEW_BOT_{self.symbol}] ❌ TP для LONG: RSI {current_rsi} уже >= {rsi_exit_long}, TP не нужен")
                    return None  # Основное условие уже сработало
                
                # Рассчитываем примерную цену для достижения целевого RSI
                # RSI растет при росте цены, используем коэффициент
                rsi_ratio = rsi_exit_long / current_rsi
                # Консервативный коэффициент роста цены
                price_multiplier = 1 + (rsi_ratio - 1) * 0.6
                tp_price = current_price * price_multiplier
                
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ TP для LONG: RSI {current_rsi}→{rsi_exit_long}, цена {current_price:.6f}→{tp_price:.6f} (множитель: {price_multiplier:.3f})")
                return tp_price
                
            elif direction == 'SHORT':
                # Для SHORT: TP когда RSI достигнет rsi_exit_short
                if current_rsi <= rsi_exit_short:
                    return None  # Основное условие уже сработало
                
                # Рассчитываем примерную цену для достижения целевого RSI
                rsi_ratio = current_rsi / rsi_exit_short
                # Консервативный коэффициент падения цены
                price_multiplier = 1 - (rsi_ratio - 1) * 0.6
                tp_price = current_price * price_multiplier
                
                logger.info(f"[NEW_BOT_{self.symbol}] 📉 TP для SHORT: RSI {current_rsi}→{rsi_exit_short}, цена {current_price:.6f}→{tp_price:.6f}")
                return tp_price
            
            return None
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка расчета TP: {e}")
            return None
    
    def update_trailing_take_profit(self, current_price, current_rsi):
        """
        Обновляет трейлинг Take Profit пока не сработало основное RSI условие
        
        Args:
            current_price (float): Текущая цена
            current_rsi (float): Текущий RSI
            
        Returns:
            bool: True если TP обновлен, False если основное условие сработало
        """
        try:
            if not self.position_side:
                return False
            
            direction = self.position_side.upper()
            
            # Проверяем основное RSI условие
            rsi_exit_long = self.config.get('rsi_exit_long', 55)
            rsi_exit_short = self.config.get('rsi_exit_short', 45)
            
            if direction == 'LONG' and current_rsi >= rsi_exit_long:
                logger.info(f"[NEW_BOT_{self.symbol}] 🎯 Основное RSI условие сработало (RSI={current_rsi} >= {rsi_exit_long}) - закрываем позицию")
                return False  # Основное условие сработало
            
            if direction == 'SHORT' and current_rsi <= rsi_exit_short:
                logger.info(f"[NEW_BOT_{self.symbol}] 🎯 Основное RSI условие сработало (RSI={current_rsi} <= {rsi_exit_short}) - закрываем позицию")
                return False  # Основное условие сработало
            
            # Основное условие НЕ сработало - обновляем TP
            new_tp = self.calculate_dynamic_take_profit(direction, current_price, current_rsi)
            
            # Проверяем, нужно ли обновлять TP (для LONG - выше цены, для SHORT - ниже)
            should_update_tp = False
            if direction == 'LONG' and new_tp and new_tp > current_price:
                should_update_tp = True
            elif direction == 'SHORT' and new_tp and new_tp < current_price:
                should_update_tp = True
            
            if should_update_tp:
                # Обновляем TP на бирже
                success = self._update_take_profit_on_exchange(new_tp)
                if success:
                    logger.info(f"[NEW_BOT_{self.symbol}] 📈 TP обновлен: {current_price:.6f} → {new_tp:.6f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления трейлинг TP: {e}")
            return False
    
    def _update_take_profit_on_exchange(self, tp_price):
        """
        Обновляет Take Profit на бирже
        
        Args:
            tp_price (float): Новая цена Take Profit
            
        Returns:
            bool: True если успешно обновлен
        """
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return False
            
            # Обновляем TP через биржу, передавая направление позиции
            result = self.exchange.update_take_profit(self.symbol, tp_price, self.position_side)
            
            if result and result.get('success'):
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ TP обновлен на бирже: {tp_price:.6f}")
                return True
            else:
                error = result.get('message', 'Unknown error') if result else 'No response'
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось обновить TP: {error}")
                return False
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления TP на бирже: {e}")
            return False
            
    def to_dict(self):
        """Преобразует бота в словарь для сохранения"""
        return {
            'symbol': self.symbol,
            'status': self.status,
            'entry_price': self.entry_price,
            'position_side': self.position_side,
            'unrealized_pnl': self.unrealized_pnl,
            'created_at': self.created_at,
            'last_signal_time': self.last_signal_time,
            'max_profit_achieved': self.max_profit_achieved,
            'trailing_stop_price': self.trailing_stop_price,
            'break_even_activated': self.break_even_activated,
            'position_start_time': self.position_start_time.isoformat() if self.position_start_time else None,
            'order_id': self.order_id,
            'entry_timestamp': self.entry_timestamp,
            'opened_by_autobot': self.opened_by_autobot
        }

