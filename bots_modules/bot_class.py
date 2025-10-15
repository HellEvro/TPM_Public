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

# Глобальные переменные (импортируются из главного файла)
bots_data_lock = threading.Lock()
bots_data = {}
rsi_data_lock = threading.Lock()
coins_rsi_data = {}
BOT_STATUS = {
    'IDLE': 'idle',
    'RUNNING': 'running',
    'IN_POSITION_LONG': 'in_position_long',
    'IN_POSITION_SHORT': 'in_position_short',
    'PAUSED': 'paused'
}

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
            with rsi_data_lock:
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
    
    def _open_position_on_exchange(self, side, price):
        """Открывает позицию на бирже"""
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return False
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Открываем позицию {side} @ {price}")
            
            # Открываем позицию на бирже
            order_result = self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                qty=None,  # Будет рассчитано по volume_value
                qty_in_usdt=self.volume_value
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

