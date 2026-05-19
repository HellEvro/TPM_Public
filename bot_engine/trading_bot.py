"""
Основной класс торгового бота с логикой RSI на 6H таймфрейме
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from .bot_config import (
    BotStatus, TrendDirection, VolumeMode,
    DEFAULT_BOT_CONFIG, TIMEFRAME, get_current_timeframe
)
from .indicators import SignalGenerator
from .scaling_calculator import calculate_scaling_for_bot

# Символы, по которым уже вывели предупреждение о делистинге (один раз за сессию)
_delisting_warned_symbols = set()


class TradingBot:
    """Торговый бот для одной монеты"""
    
    def __init__(self, symbol: str, exchange, config: dict = None):
        self.symbol = symbol
        self.exchange = exchange
        self.config = {**DEFAULT_BOT_CONFIG, **(config or {})}
        
        # Состояние бота
        self.status = self.config.get('status', BotStatus.IDLE)
        self.auto_managed = self.config.get('auto_managed', False)
        
        # Параметры сделки
        self.volume_mode = self.config.get('volume_mode', VolumeMode.FIXED_USDT)
        self.volume_value = self.config.get('volume_value', 10.0)
        self.max_loss_percent = self.config.get('max_loss_percent', 2.0)
        
        # Текущая позиция
        self.position = self.config.get('position')
        self.entry_price = self.config.get('entry_price')
        self.entry_time = self.config.get('entry_time')
        self.last_signal_time = self.config.get('last_signal_time')
        self.last_price = self.config.get('last_price')
        self.last_rsi = self.config.get('last_rsi')
        self.last_trend = self.config.get('last_trend')
        
        # Расширенные параметры позиции
        self.position_side = self.config.get('position_side')
        position_start = self.config.get('position_start_time')
        if position_start and hasattr(position_start, 'isoformat'):
            self.position_start_time = position_start
        elif isinstance(position_start, str):
            try:
                self.position_start_time = datetime.fromisoformat(position_start)
            except ValueError:
                self.position_start_time = position_start
        else:
            self.position_start_time = position_start
        
        self.position_size = self.config.get('position_size')
        self.position_size_coins = self.config.get('position_size_coins')
        self.unrealized_pnl = self.config.get('unrealized_pnl', 0.0)
        self.unrealized_pnl_usdt = self.config.get('unrealized_pnl_usdt', 0.0)
        self.realized_pnl = self.config.get('realized_pnl', 0.0)
        self.leverage = self.config.get('leverage', 1.0)
        self.margin_usdt = self.config.get('margin_usdt')
        self.max_profit_achieved = self.config.get('max_profit_achieved', 0.0)
        self.trailing_stop_price = self.config.get('trailing_stop_price')
        self.trailing_activation_profit = self.config.get('trailing_activation_profit', 0.0)
        self.trailing_activation_threshold = self.config.get('trailing_activation_threshold', 0.0)
        self.trailing_locked_profit = self.config.get('trailing_locked_profit', 0.0)
        self.trailing_active = bool(self.config.get('trailing_active', False))
        self.trailing_max_profit_usdt = float(self.config.get('trailing_max_profit_usdt', 0.0) or 0.0)
        self.trailing_step_usdt = float(self.config.get('trailing_step_usdt', 0.0) or 0.0)
        self.trailing_step_price = float(self.config.get('trailing_step_price', 0.0) or 0.0)
        self.trailing_steps = int(self.config.get('trailing_steps', 0) or 0)
        self.break_even_activated = self.config.get('break_even_activated', False)
        self.order_id = self.config.get('order_id')
        self.current_price = self.config.get('current_price')
        created = self.config.get('created_at')
        if created and hasattr(created, 'isoformat'):
            self.created_at = created
        elif isinstance(created, str):
            try:
                self.created_at = datetime.fromisoformat(created)
            except ValueError:
                self.created_at = created
        else:
            self.created_at = datetime.now()
        self.rsi_data = self.config.get('rsi_data', {})
        
        # Масштабирование (лесенка)
        self.scaling_enabled = self.config.get('scaling_enabled', False)
        self.scaling_levels = self.config.get('scaling_levels', [])
        self.scaling_current_level = self.config.get('scaling_current_level', 0)
        self.scaling_group_id = self.config.get('scaling_group_id', None)
        
        # Лимитные ордера для набора позиций
        self.limit_orders = self.config.get('limit_orders', [])  # Список активных лимитных ордеров
        self.limit_orders_entry_price = self.config.get('limit_orders_entry_price')  # Цена входа для расчета лимитных ордеров
        self.last_limit_orders_count = len(self.limit_orders)  # Количество активных лимитных ордеров на последней проверке
        
        # Логирование
        self.logger = logging.getLogger(f'TradingBot.{symbol}')
        
        # Анализ
        try:
            self.signal_generator = SignalGenerator()
            self.logger.info(f" {symbol}: SignalGenerator создан успешно")
        except Exception as e:
            self.logger.error(f" {symbol}: Ошибка создания SignalGenerator: {e}")
            raise
        self.last_analysis = None
        self.last_bar_timestamp = None
        
        self.logger.info(f"Bot initialized for {symbol} with config: {self.config}")
    
    def to_dict(self) -> Dict:
        """Преобразует состояние бота в словарь для сохранения"""
        if hasattr(self.status, 'value'):
            raw_status = self.status.value
        else:
            raw_status = str(self.status) if self.status is not None else ''

        normalized_status = raw_status.lower()

        return {
            'symbol': self.symbol,
            'status': normalized_status,
            'auto_managed': self.auto_managed,
            'volume_mode': self.volume_mode.value if hasattr(self.volume_mode, 'value') else str(self.volume_mode),
            'volume_value': self.volume_value,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time and hasattr(self.entry_time, 'isoformat') else self.entry_time,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time and hasattr(self.last_signal_time, 'isoformat') else self.last_signal_time,
            'last_bar_timestamp': self.last_bar_timestamp,
            'position_side': self.position_side or (self.position.get('side') if self.position else None),
            'position_start_time': self.position_start_time.isoformat() if self.position_start_time and hasattr(self.position_start_time, 'isoformat') else self.position_start_time,
            'position_size': self.position_size,
            'position_size_coins': self.position_size_coins,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_usdt': self.unrealized_pnl_usdt,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage,
            'margin_usdt': self.margin_usdt,
            'max_profit_achieved': self.max_profit_achieved,
            'trailing_stop_price': self.trailing_stop_price,
            'trailing_activation_profit': self.trailing_activation_profit,
            'trailing_activation_threshold': self.trailing_activation_threshold,
            'trailing_locked_profit': self.trailing_locked_profit,
            'trailing_active': self.trailing_active,
            'trailing_max_profit_usdt': self.trailing_max_profit_usdt,
            'trailing_step_usdt': self.trailing_step_usdt,
            'trailing_step_price': self.trailing_step_price,
            'trailing_steps': self.trailing_steps,
            'break_even_activated': self.break_even_activated,
            'order_id': self.order_id,
            'current_price': self.current_price,
            'last_price': self.last_price,
            'last_rsi': self.last_rsi,
            'last_trend': self.last_trend,
            'rsi_data': self.rsi_data,
            'created_at': self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else self.created_at,
            'scaling_enabled': self.scaling_enabled,
            'scaling_levels': self.scaling_levels,
            'scaling_current_level': self.scaling_current_level,
            'scaling_group_id': self.scaling_group_id,
            'limit_orders': self.limit_orders,
            'limit_orders_entry_price': self.limit_orders_entry_price
        }
    
    def update(self, force_analysis: bool = False, external_signal: str = None, external_trend: str = None) -> Dict:
        """
        Обновляет состояние бота и выполняет торговую логику
        
        Args:
            force_analysis: Принудительный анализ (игнорирует временные ограничения)
            external_signal: Внешний сигнал (ENTER_LONG, ENTER_SHORT, WAIT)
            external_trend: Внешний тренд (UP, DOWN, NEUTRAL)
            
        Returns:
            Словарь с результатами обновления
        """
        try:
            self.logger.info(f" {self.symbol}: Начинаем update method...")
            self.logger.info(f" {self.symbol}: External signal: {external_signal}, trend: {external_trend}")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: если статус указывает на позицию, но position = null, сбрасываем статус
            if self.status in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT] and self.position is None:
                self.logger.warning(f" {self.symbol}: ⚠️ Несоответствие статуса! Статус: {self.status}, но позиция: {self.position}. Сбрасываем статус.")
                self.status = BotStatus.IDLE
            
            # КРИТИЧЕСКАЯ СИНХРОНИЗАЦИЯ: проверяем реальные позиции на бирже
            try:
                exchange_positions = self.exchange.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                # Ищем позицию по нашему символу
                real_position = None
                for pos in positions_list:
                    if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                        real_position = pos
                        break
                
                # Если на бирже есть позиция, но в боте её нет - синхронизируем
                if real_position and not self.position:
                    self.logger.warning(f" {self.symbol}: 🔄 Синхронизация: на бирже есть позиция {real_position}, но в боте нет!")
                    self.position = {
                        'side': 'LONG' if float(real_position.get('size', 0)) > 0 else 'SHORT',
                        'quantity': abs(float(real_position.get('size', 0))),
                        'entry_price': real_position.get('entry_price'),
                        'order_id': real_position.get('order_id', 'unknown')
                    }
                    self.entry_price = real_position.get('entry_price')
                    self.status = BotStatus.IN_POSITION_LONG if self.position['side'] == 'LONG' else BotStatus.IN_POSITION_SHORT
                    self.logger.info(f" {self.symbol}: ✅ Синхронизировано: {self.position}")
                
                # Если в боте есть позиция, но на бирже нет - очищаем
                elif self.position and not real_position:
                    self.logger.warning(f" {self.symbol}: 🔄 Синхронизация: в боте есть позиция {self.position}, но на бирже нет!")
                    self.position = None
                    self.entry_price = None
                    self.entry_time = None
                    self.status = BotStatus.IDLE
                    self.logger.info(f" {self.symbol}: ✅ Позиция очищена")
                    
            except Exception as sync_error:
                self.logger.warning(f" {self.symbol}: Ошибка синхронизации с биржей: {sync_error}")
            
            # Если есть внешний сигнал, используем его вместо генерации
            if external_signal:
                self.logger.info(f" {self.symbol}: Используем внешний сигнал: {external_signal}")
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: если уже есть позиция, НЕ ОТКРЫВАЕМ новую!
                if self.position:
                    self.logger.warning(f" {self.symbol}: ⚠️ Уже есть позиция {self.position['side']} - ИГНОРИРУЕМ внешний сигнал {external_signal}")
                    analysis = {
                        'signal': 'WAIT',  # Игнорируем внешний сигнал
                        'trend': external_trend or 'NEUTRAL',
                        'rsi': 0,
                        'price': self._get_current_price() or 0
                    }
                else:
                    analysis = {
                        'signal': external_signal,
                        'trend': external_trend or 'NEUTRAL',
                        'rsi': 0,  # Заглушка, так как RSI не используется в торговой логике
                        'price': self._get_current_price() or 0
                    }
                self.logger.info(f" {self.symbol}: Внешний анализ: {analysis}")
            else:
                # Получаем данные свечей
                self.logger.info(f" {self.symbol}: Получаем данные свечей...")
                candles_data = self._get_candles_data()
                if not candles_data:
                    self.logger.warning(f" {self.symbol}: Не удалось получить данные свечей")
                    return {'success': False, 'error': 'failed_to_get_candles'}
                self.logger.info(f" {self.symbol}: Получено {len(candles_data)} свечей")
                
                # Проверяем, нужно ли обновлять анализ
                current_bar_timestamp = candles_data[-1].get('timestamp')
                self.logger.info(f" {self.symbol}: Проверяем необходимость обновления: force_analysis={force_analysis}, current_bar={current_bar_timestamp}, last_bar={self.last_bar_timestamp}")
                if not force_analysis and current_bar_timestamp == self.last_bar_timestamp:
                    # Бар не изменился, возвращаем последний анализ
                    self.logger.info(f" {self.symbol}: Бар не изменился, возвращаем последний анализ")
                    return self._get_current_state()
                else:
                    self.logger.info(f" {self.symbol}: Бар изменился или принудительный анализ, продолжаем...")
                
                # Выполняем анализ
                self.logger.info(f" {self.symbol}: Генерируем сигналы...")
                analysis = self.signal_generator.generate_signals(candles_data)
                self.logger.info(f" {self.symbol}: Анализ завершен: {analysis}")
                self.last_bar_timestamp = current_bar_timestamp
            
            self.last_analysis = analysis
            
            # Выполняем торговую логику
            self.logger.info(f" {self.symbol}: Выполняем торговую логику...")
            if self.status != BotStatus.PAUSED:
                action_result = self._execute_trading_logic(analysis)
                if action_result:
                    self.logger.info(f"Action executed: {action_result}")
                else:
                    self.logger.info(f" {self.symbol}: Нет действий для выполнения")
            else:
                self.logger.info(f" {self.symbol}: Бот приостановлен")
            
            self.logger.info(f" {self.symbol}: Возвращаем текущее состояние...")
            return self._get_current_state()
            
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_candles_data(self) -> List[Dict]:
        """Получает данные свечей с биржи"""
        try:
            self.logger.info(f" {self.symbol}: Получаем данные свечей...")
            self.logger.info(f" {self.symbol}: Exchange type: {type(self.exchange)}")
            # Получаем данные за последние 200 баров 6H для анализа
            chart_response = self.exchange.get_chart_data(
                symbol=self.symbol,
                timeframe=TIMEFRAME,
                period='1w'  # Используем period вместо limit
            )
            self.logger.info(f" {self.symbol}: Chart response type: {type(chart_response)}")
            
            # Проверяем успешность ответа и извлекаем свечи
            if isinstance(chart_response, dict) and chart_response.get('success'):
                candles = chart_response.get('data', {}).get('candles', [])
                
                # Конвертируем в нужный формат с timestamp
                # (порядок уже исправлен в exchange классе)
                formatted_candles = []
                for candle in candles:
                    formatted_candle = {
                        'timestamp': candle.get('time'),
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0))
                    }
                    formatted_candles.append(formatted_candle)
                
                # Логируем для проверки
                if formatted_candles:
                    pass
                    pass
                
                return formatted_candles
            else:
                self.logger.error(f"Failed to get chart data: {chart_response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get candles data: {str(e)}")
            return []
    
    def _execute_trading_logic(self, analysis: Dict) -> Optional[Dict]:
        """
        Выполняет торговую логику на основе анализа
        
        Args:
            analysis: Результат анализа сигналов
            
        Returns:
            Результат выполненного действия или None
        """
        signal = analysis.get('signal')
        trend = analysis.get('trend')
        
        # Проверяем изменение тренда для принудительного выхода
        if self._should_force_exit(trend):
            return self._force_exit_position()
        
        # Проверяем лимитные ордера и отменяем их при выходе за зону набора позиций
        if self.limit_orders:
            self._check_and_cancel_limit_orders_if_needed(analysis)
            # Проверяем сработавшие лимитные ордера и обновляем стоп-лосс
            self._check_and_update_limit_orders_fills()
        
        # Выполняем действия в зависимости от текущего статуса
        if self.status in [BotStatus.IDLE, 'running']:
            return self._handle_idle_state(signal, trend)
        
        
        elif self.status in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT]:
            # Проверяем, есть ли реальная позиция
            if not self.position:
                # Если статус IN_POSITION, но позиции нет - это ошибка синхронизации
                # Возвращаемся в IDLE и пытаемся открыть позицию заново
                self.logger.warning(f" {self.symbol}: Статус {self.status} но позиции нет! Возвращаемся в IDLE")
                self.status = BotStatus.IDLE
                return self._handle_idle_state(signal, trend)
            else:
                return self._handle_position_state(signal, trend)
        
        return None
    
    def _should_force_exit(self, current_trend: str) -> bool:
        """Проверяет, нужно ли принудительно закрыть позицию при смене тренда"""
        if not self.position:
            return False
        
        position_type = self.position.get('side')
        
        # Принудительный выход при смене тренда на противоположный
        if position_type == 'LONG' and current_trend == 'DOWN':
            return True
        elif position_type == 'SHORT' and current_trend == 'UP':
            return True
        
        return False
    
    def _handle_idle_state(self, signal: str, trend: str) -> Optional[Dict]:
        """Обрабатывает состояние IDLE - СРАЗУ открывает сделки!"""
        self.logger.info(f" {self.symbol}: _handle_idle_state: signal={signal}, trend={trend}")
        
        # Проверяем, есть ли уже позиция в боте
        if self.position:
            self.logger.warning(f" {self.symbol}: ⚠️ Уже есть позиция {self.position['side']} - пропускаем вход")
            return {'action': 'position_exists', 'side': self.position['side'], 'price': self.position.get('entry_price')}
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем реальные позиции на бирже!
        try:
            exchange_positions = self.exchange.get_positions()
            if isinstance(exchange_positions, tuple):
                positions_list = exchange_positions[0] if exchange_positions else []
            else:
                positions_list = exchange_positions if exchange_positions else []
            
            # Проверяем, есть ли уже позиция по этому символу на бирже
            for pos in positions_list:
                if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                    existing_side = pos.get('side', 'UNKNOWN')
                    position_size = pos.get('size', 0)
                    
                    self.logger.warning(f" {self.symbol}: 🚫 НА БИРЖЕ УЖЕ ЕСТЬ ПОЗИЦИЯ {existing_side} размер {position_size}!")
                    self.logger.warning(f" {self.symbol}: ❌ БЛОКИРУЕМ ОТКРЫТИЕ НОВОЙ ПОЗИЦИИ - ЗАЩИТА ОТ ДУБЛИРОВАНИЯ!")
                    
                    return {
                        'action': 'blocked_exchange_position', 
                        'side': existing_side, 
                        'size': position_size,
                        'message': f'На бирже уже есть позиция {existing_side} размер {position_size}'
                    }
            
            self.logger.info(f" {self.symbol}: ✅ На бирже нет позиций - можно открывать сделку")
            
        except Exception as check_error:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки позиций на бирже: {check_error}")
            self.logger.error(f" {self.symbol}: 🚫 БЛОКИРУЕМ ОТКРЫТИЕ ПОЗИЦИИ ИЗ-ЗА ОШИБКИ ПРОВЕРКИ!")
            return {
                'action': 'blocked_check_error', 
                'error': str(check_error),
                'message': 'Ошибка проверки позиций на бирже'
            }
        
        # ПРОВЕРКА RSI ВРЕМЕННОГО ФИЛЬТРА (по выбранному таймфрейму)
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from bots import check_rsi_time_filter

            tf_entry = get_current_timeframe()
            candles = self.exchange.get_candles(self.symbol, tf_entry, 100)
            if candles and len(candles) > 0:
                current_rsi = getattr(self, 'current_rsi', None)
                if current_rsi is None:
                    try:
                        rsi_data = self.exchange.get_rsi_data(self.symbol, tf_entry, 14)
                        current_rsi = rsi_data.get('rsi', 50) if rsi_data else 50
                    except Exception:
                        current_rsi = 50
                
                # Проверяем временной фильтр
                time_filter_result = check_rsi_time_filter(candles, current_rsi, signal)
                
                if not time_filter_result['allowed']:
                    pass
                    return {
                        'action': 'blocked_time_filter',
                        'reason': time_filter_result['reason'],
                        'last_extreme_candles_ago': time_filter_result.get('last_extreme_candles_ago')
                    }
            else:
                self.logger.warning(f" {self.symbol}: ⚠️ Не удалось получить свечи для проверки временного фильтра")
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки временного фильтра: {e}")
            # В случае ошибки разрешаем сделку (безопасность)
        
        # КРИТИЧЕСКИ ВАЖНО: Если автобот выключен - НЕ ОТКРЫВАЕМ новые позиции!
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from bots import bots_data, bots_data_lock
            
            with bots_data_lock:
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            if not auto_bot_enabled:
                self.logger.info(f" {self.symbol}: ⏹️ Auto Bot выключен - НЕ открываем новую позицию из IDLE состояния")
                return {'action': 'blocked_autobot_disabled', 'reason': 'autobot_off'}
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки автобота: {e}")
            # В случае ошибки блокируем для безопасности
            return {'action': 'blocked_check_error', 'reason': 'autobot_check_failed'}
        
        # ПРЯМАЯ ЛОГИКА: Сразу открываем сделки без промежуточных состояний
        if signal == 'ENTER_LONG':
            self.logger.info(f" {self.symbol}: 🚀 СРАЗУ открываем LONG позицию!")
            return self._enter_position('LONG')
        
        elif signal == 'ENTER_SHORT':
            self.logger.info(f" {self.symbol}: 🚀 СРАЗУ открываем SHORT позицию!")
            return self._enter_position('SHORT')
        
        self.logger.info(f" {self.symbol}: Нет сигналов для входа: signal={signal}, trend={trend}")
        return None
    
    
    def _handle_position_state(self, signal: str, trend: str) -> Optional[Dict]:
        """Обрабатывает состояния IN_POSITION_LONG/SHORT"""
        # КРИТИЧЕСКИ ВАЖНО: Если автобот выключен - НЕ ОТКРЫВАЕМ новые позиции!
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from bots import bots_data, bots_data_lock
            
            with bots_data_lock:
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            if not auto_bot_enabled:
                # Если автобот выключен - только управляем существующими позициями (стопы, трейлинг)
                # НЕ открываем новые позиции
                if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                    self.logger.info(f" {self.symbol}: ⏹️ Auto Bot выключен - НЕ открываем новые позиции из POSITION состояния")
                    return {'action': 'blocked_autobot_disabled', 'reason': 'autobot_off', 'status': self.status}
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки автобота: {e}")
        
        position_type = self.position.get('side') if self.position else None
        
        if (self.status == BotStatus.IN_POSITION_LONG and 
            (signal == 'EXIT_LONG' or position_type == 'LONG')):
            return self._exit_position()
        
        elif (self.status == BotStatus.IN_POSITION_SHORT and 
              (signal == 'EXIT_SHORT' or position_type == 'SHORT')):
            return self._exit_position()
        
        return None
    
    def _enter_position(self, side: str, force_market_entry: bool = False) -> Dict:
        """Входит в позицию. force_market_entry=True — автоматический вход, всегда по рынку (игнор лимитных ордеров)."""
        self.logger.info(f" {self.symbol}: 🎯 _enter_position вызван для {side}" + (" (вход по рынку)" if force_market_entry else ""))
        try:
            # ✅ ПРОВЕРКА ДЕЛИСТИНГА: Проверяем ДО всех остальных проверок!
            try:
                from bots_modules.sync_and_cache import load_delisted_coins
                delisted_data = load_delisted_coins()
                delisted_coins = delisted_data.get('delisted_coins', {})
                
                if self.symbol in delisted_coins:
                    delisting_info = delisted_coins[self.symbol]
                    if self.symbol not in _delisting_warned_symbols:
                        _delisting_warned_symbols.add(self.symbol)
                        self.logger.warning(f" {self.symbol}: ⚠️ Делистинг — не открываем {side} ({delisting_info.get('reason', 'Delisting detected')}). Монета помечена в списке.")
                    return {'success': False, 'error': 'coin_delisted', 'message': f'Монета в делистинге: {delisting_info.get("reason", "Delisting detected")}'}
            except Exception as delisting_check_error:
                pass
                # Продолжаем работу, если не удалось проверить делистинг
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: не открываем новую позицию, если уже есть открытая
            if self.position is not None:
                self.logger.warning(f" {self.symbol}: ⚠️ Позиция уже открыта! Текущая позиция: {self.position}")
                return {'success': False, 'error': 'position_already_exists', 'message': 'Позиция уже открыта'}
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: не открываем позицию, если статус бота указывает на позицию
            if self.status in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT]:
                self.logger.warning(f" {self.symbol}: ⚠️ Бот уже в позиции! Статус: {self.status}")
                return {'success': False, 'error': 'bot_already_in_position', 'message': f'Бот уже в позиции (статус: {self.status})'}
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: проверяем реальные позиции на бирже ПЕРЕД открытием!
            try:
                exchange_positions = self.exchange.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                # Проверяем, есть ли уже позиция по этому символу на бирже
                for pos in positions_list:
                    if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                        existing_side = pos.get('side', 'UNKNOWN')
                        position_size = pos.get('size', 0)
                        
                        self.logger.warning(f" {self.symbol}: ⚠️ На бирже уже есть позиция {existing_side} размер {position_size} — защита от дублирования")
                        
                        return {
                            'success': False, 
                            'error': 'exchange_position_exists', 
                            'message': f'На бирже уже есть позиция {existing_side} размер {position_size}',
                            'existing_side': existing_side,
                            'existing_size': position_size
                        }
                
                self.logger.info(f" {self.symbol}: ✅ Финальная проверка: на бирже нет позиций - открываем {side}")
                
            except Exception as exchange_check_error:
                self.logger.error(f" {self.symbol}: ❌ Ошибка финальной проверки позиций на бирже: {exchange_check_error}")
                self.logger.error(f" {self.symbol}: 🚫 БЛОКИРУЕМ ОТКРЫТИЕ ПОЗИЦИИ ИЗ-ЗА ОШИБКИ ПРОВЕРКИ!")
                return {
                    'success': False, 
                    'error': 'exchange_check_failed', 
                    'message': f'Ошибка проверки позиций на бирже: {exchange_check_error}'
                }
            
            # Дополнительная проверка через биржу
            try:
                exchange_positions = self.exchange.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                for pos in positions_list:
                    if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                        self.logger.warning(f" {self.symbol}: ⚠️ На бирже уже есть позиция: {pos}")
                        return {'success': False, 'error': 'exchange_position_exists', 'message': 'На бирже уже есть позиция'}
            except Exception as e:
                self.logger.warning(f" {self.symbol}: Не удалось проверить позиции на бирже: {e}")
            
            # Единая проверка входа (autobot и ручной) — без fallback RSI=50
            try:
                from bot_engine.ai.filter_utils import check_entry_allowed, log_entry_check
                from bot_engine.config_loader import get_current_timeframe
                from bots_modules.imports_and_globals import get_config_snapshot
                entry_source = 'autobot' if force_market_entry else 'manual'
                config_snapshot = get_config_snapshot(self.symbol)
                filter_config = config_snapshot.get('merged', {})
                candles = self._get_candles_data()
                if not candles or len(candles) < 10:
                    self.logger.error(
                        f" {self.symbol}: ❌ Недостаточно свечей для проверки фильтров "
                        f"({len(candles) if candles else 0})"
                    )
                    return {
                        'success': False,
                        'error': 'insufficient_candles',
                        'message': 'Недостаточно свечей для проверки фильтров',
                    }
                current_rsi = self.last_rsi
                current_trend = self.last_trend
                if current_rsi is None or current_trend is None:
                    try:
                        from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
                        with rsi_data_lock:
                            coin_data = coins_rsi_data.get('coins', {}).get(self.symbol)
                            if coin_data:
                                from bot_engine.config_loader import (
                                    get_rsi_from_coin_data,
                                    get_trend_from_coin_data,
                                )
                                if current_rsi is None:
                                    current_rsi = get_rsi_from_coin_data(coin_data)
                                if current_trend is None:
                                    current_trend = get_trend_from_coin_data(coin_data)
                    except Exception:
                        pass
                if current_rsi is None:
                    try:
                        from bots_modules.calculations import calculate_rsi
                        closes = [candle.get('close', 0) for candle in candles[-50:]]
                        if closes:
                            current_rsi = calculate_rsi(closes, 14)
                    except Exception:
                        pass
                if current_trend is None:
                    current_trend = 'NEUTRAL'
                if current_rsi is None:
                    log_entry_check(
                        self.logger,
                        self.symbol,
                        side,
                        False,
                        'RSI недоступен',
                        rsi=None,
                        timeframe=get_current_timeframe(),
                        signal='ENTER_LONG' if side == 'LONG' else 'ENTER_SHORT',
                        source=entry_source,
                        force_market_entry=force_market_entry,
                    )
                    return {
                        'success': False,
                        'error': 'rsi_unavailable',
                        'message': 'RSI недоступен — вход заблокирован',
                    }
                signal = 'ENTER_LONG' if side == 'LONG' else 'ENTER_SHORT'
                tf = get_current_timeframe()
                filters_allowed, filters_reason = check_entry_allowed(
                    self.symbol,
                    candles,
                    current_rsi,
                    signal,
                    filter_config,
                    trend=current_trend,
                    source=entry_source,
                    force_market_entry=force_market_entry,
                )
                log_entry_check(
                    self.logger,
                    self.symbol,
                    side,
                    filters_allowed,
                    filters_reason,
                    rsi=current_rsi,
                    timeframe=tf,
                    signal=signal,
                    source=entry_source,
                    force_market_entry=force_market_entry,
                )
                if not filters_allowed:
                    self.logger.warning(
                        f" {self.symbol}: 🚫 БЛОКИРОВКА: вход в {side}: {filters_reason}"
                    )
                    return {
                        'success': False,
                        'error': 'filters_blocked',
                        'message': f'Вход заблокирован фильтрами: {filters_reason}',
                    }
            except Exception as filter_error:
                self.logger.error(f" {self.symbol}: ❌ Ошибка проверки фильтров: {filter_error}")
                return {'success': False, 'error': 'filter_check_failed', 'message': str(filter_error)}
            
            self.logger.info(f" {self.symbol}: Начинаем открытие {side} позиции...")
            
            # Адаптируем размер позиции с помощью AI (если доступно)
            try:
                from bot_engine.config_loader import AIConfig
                if AIConfig.AI_ENABLED and AIConfig.AI_RISK_MANAGEMENT_ENABLED:
                    from bot_engine.ai import get_ai_manager
                    ai_manager = get_ai_manager()
                    
                    if ai_manager and ai_manager.risk_manager and self.volume_mode == VolumeMode.FIXED_USDT:
                        tf_use = self.config.get('entry_timeframe') or get_current_timeframe()
                        chart_response = self.exchange.get_chart_data(self.symbol, tf_use, '14d')
                        candles = chart_response.get('data', {}).get('candles', []) if chart_response and chart_response.get('success') else None
                        balance = self._get_available_balance() or 1000  # Fallback
                        
                        if candles and len(candles) >= 20:
                            dynamic_size = ai_manager.risk_manager.calculate_position_size(
                                self.symbol, candles, balance, signal_confidence=0.7
                            )
                            
                            # Обновляем volume_value для адаптивного размера
                            original_size = self.volume_value
                            self.volume_value = dynamic_size['size_usdt']
                            
                            self.logger.info(
                                f" {self.symbol}: 🤖 AI адаптировал размер: "
                                f"{original_size} USDT → {self.volume_value} USDT "
                                f"({dynamic_size['reason']})"
                            )
            except Exception as ai_error:
                pass
            
            # Получаем конфигурацию для набора позиций (только глобальные настройки)
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from bots import bots_data, bots_data_lock
                
                with bots_data_lock:
                    auto_config = bots_data.get('auto_bot_config', {})
                    limit_orders_enabled = auto_config.get('limit_orders_entry_enabled', False)
                    percent_steps = auto_config.get('limit_orders_percent_steps', [1, 2, 3, 4, 5])
                    margin_amounts = auto_config.get('limit_orders_margin_amounts', [0.2, 0.3, 0.5, 1, 2])
                
                # ✅ Логируем конфигурацию для диагностики
                self.logger.info(f" {self.symbol}: 🔍 Конфигурация лимитных ордеров (глобальные): enabled={limit_orders_enabled}, steps={percent_steps}, amounts={margin_amounts}")
            except Exception as e:
                self.logger.warning(f" {self.symbol}: Не удалось получить конфигурацию лимитных ордеров: {e}")
                limit_orders_enabled = False
                percent_steps = []
                margin_amounts = []
            
            # ✅ АВТОВХОД: при force_market_entry всегда по рынку, лимитные ордера не используем
            if force_market_entry:
                self.logger.info(f" {self.symbol}: 🚀 Автовход — вход строго по рынку (лимитные ордера не используются)")
            # ✅ КРИТИЧНО: Если режим лимитных ордеров ВЫКЛЮЧЕН - пропускаем ВСЮ логику лимитных ордеров!
            # Если включен набор позиций лимитными ордерами (и не принудительный рыночный вход)
            elif limit_orders_enabled and percent_steps and margin_amounts:
                # ✅ КРИТИЧНО: Проверяем, не размещены ли уже лимитные ордера
                # Проверяем как в памяти бота, так и на бирже
                has_limit_orders_in_memory = self.limit_orders and len(self.limit_orders) > 0
                limit_orders_on_exchange = []
                
                # Проверяем открытые ордера на бирже
                if hasattr(self.exchange, 'get_open_orders'):
                    try:
                        open_orders = self.exchange.get_open_orders(self.symbol)
                        # Фильтруем только лимитные ордера нужного направления
                        limit_side = 'Buy' if side == 'LONG' else 'Sell'
                        limit_orders_on_exchange = [
                            o for o in open_orders 
                            if o.get('order_type', '').lower() == 'limit' 
                            and o.get('side', '') == limit_side
                        ]
                    except Exception as e:
                        pass
                
                # Если в памяти нет ордеров, но на бирже есть - проверяем, что это ордера бота
                if not has_limit_orders_in_memory and limit_orders_on_exchange:
                    # Получаем текущую цену для проверки соответствия ордеров конфигурации
                    current_price = self._get_current_price()
                    if not current_price:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось получить текущую цену для проверки ордеров на бирже")
                        # Если не можем проверить - не восстанавливаем, размещаем новые
                        self.logger.info(f" {self.symbol}: ✅ Размещаем новые ордера (не удалось проверить существующие)")
                        return self._enter_position_with_limit_orders(side, percent_steps, margin_amounts)
                    
                    # ✅ ПРОВЕРКА: Соответствуют ли ордера на бирже конфигурации бота?
                    # Проверяем, что цены ордеров находятся в разумном диапазоне от текущей цены
                    # (в пределах максимального percent_step из конфигурации + небольшой запас)
                    max_percent_step = max(percent_steps) if percent_steps else 10
                    max_price_deviation = max_percent_step / 100.0 + 0.05  # +5% запас на случай изменения цены
                    
                    valid_orders = []
                    for order in limit_orders_on_exchange:
                        order_price = float(order.get('price', 0))
                        if not order_price:
                            continue
                        
                        # Проверяем, что цена ордера находится в разумном диапазоне
                        if side == 'LONG':
                            # Для лонга: лимитная цена должна быть ниже текущей (покупка по более низкой цене)
                            # Проверяем, что цена не слишком далеко (не более max_percent_step% ниже)
                            if order_price < current_price and (current_price - order_price) / current_price <= max_price_deviation:
                                valid_orders.append(order)
                            else:
                                self.logger.warning(f" {self.symbol}: ⚠️ Ордер {order.get('order_id', 'unknown')} @ {order_price:.6f} не соответствует конфигурации LONG (текущая: {current_price:.6f}, отклонение: {abs(current_price - order_price) / current_price * 100:.2f}%)")
                        else:  # SHORT
                            # Для шорта: лимитная цена должна быть выше текущей (продажа по более высокой цене)
                            # Проверяем, что цена не слишком далеко (не более max_percent_step% выше)
                            if order_price > current_price and (order_price - current_price) / current_price <= max_price_deviation:
                                valid_orders.append(order)
                            else:
                                self.logger.warning(f" {self.symbol}: ⚠️ Ордер {order.get('order_id', 'unknown')} @ {order_price:.6f} не соответствует конфигурации SHORT (текущая: {current_price:.6f}, отклонение: {abs(order_price - current_price) / current_price * 100:.2f}%)")
                    
                    if not valid_orders:
                        # На бирже есть ордера, но они не соответствуют конфигурации бота
                        # Это не ордера бота - размещаем новые
                        self.logger.warning(f" {self.symbol}: ⚠️ На бирже есть {len(limit_orders_on_exchange)} лимитных ордеров, но они НЕ соответствуют конфигурации бота (max_step={max_percent_step}%)")
                        self.logger.info(f" {self.symbol}: ✅ Размещаем новые ордера (существующие на бирже - не от этого бота)")
                        return self._enter_position_with_limit_orders(side, percent_steps, margin_amounts)
                    
                    # Ордера соответствуют конфигурации - восстанавливаем их в память
                    self.logger.info(f" {self.symbol}: 🔄 Обнаружены {len(valid_orders)} лимитных ордеров на бирже, соответствующих конфигурации (из {len(limit_orders_on_exchange)} всего) - восстановление после перезапуска")
                    # Восстанавливаем список ордеров из биржи
                    self.limit_orders = []
                    for order in valid_orders:
                        order_price = float(order.get('price', 0))
                        # Вычисляем приблизительный percent_step на основе цены
                        if side == 'LONG':
                            percent_step = (current_price - order_price) / current_price * 100
                        else:  # SHORT
                            percent_step = (order_price - current_price) / current_price * 100
                        
                        order_info = {
                            'order_id': order.get('order_id') or order.get('orderId') or order.get('id', ''),
                            'type': 'limit',
                            'price': order_price,
                            'quantity': float(order.get('quantity', 0)),
                            'percent_step': round(percent_step, 2)
                        }
                        self.limit_orders.append(order_info)
                    self.last_limit_orders_count = len(self.limit_orders)
                    self.logger.info(f" {self.symbol}: ✅ Восстановлено {len(self.limit_orders)} лимитных ордеров в памяти")
                    # Сохраняем цену входа (используем текущую цену)
                    self.limit_orders_entry_price = current_price
                    return {'success': True, 'message': 'limit_orders_restored', 'orders_count': len(self.limit_orders)}
                
                # Если в памяти есть ордера - проверяем, существуют ли они на бирже
                if has_limit_orders_in_memory:
                    # ✅ КРИТИЧНО: Проверяем, не были ли ордера удалены с биржи между перезагрузками
                    if limit_orders_on_exchange:
                        # Ордера есть и в памяти, и на бирже - все в порядке
                        self.logger.warning(f" {self.symbol}: ⚠️ Лимитные ордера уже размещены (в памяти: {len(self.limit_orders)} шт., на бирже: {len(limit_orders_on_exchange)} шт.), пропускаем повторное размещение")
                        return {'success': False, 'error': 'limit_orders_already_placed', 'message': 'Лимитные ордера уже размещены'}
                    else:
                        # Ордера есть в памяти, но НЕТ на бирже - они были удалены!
                        # Очищаем память и разрешаем размещение новых ордеров
                        self.logger.warning(f" {self.symbol}: ⚠️ Лимитные ордера есть в памяти ({len(self.limit_orders)} шт.), но НЕТ на бирже - они были удалены!")
                        self.logger.info(f" {self.symbol}: 🗑️ Очищаем память от несуществующих ордеров и размещаем новые")
                        self.limit_orders = []
                        self.limit_orders_entry_price = None
                        self.last_limit_orders_count = 0
                        # Продолжаем размещение новых ордеров
                
                # ✅ КРИТИЧНО: Если режим лимитных ордеров включен - ВСЕГДА используем его, НЕ рыночный вход!
                self.logger.info(f" {self.symbol}: ✅ Режим лимитных ордеров включен, размещаем ордера...")
                return self._enter_position_with_limit_orders(side, percent_steps, margin_amounts)
            else:
                self.logger.info(f" {self.symbol}: ℹ️ Режим лимитных ордеров выключен или не настроен (enabled={limit_orders_enabled}, steps={bool(percent_steps)}, amounts={bool(margin_amounts)}), используем рыночный вход")
            
            # Стандартный рыночный вход
            # Рассчитываем размер позиции
            quantity = self._calculate_position_size()
            self.logger.info(f" {self.symbol}: Рассчитанный размер позиции: {quantity}")
            if not quantity:
                self.logger.error(f" {self.symbol}: Не удалось рассчитать размер позиции")
                return {'success': False, 'error': 'failed_to_calculate_position_size'}
            
            # Размещаем ордер
            self.logger.info(f" {self.symbol}: Размещаем {side} ордер на {quantity}...")
            # Получаем leverage из конфига бота
            leverage = self.config.get('leverage')
            self.logger.info(f" {self.symbol}: 📊 Используемое плечо из конфига: {leverage}x")
            order_result = self.exchange.place_order(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                order_type='market',
                leverage=leverage
            )
            self.logger.info(f" {self.symbol}: Результат ордера: {order_result}")
            
            if order_result.get('success'):
                try:
                    from bots_modules.imports_and_globals import set_insufficient_funds
                    set_insufficient_funds(False)
                except Exception:
                    pass
                # Обновляем состояние
                self.position = {
                    'side': side,
                    'quantity': quantity,
                    'entry_price': order_result.get('price'),
                    'order_id': order_result.get('order_id')
                }
                self.entry_price = order_result.get('price')
                self.entry_time = datetime.now()
                self.status = (BotStatus.IN_POSITION_LONG if side == 'LONG' 
                              else BotStatus.IN_POSITION_SHORT)
                
                # ✅ РЕГИСТРИРУЕМ ПОЗИЦИЮ В РЕЕСТРЕ
                try:
                    from bots_modules.imports_and_globals import register_bot_position
                    order_id = order_result.get('order_id')
                    if order_id:
                        register_bot_position(
                            symbol=self.symbol,
                            order_id=order_id,
                            side=side,
                            entry_price=order_result.get('price'),
                            quantity=quantity
                        )
                        self.logger.info(f" {self.symbol}: ✅ Позиция зарегистрирована в реестре: order_id={order_id}")
                    else:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось зарегистрировать позицию - нет order_id")
                except Exception as registry_error:
                    self.logger.error(f" {self.symbol}: ❌ Ошибка регистрации позиции в реестре: {registry_error}")
                    # Не блокируем торговлю из-за ошибки реестра
                
                # Устанавливаем стоп-лосс (с AI адаптацией если доступно)
                try:
                    # Пытаемся получить динамический SL от AI
                    sl_percent = self.max_loss_percent
                    ai_reason = None
                    
                    try:
                        from bot_engine.config_loader import AIConfig
                        if AIConfig.AI_ENABLED and AIConfig.AI_RISK_MANAGEMENT_ENABLED:
                            from bot_engine.ai import get_ai_manager
                            ai_manager = get_ai_manager()
                            
                            if ai_manager and ai_manager.risk_manager:
                                tf_use = self.config.get('entry_timeframe') or get_current_timeframe()
                                chart_response = self.exchange.get_chart_data(self.symbol, tf_use, '14d')
                                candles = chart_response.get('data', {}).get('candles', []) if chart_response and chart_response.get('success') else None
                                
                                if candles and len(candles) >= 20:
                                    dynamic_sl = ai_manager.risk_manager.calculate_dynamic_sl(
                                        self.symbol, candles, side
                                    )
                                    
                                    sl_percent = dynamic_sl['sl_percent']
                                    ai_reason = dynamic_sl['reason']
                                    
                                    self.logger.info(
                                        f" {self.symbol}: 🤖 AI адаптировал SL: "
                                        f"{self.max_loss_percent}% → {sl_percent}% "
                                        f"({ai_reason})"
                                    )
                    except Exception as ai_error:
                        pass
                    
                    # Устанавливаем стоп-лосс (стандартный или адаптивный)
                    stop_result = self._place_stop_loss(side, self.entry_price, sl_percent)
                    if stop_result and stop_result.get('success'):
                        self.logger.info(f" {self.symbol}: ✅ Стоп-лосс установлен на {sl_percent}%")
                    else:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось установить стоп-лосс")
                except Exception as stop_error:
                    self.logger.error(f" {self.symbol}: ❌ Ошибка установки стоп-лосса: {stop_error}")
                
                # ✅ Устанавливаем тейк-профит
                try:
                    from bots import bots_data, bots_data_lock
                    with bots_data_lock:
                        auto_config = bots_data.get('auto_bot_config', {})
                        take_profit_percent = auto_config.get('take_profit_percent', 20.0)
                    
                    # Рассчитываем цену тейк-профита
                    if side == 'LONG':
                        take_profit_price = self.entry_price * (1 + take_profit_percent / 100.0)
                    else:  # SHORT
                        take_profit_price = self.entry_price * (1 - take_profit_percent / 100.0)
                    
                    # Устанавливаем тейк-профит через биржу
                    # Используем метод update_take_profit если доступен, иначе через place_order с take_profit параметром
                    if hasattr(self.exchange, 'update_take_profit'):
                        tp_result = self.exchange.update_take_profit(
                            symbol=self.symbol,
                            take_profit_price=take_profit_price,
                            position_side=side
                        )
                    else:
                        # Fallback: используем place_order с параметром take_profit
                        tp_result = self.exchange.place_order(
                            symbol=self.symbol,
                            side=side,
                            quantity=quantity,
                            order_type='market',
                            take_profit=take_profit_price
                        )
                    
                    if tp_result and tp_result.get('success'):
                        self.logger.info(f" {self.symbol}: ✅ Тейк-профит установлен на {take_profit_percent}% (цена: {take_profit_price:.6f})")
                    else:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось установить тейк-профит: {tp_result.get('error', 'unknown error') if tp_result else 'no response'}")
                except Exception as tp_error:
                    self.logger.error(f" {self.symbol}: ❌ Ошибка установки тейк-профита: {tp_error}")
                
                self.logger.info(f"Entered {side} position: {quantity} at {self.entry_price}")
                return {
                    'success': True,
                    'action': 'position_entered',
                    'side': side,
                    'quantity': quantity,
                    'entry_price': self.entry_price
                }
            else:
                error_message = str(order_result.get('message', '') or order_result.get('error', ''))
                error_code = str(order_result.get('error_code', ''))
                if '30228' in error_code or '30228' in error_message or 'delisting' in error_message.lower() or 'No new positions during delisting' in error_message:
                    try:
                        from bots_modules.sync_and_cache import add_symbol_to_delisted
                        add_symbol_to_delisted(self.symbol, reason="No new positions during delisting (ErrCode: 30228)")
                    except Exception as add_err:
                        pass
                    if self.symbol not in _delisting_warned_symbols:
                        _delisting_warned_symbols.add(self.symbol)
                        self.logger.warning(f" {self.symbol}: ⚠️ Делистинг — открытие позиции запрещено биржей (ErrCode: 30228). Монета добавлена в список.")
                if error_code == 'MIN_NOTIONAL' or 'меньше минимального ордера' in error_message:
                    self.logger.warning(f" {self.symbol}: 📏 {error_message}")
                elif '110007' in error_code or '110007' in error_message:
                    self.logger.warning(f" {self.symbol}: 💰 Недостаточно средств на счёте для открытия позиции (ErrCode: 110007)")
                    try:
                        from bots_modules.imports_and_globals import set_insufficient_funds
                        set_insufficient_funds(True)
                    except Exception:
                        pass
                # MIN_NOTIONAL и недостаток средств — штатная ситуация, только WARNING
                is_expected = (
                    error_code == 'MIN_NOTIONAL' or '110007' in (error_code or '') or
                    'меньше минимального ордера' in (error_message or '') or
                    'Недостаточно доступного остатка' in (error_message or '') or
                    'Недостаточно средств' in (error_message or '') or 'баланс/маржа' in (error_message or '')
                )
                if is_expected:
                    self.logger.warning(f"Failed to enter position: {order_result}")
                else:
                    self.logger.error(f"Failed to enter position: {order_result}")
                return {'success': False, 'error': error_message or order_result.get('error', 'order_failed')}
                
        except Exception as e:
            self.logger.error(f"Error entering position: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _exit_position(self) -> Dict:
        """Выходит из позиции"""
        try:
            if not self.position:
                return {'success': False, 'error': 'no_position_to_exit'}
            
            # Размещаем ордер на закрытие
            side = 'SELL' if self.position['side'] == 'LONG' else 'BUY'
            order_result = self.exchange.place_order(
                symbol=self.symbol,
                side=side,
                quantity=self.position['quantity'],
                order_type='market'
            )
            
            if order_result.get('success'):
                exit_price = order_result.get('fill_price')
                
                # Рассчитываем PnL только если exit_price не None
                pnl = 0.0
                if exit_price is not None:
                    pnl = self._calculate_pnl(exit_price)
                
                self.logger.info(f"Exited position: PnL = {pnl}")
                
                # ✅ УДАЛЯЕМ ПОЗИЦИЮ ИЗ РЕЕСТРА
                try:
                    from bots_modules.imports_and_globals import unregister_bot_position
                    order_id = self.position.get('order_id') if self.position else None
                    if order_id:
                        unregister_bot_position(order_id)
                        self.logger.info(f" {self.symbol}: ✅ Позиция удалена из реестра: order_id={order_id}")
                    else:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось удалить позицию из реестра - нет order_id")
                except Exception as registry_error:
                    self.logger.error(f" {self.symbol}: ❌ Ошибка удаления позиции из реестра: {registry_error}")
                    # Не блокируем торговлю из-за ошибки реестра
                
                # Сбрасываем состояние
                self.position = None
                self.entry_price = None
                self.entry_time = None
                self.status = BotStatus.IDLE
                
                return {
                    'success': True,
                    'action': 'position_exited',
                    'exit_price': exit_price,
                    'pnl': pnl
                }
            else:
                error_message = order_result.get('message') or order_result.get('error', 'order_failed')
                self.logger.error(f"Failed to exit position: {order_result}")
                return {'success': False, 'error': error_message}
                
        except Exception as e:
            self.logger.error(f"Error exiting position: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _force_exit_position(self) -> Dict:
        """Принудительный выход из позиции при смене тренда"""
        self.logger.warning("Force exiting position due to trend change")
        return self._exit_position()
    
    def _calculate_position_size(self) -> Optional[float]:
        """Рассчитывает размер позиции"""
        try:
            self.logger.info(f" {self.symbol}: Рассчитываем размер позиции...")
            self.logger.info(f" {self.symbol}: volume_mode={self.volume_mode}, volume_value={self.volume_value}")
            
            if self.volume_mode == VolumeMode.FIXED_QTY or self.volume_mode == 'qty':
                self.logger.info(f" {self.symbol}: Режим FIXED_QTY, возвращаем {self.volume_value}")
                return self.volume_value
            
            elif self.volume_mode == VolumeMode.FIXED_USDT or self.volume_mode == 'usdt':
                self.logger.info(f" {self.symbol}: Режим FIXED_USDT, используем {self.volume_value} USDT")
                return self.volume_value
            
            elif self.volume_mode == VolumeMode.PERCENT_BALANCE or self.volume_mode == 'percent':
                self.logger.info(f" {self.symbol}: Режим PERCENT_BALANCE (процент от депозита)")
                deposit_balance = self._get_total_balance()
                if deposit_balance is not None and deposit_balance > 0:
                    usdt_amount = deposit_balance * (self.volume_value / 100)
                    self.logger.info(
                        f" {self.symbol}: Депозит {deposit_balance:.4f} USDT, {self.volume_value}% → {usdt_amount:.4f} USDT"
                    )
                    return usdt_amount
                else:
                    self.logger.warning(f" {self.symbol}: Не удалось получить общий баланс депозита (balance={deposit_balance})")
                    return None
            self.logger.warning(f" {self.symbol}: Неизвестный режим volume_mode: {self.volume_mode}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return None
    
    def _calculate_scaling_levels(self) -> Dict:
        """Рассчитывает уровни лесенки для текущего бота"""
        try:
            if not self.scaling_enabled:
                return {
                    'success': False,
                    'error': 'Масштабирование отключено',
                    'levels': []
                }
            
            # Получаем текущую цену
            current_price = self._get_current_price()
            if not current_price:
                return {
                    'success': False,
                    'error': 'Не удалось получить текущую цену',
                    'levels': []
                }
            
            # Получаем конфигурацию масштабирования из автобота
            scaling_config = {
                'scaling_enabled': self.scaling_enabled,
                'scaling_mode': 'auto_double',  # Пока используем только автоматическое удвоение
                'auto_double_start_percent': 1.0,
                'auto_double_max_levels': 5,
                'scaling_min_usdt_per_trade': 5.0
            }
            
            # Рассчитываем лесенку
            result = calculate_scaling_for_bot(
                base_usdt=self.volume_value,
                price=current_price,
                scaling_config=scaling_config
            )
            
            if result['success']:
                self.scaling_levels = result['levels']
                self.logger.info(f" {self.symbol}: ✅ Лесенка рассчитана: {len(result['levels'])} уровней")
                for i, level in enumerate(result['levels']):
                    self.logger.info(f" {self.symbol}: Уровень {i+1}: {level['percent']}% = {level['usdt']:.2f} USDT")
            else:
                self.logger.warning(f" {self.symbol}: ❌ Ошибка расчета лесенки: {result['error']}")
                if result.get('recommendation'):
                    rec = result['recommendation']
                    self.logger.info(f" {self.symbol}: 💡 Рекомендация: минимум {rec['min_base_usdt']:.2f} USDT для {rec['min_levels']} уровней")
            
            return result
            
        except Exception as e:
            self.logger.error(f" {self.symbol}: Ошибка расчета лесенки: {e}")
            return {
                'success': False,
                'error': str(e),
                'levels': []
            }
    
    def _get_current_price(self) -> Optional[float]:
        """Получает текущую цену с retry логикой для обработки таймаутов"""
        max_retries = 3
        retry_delay = 2  # секунды
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f" {self.symbol}: Получаем цену... (попытка {attempt + 1}/{max_retries})")
                ticker = self.exchange.get_ticker(self.symbol)
                self.logger.info(f" {self.symbol}: Ticker response: {ticker}")
                if ticker:
                    price = float(ticker.get('last', 0))
                    if price > 0:
                        self.logger.info(f" {self.symbol}: Цена получена: {price}")
                        return price
                    else:
                        self.logger.warning(f" {self.symbol}: Некорректная цена: {price}")
                else:
                    self.logger.warning(f" {self.symbol}: Ticker пустой")
                
                # Если это не последняя попытка, повторяем
                if attempt < max_retries - 1:
                    pass
                    time.sleep(retry_delay)
                    continue
                else:
                    return None
                    
            except TimeoutError as timeout_error:
                error_msg = str(timeout_error)
                self.logger.warning(f" {self.symbol}: ⏱️ Таймаут получения цены (попытка {attempt + 1}/{max_retries}): {error_msg}")
                
                # Если это не последняя попытка, повторяем
                if attempt < max_retries - 1:
                    pass
                    time.sleep(retry_delay)
                    continue
                else:
                    self.logger.error(f" {self.symbol}: ❌ Не удалось получить цену после {max_retries} попыток (таймаут)")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                # Проверяем, является ли это ошибкой таймаута (может быть в тексте ошибки)
                if 'timeout' in error_msg.lower() or 'exceeded timeout' in error_msg.lower():
                    self.logger.warning(f" {self.symbol}: ⏱️ Таймаут получения цены (попытка {attempt + 1}/{max_retries}): {error_msg}")
                    
                    # Если это не последняя попытка, повторяем
                    if attempt < max_retries - 1:
                        pass
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.logger.error(f" {self.symbol}: ❌ Не удалось получить цену после {max_retries} попыток (таймаут)")
                        return None
                else:
                    # Другая ошибка - логируем и возвращаем None
                    self.logger.error(f" {self.symbol}: ❌ Ошибка получения цены: {error_msg}")
                    return None
        
        return None
    
    def _get_wallet_balance_data(self) -> Optional[Dict]:
        """Получает словарь с данными кошелька"""
        try:
            return self.exchange.get_wallet_balance()
        except Exception as e:
            self.logger.error(f"Error getting wallet balance: {str(e)}")
            return None
    
    def _get_available_balance(self) -> Optional[float]:
        """Получает доступный баланс в USDT"""
        balance_data = self._get_wallet_balance_data()
        if not balance_data:
            return None
        try:
            v = balance_data.get('available_balance', 0)
            if v is None or v == '':
                return 0.0
            return float(v)
        except (TypeError, ValueError):
            self.logger.error("Received invalid available_balance from exchange response")
            return None

    def _get_total_balance(self) -> Optional[float]:
        """Получает общий баланс (депозит) в USDT"""
        balance_data = self._get_wallet_balance_data()
        if not balance_data:
            return None
        balance_value = balance_data.get('total_balance')
        if balance_value is None:
            balance_value = balance_data.get('available_balance')
        if balance_value is None or balance_value == '':
            return 0.0
        try:
            return float(balance_value)
        except (TypeError, ValueError):
            self.logger.error("Received invalid total_balance from exchange response")
            return None
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """Рассчитывает PnL"""
        try:
            if not self.position or not self.entry_price or exit_price is None:
                return 0.0
            
            quantity = self.position.get('quantity', 0)
            entry_price = self.entry_price
            
            if self.position['side'] == 'LONG':
                return (exit_price - entry_price) * quantity
            else:  # SHORT
                return (entry_price - exit_price) * quantity
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
            return 0.0
    
    def _get_current_state(self) -> Dict:
        """Возвращает текущее состояние бота"""
        current_price = self._get_current_price()
        current_pnl = 0.0
        
        if self.position and current_price:
            current_pnl = self._calculate_pnl(current_price)
        
        return {
            'success': True,
            'symbol': self.symbol,
            'status': self.status,
            'auto_managed': self.auto_managed,
            'trend': self.last_analysis.get('trend') if self.last_analysis else 'NEUTRAL',
            'rsi': self.last_analysis.get('rsi') if self.last_analysis else None,
            'price': current_price,
            'position': self.position,
            'pnl': current_pnl,
            'volume_mode': self.volume_mode,
            'volume_value': self.volume_value,
            'last_update': datetime.now().isoformat()
        }
    
    # Методы управления ботом
    def start(self, volume_mode: str = None, volume_value: float = None) -> Dict:
        """Запускает бота"""
        if volume_mode:
            self.volume_mode = volume_mode
        if volume_value:
            self.volume_value = volume_value
        
        if self.status == BotStatus.PAUSED:
            self.logger.info("Bot resumed from pause")
        else:
            self.status = BotStatus.IDLE
            self.logger.info("Bot started")
        
        return {'success': True, 'action': 'started'}
    
    def pause(self) -> Dict:
        """Приостанавливает бота"""
        self.status = BotStatus.PAUSED
        self.logger.info("Bot paused")
        return {'success': True, 'action': 'paused'}
    
    def stop(self) -> Dict:
        """Останавливает бота"""
        # Если есть открытая позиция, закрываем её
        if self.position:
            exit_result = self._exit_position()
            if not exit_result.get('success'):
                return exit_result
        
        self.status = BotStatus.IDLE
        self.position = None
        self.entry_price = None
        self.entry_time = None
        
        self.logger.info("Bot stopped")
        return {'success': True, 'action': 'stopped'}
    
    def force_close_position(self) -> Dict:
        """Принудительно закрывает позицию"""
        if not self.position:
            return {'success': False, 'error': 'no_position_to_close'}
        
        return self._exit_position()
    
    def get_state_dict(self) -> Dict:
        """Возвращает состояние для сохранения"""
        return {
            'symbol': self.symbol,
            'status': self.status,
            'auto_managed': self.auto_managed,
            'volume_mode': self.volume_mode,
            'volume_value': self.volume_value,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'last_bar_timestamp': self.last_bar_timestamp
        }
    
    def restore_from_state(self, state_dict: Dict):
        """Восстанавливает состояние из словаря"""
        self.status = state_dict.get('status', BotStatus.IDLE)
        self.auto_managed = state_dict.get('auto_managed', False)
        self.volume_mode = state_dict.get('volume_mode', VolumeMode.FIXED_USDT)
        self.volume_value = state_dict.get('volume_value', 10.0)
        self.position = state_dict.get('position')
        self.entry_price = state_dict.get('entry_price')
        self.last_bar_timestamp = state_dict.get('last_bar_timestamp')
        
        entry_time_str = state_dict.get('entry_time')
        if entry_time_str:
            self.entry_time = datetime.fromisoformat(entry_time_str)
        
        self.logger.info(f"Bot state restored: {self.status}")
    
    def _enter_position_with_limit_orders(self, side: str, percent_steps: List[float], margin_amounts: List[float]) -> Dict:
        """
        Входит в позицию через набор лимитных ордеров
        
        Args:
            side: 'LONG' или 'SHORT'
            percent_steps: Список процентов от цены входа [1, 2, 3, 4, 5]
            margin_amounts: Список объемов маржи в USDT [0.2, 0.3, 0.5, 1, 2]
        """
        try:
            self.logger.info(f" {self.symbol}: 🚀 Начинаем размещение лимитных ордеров: side={side}, steps={percent_steps}, amounts={margin_amounts}")
            
            # Получаем текущую цену
            current_price = self._get_current_price()
            if not current_price or current_price <= 0:
                self.logger.error(f" {self.symbol}: Не удалось получить текущую цену")
                return {'success': False, 'error': 'failed_to_get_price'}
            
            self.logger.info(f" {self.symbol}: 💰 Текущая цена: {current_price}")

            # ✅ КРИТИЧНО: Плечо из конфига — передаём в place_order для установки перед каждым ордером
            leverage = self.config.get('leverage') or self.leverage
            self.logger.info(f" {self.symbol}: 📊 Используемое плечо для лимитных ордеров: {leverage}x")
            
            # Сохраняем цену входа для расчета лимитных ордеров
            self.limit_orders_entry_price = current_price
            
            # ✅ ПРОВЕРКА: Есть ли уже открытые лимитные ордера на бирже (добавленные вручную)?
            existing_orders = []
            if hasattr(self.exchange, 'get_open_orders'):
                try:
                    existing_orders = self.exchange.get_open_orders(self.symbol)
                    if existing_orders:
                        self.logger.warning(f" {self.symbol}: ⚠️ Обнаружены существующие открытые ордера на бирже: {len(existing_orders)} шт.")
                        for order in existing_orders:
                            self.logger.warning(f" {self.symbol}:   - Ордер {order.get('order_id', 'unknown')}: {order.get('side', 'unknown')} {order.get('quantity', 0)} @ {order.get('price', 0):.6f}")
                except Exception as e:
                    pass
            
            self.limit_orders = []
            self.last_limit_orders_count = 0  # Сбрасываем счетчик при размещении новых ордеров
            
            # Проверяем, что массивы одинаковой длины
            if len(percent_steps) != len(margin_amounts):
                self.logger.error(f" {self.symbol}: Несоответствие длины массивов: percent_steps={len(percent_steps)}, margin_amounts={len(margin_amounts)}")
                return {'success': False, 'error': 'arrays_length_mismatch'}
            
            placed_orders = []
            first_order_market = False
            delisting_detected = False  # Флаг для обнаружения делистинга
            
            # Размещаем лимитные ордера
            for i, (percent_step, margin_amount) in enumerate(zip(percent_steps, margin_amounts)):
                # ✅ ПРОВЕРКА: Если обнаружен делистинг - прекращаем размещение остальных ордеров
                if delisting_detected:
                    self.logger.warning(f" {self.symbol}: ⚠️ Делистинг обнаружен, пропускаем размещение остальных ордеров")
                    break
                # ✅ КРИТИЧНО: Проверяем, что margin_amount действительно из массива, а не дефолтное значение
                if margin_amount <= 0:
                    self.logger.warning(f" {self.symbol}: ⚠️ Ордер #{i+1}: margin_amount={margin_amount} <= 0, пропускаем")
                    continue
                
                # ✅ КРИТИЧНО: Убеждаемся, что используем именно margin_amount из массива, а не self.volume_value
                actual_quantity = margin_amount  # Используем значение из массива
                
                # Если первый шаг = 0, то первая сделка по рынку
                if i == 0 and percent_step == 0:
                    first_order_market = True
                    # Размещаем рыночный ордер
                    # ✅ КРИТИЧНО: Используем margin_amount из массива, а НЕ self.volume_value!
                    actual_quantity = margin_amount
                    order_result = self.exchange.place_order(
                        symbol=self.symbol,
                        side=side,
                        quantity=actual_quantity,  # ✅ Используем значение из массива
                        order_type='market',
                        quantity_is_usdt=True,
                        skip_min_notional_enforcement=True,  # ✅ Для лимитных ордеров из набора - специальное предупреждение при увеличении до минимума
                        leverage=leverage  # ✅ Кредитное плечо
                    )
                    if order_result.get('success'):
                        order_id = order_result.get('order_id')
                        order_price = order_result.get('price', current_price)
                        placed_orders.append({
                            'order_id': order_id,
                            'type': 'market',
                            'price': order_price,
                            'quantity': margin_amount,
                            'percent_step': 0
                        })
                        self.logger.info(f" {self.symbol}: ✅ Рыночный ордер размещен: {margin_amount} USDT")
                        # Логируем в историю
                        try:
                            from bot_engine.bot_history import log_limit_order_placed
                            log_limit_order_placed(
                                bot_id=self.symbol,
                                symbol=self.symbol,
                                order_type='market',
                                order_id=str(order_id) if order_id else 'unknown',
                                price=order_price,
                                quantity=margin_amount,
                                side=side,
                                percent_step=0
                            )
                        except Exception as log_err:
                            pass
                    else:
                        # Обработка ошибки для рыночного ордера
                        error_message = order_result.get('message', 'unknown error')
                        error_code = order_result.get('error_code', '')
                        
                        # ✅ ПРОВЕРКА: Обнаружен делистинг (ErrCode: 30228)
                        if '30228' in str(error_code) or '30228' in error_message or 'delisting' in error_message.lower() or 'No new positions during delisting' in error_message:
                            delisting_detected = True
                            if self.symbol not in _delisting_warned_symbols:
                                _delisting_warned_symbols.add(self.symbol)
                                self.logger.warning(f" {self.symbol}: ⚠️ Делистинг — монета в процессе удаления с биржи (ErrCode: 30228). Монета добавлена в список.")
                            
                            # ✅ КРИТИЧНО: Автоматически добавляем монету в delisted.json
                            try:
                                from bots_modules.sync_and_cache import load_delisted_coins, save_delisted_coins
                                delisted_data = load_delisted_coins()
                                if 'delisted_coins' not in delisted_data:
                                    delisted_data['delisted_coins'] = {}
                                
                                # Добавляем монету в список делистинговых, если её там еще нет
                                if self.symbol not in delisted_data['delisted_coins']:
                                    from datetime import datetime
                                    delisted_data['delisted_coins'][self.symbol] = {
                                        'status': 'Delisting',
                                        'reason': f'Delisting detected via order placement error (ErrCode: 30228)',
                                        'delisting_date': datetime.now().strftime('%Y-%m-%d'),
                                        'detected_at': datetime.now().isoformat(),
                                        'source': 'order_placement_error_30228'
                                    }
                                    save_delisted_coins(delisted_data)
                                    self.logger.warning(f" {self.symbol}: ✅ Монета автоматически добавлена в delisted.json")
                                else:
                                    pass
                            except Exception as delisting_error:
                                self.logger.error(f" {self.symbol}: ❌ Ошибка добавления монеты в delisted.json: {delisting_error}")
                            
                            # ✅ КРИТИЧНО: Если у бота уже есть открытая позиция - закрываем её НЕМЕДЛЕННО!
                            if self.position is not None or self.status in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT]:
                                self.logger.warning(f" {self.symbol}: 🚨 ОТКРЫТАЯ ПОЗИЦИЯ ОБНАРУЖЕНА ПРИ ДЕЛИСТИНГЕ! Закрываем немедленно!")
                                try:
                                    from bots_modules.bot_class import NewTradingBot
                                    from bots_modules.imports_and_globals import get_exchange
                                    from bots_modules.sync_and_cache import bots_data, bots_data_lock
                                    
                                    with bots_data_lock:
                                        if self.symbol in bots_data.get('bots', {}):
                                            bot_data = bots_data['bots'][self.symbol]
                                            exchange_obj = get_exchange()
                                            if exchange_obj:
                                                bot_instance = NewTradingBot(self.symbol, bot_data, exchange_obj)
                                                emergency_result = bot_instance.emergency_close_delisting()
                                                if emergency_result:
                                                    self.logger.warning(f" {self.symbol}: ✅ ЭКСТРЕННОЕ ЗАКРЫТИЕ УСПЕШНО")
                                                else:
                                                    self.logger.error(f" {self.symbol}: ❌ ЭКСТРЕННОЕ ЗАКРЫТИЕ НЕУДАЧНО")
                                except Exception as emergency_close_error:
                                    self.logger.error(f" {self.symbol}: ❌ Ошибка экстренного закрытия позиции: {emergency_close_error}")
                            
                            # Прекращаем размещение остальных ордеров
                            break
                        
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось разместить рыночный ордер: {error_message}")
                    continue
                
                # Рассчитываем цену лимитного ордера
                if side == 'LONG':
                    # Для лонга: цена ниже текущей на percent_step%
                    limit_price = current_price * (1 - percent_step / 100)
                else:  # SHORT
                    # Для шорта: цена выше текущей на percent_step%
                    limit_price = current_price * (1 + percent_step / 100)
                
                # Размещаем лимитный ордер
                # ✅ КРИТИЧНО: Используем margin_amount из массива, а НЕ self.volume_value!
                actual_quantity = margin_amount
                order_result = self.exchange.place_order(
                    symbol=self.symbol,
                    side=side,
                    quantity=actual_quantity,  # ✅ Используем значение из массива
                    order_type='limit',
                    price=limit_price,
                    quantity_is_usdt=True,
                    skip_min_notional_enforcement=True,  # ✅ Для лимитных ордеров из набора не принуждаем к minNotionalValue
                    leverage=leverage  # ✅ Кредитное плечо
                )
                
                if order_result.get('success'):
                    order_id = order_result.get('order_id')
                    order_info = {
                        'order_id': order_id,
                        'type': 'limit',
                        'price': limit_price,
                        'quantity': margin_amount,
                        'percent_step': percent_step
                    }
                    placed_orders.append(order_info)
                    self.limit_orders.append(order_info)
                    self.logger.info(f" {self.symbol}: ✅ Лимитный ордер #{i+1} размещен: {margin_amount} USDT @ {limit_price:.6f} ({percent_step}%)")
                    # Логируем в историю
                    try:
                        from bot_engine.bot_history import log_limit_order_placed
                        log_limit_order_placed(
                            bot_id=self.symbol,
                            symbol=self.symbol,
                            order_type='limit',
                            order_id=str(order_id) if order_id else 'unknown',
                            price=limit_price,
                            quantity=margin_amount,
                            side=side,
                            percent_step=percent_step
                        )
                    except Exception as log_err:
                        pass
                else:
                    error_message = order_result.get('message', 'unknown error')
                    error_code = order_result.get('error_code', '')
                    
                    # ✅ ПРОВЕРКА: Обнаружен делистинг (ErrCode: 30228)
                    if '30228' in str(error_code) or 'delisting' in error_message.lower() or 'No new positions during delisting' in error_message:
                        delisting_detected = True
                        if self.symbol not in _delisting_warned_symbols:
                            _delisting_warned_symbols.add(self.symbol)
                            self.logger.warning(f" {self.symbol}: ⚠️ Делистинг — монета в процессе удаления с биржи (ErrCode: 30228). Монета добавлена в список.")
                        
                        # ✅ КРИТИЧНО: Автоматически добавляем монету в delisted.json
                        try:
                            from bots_modules.sync_and_cache import load_delisted_coins, save_delisted_coins
                            delisted_data = load_delisted_coins()
                            if 'delisted_coins' not in delisted_data:
                                delisted_data['delisted_coins'] = {}
                            
                            # Добавляем монету в список делистинговых, если её там еще нет
                            if self.symbol not in delisted_data['delisted_coins']:
                                from datetime import datetime
                                delisted_data['delisted_coins'][self.symbol] = {
                                    'status': 'Delisting',
                                    'reason': f'Delisting detected via order placement error (ErrCode: 30228)',
                                    'delisting_date': datetime.now().strftime('%Y-%m-%d'),
                                    'detected_at': datetime.now().isoformat(),
                                    'source': 'order_placement_error_30228'
                                }
                                save_delisted_coins(delisted_data)
                                self.logger.warning(f" {self.symbol}: ✅ Монета автоматически добавлена в delisted.json")
                            else:
                                pass
                        except Exception as delisting_error:
                            self.logger.error(f" {self.symbol}: ❌ Ошибка добавления монеты в delisted.json: {delisting_error}")
                        
                        # ✅ КРИТИЧНО: Если у бота уже есть открытая позиция - закрываем её НЕМЕДЛЕННО!
                        if self.position is not None or self.status in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT]:
                            self.logger.warning(f" {self.symbol}: 🚨 ОТКРЫТАЯ ПОЗИЦИЯ ОБНАРУЖЕНА ПРИ ДЕЛИСТИНГЕ! Закрываем немедленно!")
                            try:
                                from bots_modules.bot_class import NewTradingBot
                                from bots_modules.imports_and_globals import get_exchange
                                from bots_modules.sync_and_cache import bots_data, bots_data_lock
                                
                                with bots_data_lock:
                                    if self.symbol in bots_data.get('bots', {}):
                                        bot_data = bots_data['bots'][self.symbol]
                                        exchange_obj = get_exchange()
                                        if exchange_obj:
                                            bot_instance = NewTradingBot(self.symbol, bot_data, exchange_obj)
                                            emergency_result = bot_instance.emergency_close_delisting()
                                            if emergency_result:
                                                self.logger.warning(f" {self.symbol}: ✅ ЭКСТРЕННОЕ ЗАКРЫТИЕ УСПЕШНО")
                                            else:
                                                self.logger.error(f" {self.symbol}: ❌ ЭКСТРЕННОЕ ЗАКРЫТИЕ НЕУДАЧНО")
                            except Exception as emergency_close_error:
                                self.logger.error(f" {self.symbol}: ❌ Ошибка экстренного закрытия позиции: {emergency_close_error}")
                        
                        # Прекращаем размещение остальных ордеров
                        break
                    
                    self.logger.warning(f" {self.symbol}: ⚠️ Не удалось разместить лимитный ордер #{i+1}: {error_message}")
            
            if not placed_orders:
                self.logger.error(f" {self.symbol}: Не удалось разместить ни одного ордера")
                return {'success': False, 'error': 'no_orders_placed'}
            
            # Обновляем счетчик активных лимитных ордеров
            self.last_limit_orders_count = len(self.limit_orders)
            
            # Если был рыночный ордер, обновляем позицию
            if first_order_market and placed_orders:
                market_order = placed_orders[0]
                self.position = {
                    'side': side,
                    'quantity': market_order['quantity'],
                    'entry_price': market_order['price'],
                    'order_id': market_order['order_id']
                }
                self.entry_price = market_order['price']
                self.entry_time = datetime.now()
                self.status = (BotStatus.IN_POSITION_LONG if side == 'LONG' 
                              else BotStatus.IN_POSITION_SHORT)
            
            self.logger.info(f" {self.symbol}: ✅ Набор позиций начат: {len(placed_orders)} ордеров размещено")
            return {
                'success': True,
                'action': 'limit_orders_placed',
                'side': side,
                'orders_count': len(placed_orders),
                'orders': placed_orders,
                'entry_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка размещения лимитных ордеров: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _check_and_cancel_limit_orders_if_needed(self, analysis: Dict) -> None:
        """
        Проверяет RSI и отменяет лимитные ордера при выходе за зону набора позиций
        
        Для LONG: отменяем если RSI > rsi_time_filter_lower (35)
        Для SHORT: отменяем если RSI < rsi_time_filter_upper (65)
        """
        if not self.limit_orders:
            return
        
        try:
            # Получаем текущий RSI
            current_rsi = analysis.get('rsi')
            if current_rsi is None:
                return
            
            # Получаем границы из конфига
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from bots import bots_data, bots_data_lock
                
                with bots_data_lock:
                    auto_config = bots_data.get('auto_bot_config', {})
                    rsi_time_filter_lower = auto_config.get('rsi_time_filter_lower', 35)
                    rsi_time_filter_upper = auto_config.get('rsi_time_filter_upper', 65)
            except Exception as e:
                self.logger.warning(f" {self.symbol}: Не удалось получить границы RSI: {e}")
                rsi_time_filter_lower = 35
                rsi_time_filter_upper = 65
            
            # Определяем направление по первому ордеру или позиции
            side = None
            if self.position:
                side = self.position.get('side')
            elif self.limit_orders:
                # Определяем по цене лимитного ордера относительно текущей цены
                current_price = self._get_current_price()
                if current_price and self.limit_orders_entry_price:
                    if self.limit_orders[0].get('price', 0) < current_price:
                        side = 'LONG'  # Лимитный ордер ниже цены = покупка
                    else:
                        side = 'SHORT'  # Лимитный ордер выше цены = продажа
            
            if not side:
                return
            
            should_cancel = False
            reason = ""
            
            if side == 'LONG':
                # Для лонга: отменяем если RSI выше нижней границы
                if current_rsi > rsi_time_filter_lower:
                    should_cancel = True
                    reason = f"RSI {current_rsi:.2f} > {rsi_time_filter_lower} (выход из зоны набора LONG)"
            else:  # SHORT
                # Для шорта: отменяем если RSI ниже верхней границы
                if current_rsi < rsi_time_filter_upper:
                    should_cancel = True
                    reason = f"RSI {current_rsi:.2f} < {rsi_time_filter_upper} (выход из зоны набора SHORT)"
            
            if should_cancel:
                self.logger.info(f" {self.symbol}: 🚫 Отменяем лимитные ордера: {reason}")
                self._cancel_all_limit_orders()
        
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки лимитных ордеров: {e}")
    
    def _remove_cancelled_orders_from_list(self) -> None:
        """
        Проверяет статус ордеров на бирже и удаляет из списка те, которые были отменены вручную
        """
        if not self.limit_orders:
            return
        
        try:
            # Получаем открытые ордера с биржи (если метод доступен)
            open_orders = []
            if hasattr(self.exchange, 'get_open_orders'):
                try:
                    orders_result = self.exchange.get_open_orders(self.symbol)
                    if orders_result and isinstance(orders_result, list):
                        open_orders = orders_result
                    elif orders_result and isinstance(orders_result, dict):
                        open_orders = orders_result.get('orders', [])
                except Exception as e:
                    pass
            
            # Если метод недоступен, пытаемся проверить через попытку отмены
            # (если ордер не существует, отмена вернет ошибку)
            if not hasattr(self.exchange, 'get_open_orders'):
                orders_to_remove = []
                for order_info in self.limit_orders[:]:
                    order_id = order_info.get('order_id')
                    if not order_id:
                        continue
                    
                    # Пытаемся проверить статус через попытку отмены
                    # Если ордер не существует, метод вернет ошибку
                    try:
                        if hasattr(self.exchange, 'cancel_order'):
                            # Проверяем, существует ли ордер, пытаясь его отменить
                            # Если ордер уже отменен/не существует, получим ошибку
                            # Но это не идеальный способ, так как мы не хотим отменять существующие ордера
                            # Поэтому просто пропускаем проверку, если метод get_open_orders недоступен
                            pass
                    except Exception:
                        # Если ошибка при проверке - оставляем ордер в списке (безопаснее)
                        pass
                
                # Если метод проверки недоступен, просто логируем предупреждение
                pass
                return
            
            # Создаем множество ID открытых ордеров на бирже
            open_order_ids = set()
            for order in open_orders:
                order_id = str(order.get('orderId') or order.get('order_id') or order.get('id', ''))
                if order_id:
                    open_order_ids.add(order_id)
            
            # Удаляем из списка ордера, которых нет на бирже
            removed_count = 0
            for order_info in self.limit_orders[:]:
                order_id = str(order_info.get('order_id', ''))
                if order_id and order_id not in open_order_ids:
                    # Ордер был удален вручную на бирже
                    self.limit_orders.remove(order_info)
                    removed_count += 1
                    self.logger.warning(f" {self.symbol}: ⚠️ Лимитный ордер {order_id} был удален вручную на бирже, удаляем из списка")
            
            if removed_count > 0:
                self.logger.info(f" {self.symbol}: 🗑️ Удалено {removed_count} несуществующих ордеров из списка")
                # Обновляем счетчик
                self.last_limit_orders_count = len(self.limit_orders)
        
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки статуса ордеров: {e}")
    
    def _cancel_all_limit_orders(self) -> None:
        """Отменяет все активные лимитные ордера"""
        if not self.limit_orders:
            return
        
        cancelled_count = 0
        for order_info in self.limit_orders[:]:  # Копируем список для безопасной итерации
            try:
                order_id = order_info.get('order_id')
                if not order_id:
                    continue
                
                # Используем метод биржи для отмены ордера
                # Проверяем, есть ли метод cancel_order
                if hasattr(self.exchange, 'cancel_order'):
                    cancel_result = self.exchange.cancel_order(
                        symbol=self.symbol,
                        order_id=order_id
                    )
                    if cancel_result and cancel_result.get('success'):
                        cancelled_count += 1
                        self.logger.info(f" {self.symbol}: ✅ Лимитный ордер {order_id} отменен")
                    else:
                        self.logger.warning(f" {self.symbol}: ⚠️ Не удалось отменить ордер {order_id}")
                else:
                    # Если метода нет, пытаемся через универсальный API
                    # Для Bybit можно использовать client.cancel_order
                    self.logger.warning(f" {self.symbol}: ⚠️ Метод cancel_order не найден, пропускаем ордер {order_id}")
                
            except Exception as e:
                self.logger.error(f" {self.symbol}: ❌ Ошибка отмены ордера {order_info.get('order_id')}: {e}")
        
        # Очищаем список лимитных ордеров
        total_orders = len(self.limit_orders)
        self.limit_orders = []
        self.limit_orders_entry_price = None
        self.last_limit_orders_count = 0  # Сбрасываем счетчик при отмене всех ордеров
        self.logger.info(f" {self.symbol}: ✅ Отменено лимитных ордеров: {cancelled_count}/{total_orders}")
    
    def _check_and_update_limit_orders_fills(self) -> None:
        """
        Проверяет сработавшие лимитные ордера, пересчитывает среднюю цену входа
        и обновляет стоп-лосс ТОЛЬКО при срабатывании нового ордера
        
        Также проверяет, не были ли ордера удалены вручную на бирже
        """
        if not self.limit_orders:
            # Если нет активных ордеров, обновляем счетчик
            self.last_limit_orders_count = 0
            return
        
        try:
            # ✅ ПРОВЕРКА 1: Проверяем, существуют ли ордера на бирже
            # Если ордер был удален вручную, удаляем его из списка
            self._remove_cancelled_orders_from_list()
            
            # Сохраняем текущее количество ордеров ДО проверки
            current_orders_count = len(self.limit_orders)
            
            # Если после проверки ордеров список пуст, выходим
            if not self.limit_orders:
                self.last_limit_orders_count = 0
                return
            
            # Получаем реальную позицию с биржи
            exchange_positions = self.exchange.get_positions()
            if isinstance(exchange_positions, tuple):
                positions_list = exchange_positions[0] if exchange_positions else []
            else:
                positions_list = exchange_positions if exchange_positions else []
            
            # Ищем позицию по нашему символу
            real_position = None
            for pos in positions_list:
                if pos.get('symbol') == self.symbol and abs(float(pos.get('size', 0))) > 0:
                    real_position = pos
                    break
            
            if not real_position:
                # Если позиции нет, но есть лимитные ордера - это нормально (еще не сработали)
                return
            
            # Получаем данные реальной позиции
            real_size = abs(float(real_position.get('size', 0)))
            real_avg_price = float(real_position.get('avg_price', 0))
            real_side = real_position.get('side', '')
            
            # Определяем сторону позиции
            if real_side.upper() in ['LONG', 'BUY']:
                side = 'LONG'
            elif real_side.upper() in ['SHORT', 'SELL']:
                side = 'SHORT'
            else:
                return
            
            # Получаем текущий размер позиции в боте
            current_bot_size = self.position.get('quantity', 0) if self.position else 0
            current_bot_price = self.position.get('entry_price', 0) if self.position else 0
            
            # ✅ ПРОВЕРКА: Рассчитываем ожидаемый размер позиции от всех наших лимитных ордеров
            # Это поможет обнаружить "чужие" ордера, добавленные вручную
            expected_size_from_orders = sum(order.get('quantity', 0) for order in self.limit_orders)
            if self.position:
                expected_total_size = current_bot_size + expected_size_from_orders
            else:
                expected_total_size = expected_size_from_orders
            
            # Проверяем, не превышает ли реальный размер ожидаемый (значит есть "чужие" ордера)
            if real_size > expected_total_size * 1.01:  # 1% допуск
                extra_size = real_size - expected_total_size
                self.logger.warning(f" {self.symbol}: ⚠️ Обнаружено несоответствие размера позиции! Реальный: {real_size:.6f}, ожидаемый от наших ордеров: {expected_total_size:.6f}, разница: {extra_size:.6f}")
                self.logger.warning(f" {self.symbol}: ⚠️ Возможно, на бирже есть лимитные ордера, добавленные вручную, или сработали ордера, которых нет в нашем списке")
            
            # Проверяем, изменилась ли позиция на бирже
            # Если размер увеличился или средняя цена изменилась, значит сработали ордера
            size_changed = abs(real_size - current_bot_size) > 0.001
            price_changed = current_bot_price > 0 and abs(real_avg_price - current_bot_price) / current_bot_price > 0.001
            
            if size_changed or price_changed:
                # Позиция изменилась - пересчитываем среднюю цену входа
                # Используем реальную среднюю цену с биржи (она уже рассчитана с учетом всех сработавших ордеров)
                
                # ✅ УЛУЧШЕННАЯ ЛОГИКА: Проверяем статус ордеров на бирже напрямую
                # Это позволяет точно определить, какие ордера сработали, даже если на бирже есть чужие ордера
                orders_to_remove = []
                
                # Получаем открытые ордера с биржи для проверки статуса
                open_orders_on_exchange = []
                if hasattr(self.exchange, 'get_open_orders'):
                    try:
                        orders_result = self.exchange.get_open_orders(self.symbol)
                        if orders_result and isinstance(orders_result, list):
                            open_orders_on_exchange = orders_result
                        elif orders_result and isinstance(orders_result, dict):
                            open_orders_on_exchange = orders_result.get('orders', [])
                    except Exception as e:
                        pass
                
                # Создаем множество ID открытых ордеров на бирже
                open_order_ids_on_exchange = set()
                for order in open_orders_on_exchange:
                    order_id = str(order.get('orderId') or order.get('order_id') or order.get('id', ''))
                    if order_id:
                        open_order_ids_on_exchange.add(order_id)
                
                # Проверяем каждый ордер из списка бота
                for order_info in self.limit_orders:
                    order_id = str(order_info.get('order_id', ''))
                    order_price = order_info.get('price', 0)
                    
                    # ✅ МЕТОД 1: Проверяем по статусу на бирже (наиболее точный)
                    if order_id and open_order_ids_on_exchange:
                        # Если ордера нет в списке открытых на бирже - он сработал или был отменен
                        if order_id not in open_order_ids_on_exchange:
                            orders_to_remove.append(order_info)
                            continue
                    
                    # ✅ МЕТОД 2: Если метод проверки недоступен, используем проверку по цене (fallback)
                    # Но только если размер позиции увеличился (значит ордера сработали)
                    if not open_order_ids_on_exchange and size_changed and real_size > current_bot_size:
                        # Проверяем по цене только если позиция увеличилась
                        if side == 'LONG':
                            # Для лонга: ордер сработал, если его цена ниже или равна текущей средней
                            if order_price <= real_avg_price * 1.01:  # 1% допуск
                                orders_to_remove.append(order_info)
                        else:  # SHORT
                            # Для шорта: ордер сработал, если его цена выше или равна текущей средней
                            if order_price >= real_avg_price * 0.99:  # 1% допуск
                                orders_to_remove.append(order_info)
                
                # Удаляем сработавшие ордера из списка активных
                orders_removed_count = 0
                for order_info in orders_to_remove:
                    if order_info in self.limit_orders:
                        self.limit_orders.remove(order_info)
                        orders_removed_count += 1
                        order_id = order_info.get('order_id', 'unknown')
                        self.logger.info(f" {self.symbol}: ✅ Лимитный ордер сработал: {order_info.get('quantity', 0)} USDT @ {order_info.get('price', 0):.6f} (ID: {order_id})")
                
                # КРИТИЧНО: Пересчитываем стоп-лосс ТОЛЬКО если действительно сработал новый ордер
                # Проверяем, изменилось ли количество активных ордеров
                new_orders_count = len(self.limit_orders)
                order_filled = (new_orders_count < self.last_limit_orders_count) or (orders_removed_count > 0)
                
                if order_filled:
                    # Обновляем позицию с реальными данными с биржи
                    self.position = {
                        'side': side,
                        'quantity': real_size,  # Реальный размер с биржи
                        'entry_price': real_avg_price,  # Реальная средняя цена входа с биржи
                        'order_id': 'limit_orders_filled'
                    }
                    self.entry_price = real_avg_price
                    
                    # Если позиция еще не была установлена, обновляем статус
                    if self.status not in [BotStatus.IN_POSITION_LONG, BotStatus.IN_POSITION_SHORT]:
                        self.status = (BotStatus.IN_POSITION_LONG if side == 'LONG' 
                                      else BotStatus.IN_POSITION_SHORT)
                        if not self.entry_time:
                            self.entry_time = datetime.now()
                    
                    self.logger.info(f" {self.symbol}: 📊 Обновлена позиция: {side} {real_size:.6f} @ {real_avg_price:.6f} (средняя цена с биржи)")
                    
                    # Пересчитываем и обновляем стоп-лосс от новой средней цены ТОЛЬКО при срабатывании ордера
                    try:
                        # Получаем процент стоп-лосса из конфига
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from bots import bots_data, bots_data_lock
                        
                        with bots_data_lock:
                            auto_config = bots_data.get('auto_bot_config', {})
                            loss_percent = auto_config.get('max_loss_percent', 15.0)
                        
                        # Обновляем стоп-лосс от новой средней цены
                        stop_result = self._place_stop_loss(side, real_avg_price, loss_percent)
                        if stop_result.get('success'):
                            self.logger.info(f" {self.symbol}: ✅ Стоп-лосс обновлен: {stop_result.get('stop_price'):.6f} (от средней цены {real_avg_price:.6f})")
                        else:
                            self.logger.warning(f" {self.symbol}: ⚠️ Не удалось обновить стоп-лосс: {stop_result.get('error')}")
                    except Exception as e:
                        self.logger.error(f" {self.symbol}: ❌ Ошибка обновления стоп-лосса: {e}")
                    
                    # Обновляем счетчик активных ордеров
                    self.last_limit_orders_count = new_orders_count
                else:
                    # Ордера не сработали, просто обновляем позицию без пересчета стоп-лосса
                    if size_changed:
                        self.position = {
                            'side': side,
                            'quantity': real_size,
                            'entry_price': real_avg_price,
                            'order_id': self.position.get('order_id', 'limit_orders_filled') if self.position else 'limit_orders_filled'
                        }
                        self.entry_price = real_avg_price
            else:
                # Позиция не изменилась - обновляем только счетчик
                self.last_limit_orders_count = current_orders_count
        
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка проверки лимитных ордеров: {e}")
            import traceback
            traceback.print_exc()
    
    def _place_stop_loss(self, side: str, entry_price: float, loss_percent: float) -> Dict:
        """Устанавливает стоп-лосс для позиции"""
        try:
            if not entry_price or entry_price <= 0:
                self.logger.error(f" {self.symbol}: Некорректная цена входа для стоп-лосса: {entry_price}")
                return {'success': False, 'error': 'invalid_entry_price'}
            
            # Рассчитываем цену стоп-лосса
            if side == 'LONG':
                # Для лонга: стоп-лосс ниже цены входа
                stop_price = entry_price * (1 - loss_percent / 100)
            else:  # SHORT
                # Для шорта: стоп-лосс выше цены входа
                stop_price = entry_price * (1 + loss_percent / 100)
            
            self.logger.info(f" {self.symbol}: Устанавливаем стоп-лосс: {side} @ {stop_price:.6f} (потеря: {loss_percent}%)")
            
            # Размещаем стоп-лосс ордер (делегируем бирже расчет финального ордера)
            stop_result = self.exchange.place_stop_loss(
                symbol=self.symbol,
                side=side,
                entry_price=entry_price,
                loss_percent=loss_percent
            )
            
            if stop_result and stop_result.get('success'):
                self.logger.info(f" {self.symbol}: ✅ Стоп-лосс установлен успешно")
                return {'success': True, 'stop_price': stop_price, 'order_id': stop_result.get('order_id')}
            else:
                self.logger.warning(f" {self.symbol}: ⚠️ Не удалось установить стоп-лосс: {stop_result}")
                return {'success': False, 'error': stop_result.get('error', 'stop_loss_failed')}
                
        except Exception as e:
            self.logger.error(f" {self.symbol}: ❌ Ошибка установки стоп-лосса: {e}")
            return {'success': False, 'error': str(e)}
