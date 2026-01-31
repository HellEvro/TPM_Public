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
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger('BotsService')

# Импортируем глобальные переменные
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
        BOT_STATUS, get_exchange, system_initialized, get_auto_bot_config
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

# Импорт AI Risk Manager для умного расчета TP/SL
try:
    from bot_engine.ai.risk_manager import DynamicRiskManager
    AI_RISK_MANAGER_AVAILABLE = True
except ImportError:
    DynamicRiskManager = None
    AI_RISK_MANAGER_AVAILABLE = False

try:
    from bot_engine.protections import ProtectionState, evaluate_protections
except ImportError:
    @dataclass
    class ProtectionState:
        position_side: str = 'LONG'
        entry_price: float = 0.0
        entry_time: Optional[float] = None
        quantity: Optional[float] = None
        notional_usdt: Optional[float] = None
        max_profit_percent: float = 0.0
        break_even_activated: bool = False
        break_even_stop_set: bool = False  # Флаг, что break-even стоп уже установлен на бирже (устанавливается один раз)
        break_even_stop_price: Optional[float] = None
        trailing_active: bool = False
        trailing_reference_price: Optional[float] = None
        trailing_stop_price: Optional[float] = None
        trailing_take_profit_price: Optional[float] = None
        trailing_last_update_ts: float = 0.0

    def evaluate_protections(*args, **kwargs):
        class _Decision:
            should_close = False
            reason = None
            state = ProtectionState()
            profit_percent = 0.0
        return _Decision()

class NewTradingBot:
    """Новый торговый бот согласно требованиям"""
    
    BREAK_EVEN_FEE_MULTIPLIER = 2.5
    
    @staticmethod
    def _safe_float(value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def __init__(self, symbol, config=None, exchange=None):
        self.symbol = symbol
        self.config = config or {}
        self.exchange = exchange
        
        # КРИТИЧНО: НЕ логируем BOT_START здесь - это будет сделано в create_bot() после сохранения в bots_data
        # Это предотвращает логирование временных ботов, которые не сохраняются в bots_state.json
        
        # Параметры сделки из конфига
        self.volume_mode = self.config.get('volume_mode', 'usdt')
        self.volume_value = self.config.get('volume_value', 10.0)
        
        # Состояние бота
        self.status = self.config.get('status', BOT_STATUS['IDLE'])
        self.entry_price = self.config.get('entry_price', None)
        self.position_side = self.config.get('position_side', None)
        self.position_size = self.config.get('position_size', None)  # Размер позиции в монетах
        self.position_size_coins = self.config.get('position_size_coins', None)
        self.unrealized_pnl = self.config.get('unrealized_pnl', 0.0)
        self.unrealized_pnl_usdt = self.config.get('unrealized_pnl_usdt', 0.0)
        self.realized_pnl = self.config.get('realized_pnl', 0.0)
        self.leverage = self.config.get('leverage', 1.0)
        self.margin_usdt = self.config.get('margin_usdt', None)
        self.trailing_activation_profit = self.config.get('trailing_activation_profit', 0.0)
        self.trailing_locked_profit = self.config.get('trailing_locked_profit', 0.0)
        self.created_at = self.config.get('created_at', datetime.now().isoformat())
        self.last_signal_time = self.config.get('last_signal_time', None)
        
        # Защитные механизмы
        self.max_profit_achieved = self.config.get('max_profit_achieved', 0.0)
        self.trailing_stop_price = self.config.get('trailing_stop_price', None)
        self.break_even_activated = bool(self.config.get('break_even_activated', False))
        self.break_even_stop_set = bool(self.config.get('break_even_stop_set', False))  # Флаг, что защитный стоп уже установлен на бирже
        # ✅ ИСПРАВЛЕНО: break_even_stop_price должен быть None, если защита не активирована
        if self.break_even_activated:
            break_even_stop = self.config.get('break_even_stop_price')
            try:
                self.break_even_stop_price = float(break_even_stop) if break_even_stop is not None else None
            except (TypeError, ValueError):
                self.break_even_stop_price = None
        else:
            self.break_even_stop_price = None
            self.break_even_stop_set = False
        self.trailing_activation_threshold = self.config.get('trailing_activation_threshold', 0.0)
        self.trailing_active = bool(self.config.get('trailing_active', False))
        self.trailing_max_profit_usdt = float(self.config.get('trailing_max_profit_usdt', 0.0) or 0.0)
        self.trailing_step_usdt = float(self.config.get('trailing_step_usdt', 0.0) or 0.0)
        self.trailing_step_price = float(self.config.get('trailing_step_price', 0.0) or 0.0)
        self.trailing_steps = int(self.config.get('trailing_steps', 0) or 0)
        entry_price_float = self._safe_float(self.entry_price)
        self.trailing_reference_price = self._safe_float(
            self.config.get('trailing_reference_price'),
            entry_price_float
        )
        self.trailing_take_profit_price = self._safe_float(self.config.get('trailing_take_profit_price'))
        self.trailing_last_update_ts = self._safe_float(self.config.get('trailing_last_update_ts'), 0.0) or 0.0
        self.trailing_take_profit_price = self.config.get('trailing_take_profit_price', None)
        
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
        
        # Дополнительные поля для сохранения
        self.stop_loss = self.config.get('stop_loss', None)
        self.take_profit = self.config.get('take_profit', None)
        self.current_price = self.config.get('current_price', None)
        
        # ✅ Тренд при входе в позицию (для определения уровня RSI выхода)
        self.entry_trend = self.config.get('entry_trend', None)
        
        # ✅ Таймфрейм при входе в позицию (критично для работы с позициями из разных ТФ)
        # По умолчанию — выбранный системный ТФ; для старых записей из БД может быть значение из конфига
        try:
            from bot_engine.bot_config import get_current_timeframe
            _default_tf = get_current_timeframe()
        except Exception:
            # ✅ КРИТИЧНО: Используем TIMEFRAME из конфига вместо хардкода '6h'
            from bot_engine.bot_config import TIMEFRAME
            _default_tf = TIMEFRAME
        self.entry_timeframe = self.config.get('entry_timeframe') or _default_tf

        # AI метаданные
        self.ai_decision_id = self.config.get('ai_decision_id')
        self._last_decision_source = 'SCRIPT'
        self._last_ai_decision_meta = None
        self._last_entry_context = {}
        
        
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
            # Логируем открытие позиции, если это переход из не-позиционного статуса
            # и позиция действительно открыта (есть entry_price и position_size)
            was_in_position = old_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]
            has_entry_price = self.entry_price and self.entry_price > 0
            has_position_size = (self.position_size and self.position_size > 0) or (self.position_size_coins and self.position_size_coins > 0)
            
            # КРИТИЧНО: Проверяем, не логировали ли мы уже эту позицию в истории
            # Это предотвращает дублирование при синхронизации с биржей
            # НО: если позиция есть с EXCHANGE_IMPORT, а бот реально активен - логируем с SCRIPT
            position_already_logged_by_bot = False
            if has_entry_price:
                try:
                    from bot_engine.bot_history import bot_history_manager
                    with bot_history_manager.lock:
                        for existing_trade in bot_history_manager.trades:
                            if (existing_trade.get('symbol') == self.symbol and
                                existing_trade.get('status') == 'OPEN' and
                                existing_trade.get('direction') == (self.position_side or 'LONG')):
                                existing_entry_price = existing_trade.get('entry_price')
                                if existing_entry_price and abs(float(existing_entry_price) - float(self.entry_price)) < 0.0001:
                                    # Проверяем decision_source - если это SCRIPT или AI, то бот уже залогировал
                                    existing_source = existing_trade.get('decision_source', '')
                                    if existing_source in ('SCRIPT', 'AI'):
                                        position_already_logged_by_bot = True
                                        break
                                    # Если это EXCHANGE_IMPORT - это нормально, бот должен залогировать свою версию
                                    elif existing_source == 'EXCHANGE_IMPORT':
                                        break
                except Exception:
                    pass
            
            # Логируем только если:
            # 1. Это переход в позицию (не был в позиции)
            # 2. Есть данные о позиции (цена входа и размер)
            # 3. Это не повторный вызов (проверяем через флаг _position_logged)
            # 4. Позиция еще не залогирована ботом (SCRIPT/AI) - EXCHANGE_IMPORT не считается
            if not was_in_position and has_entry_price and has_position_size and not position_already_logged_by_bot:
                # Проверяем, не логировали ли мы уже эту позицию
                position_logged = getattr(self, '_position_logged', False)
                if not position_logged:
                    logger.info(f"[NEW_BOT_{self.symbol}] 📝 Логируем открытие позиции в bot_history.json")
                    try:
                        self._on_position_opened(
                            direction=self.position_side or (new_status.split('_')[-1] if '_' in new_status else 'LONG'),
                            entry_price=self.entry_price,
                            position_size=self.position_size or self.position_size_coins
                        )
                        self._position_logged = True  # Помечаем, что логирование выполнено
                        logger.info(f"[NEW_BOT_{self.symbol}] ✅ Позиция успешно записана в bot_history.json")
                    except Exception as log_error:
                        logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось записать историю при update_status: {log_error}")
                else:
                    pass
            else:
                if not was_in_position:
                    reason = []
                    if position_already_logged_by_bot:
                        reason.append("уже залогирована ботом")
                    if not has_entry_price:
                        reason.append("нет entry_price")
                    if not has_position_size:
                        reason.append("нет position_size")
            
            self.position_start_time = datetime.now()
            self.max_profit_achieved = 0.0
            self.trailing_stop_price = None
            self.break_even_activated = False
            self.break_even_stop_price = None
            self.break_even_stop_set = False
            self.trailing_active = False
            self.trailing_activation_profit = 0.0
            self.trailing_activation_threshold = 0.0
            self.trailing_locked_profit = 0.0
            self.trailing_max_profit_usdt = 0.0
            self.trailing_step_usdt = 0.0
            self.trailing_step_price = 0.0
            self.trailing_steps = 0
            current_entry = self._safe_float(self.entry_price)
            self.trailing_reference_price = current_entry
            self.trailing_take_profit_price = None
            self.trailing_last_update_ts = 0.0
            self.trailing_take_profit_price = None
        else:
            # При выходе из позиции сбрасываем флаг логирования
            if old_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                self._position_logged = False
    
    def _remember_entry_context(self, rsi: Optional[float], trend: Optional[str]):
        """Сохраняет рыночный контекст последнего входа."""
        self._last_entry_context = {
            'rsi': rsi,
            'trend': trend
        }
    
    def _set_decision_source(self, source: str = 'SCRIPT', ai_meta: Optional[Dict] = None):
        """Устанавливает источник решения для последующего логирования."""
        normalized = 'AI' if source == 'AI' else 'SCRIPT'
        self._last_decision_source = normalized
        if normalized == 'AI' and ai_meta:
            # Храним только необходимые поля
            self._last_ai_decision_meta = {
                'ai_confidence': ai_meta.get('ai_confidence'),
                'ai_signal': ai_meta.get('ai_signal') or ai_meta.get('signal')
            }
        else:
            self._last_ai_decision_meta = None
            self.ai_decision_id = None
    
    def _on_position_opened(self, direction: str, entry_price: Optional[float], position_size: Optional[float]):
        """Логирует открытие позиции в историю ботов."""
        try:
            from bot_engine.bot_history import log_position_opened
        except ImportError:
            return
        
        try:
            size = position_size or self._get_position_quantity() or 0.0
            price = entry_price or self.entry_price or 0.0
            decision_source = getattr(self, '_last_decision_source', 'SCRIPT')
            ai_meta = getattr(self, '_last_ai_decision_meta', None) or {}
            ctx = getattr(self, '_last_entry_context', {}) or {}
            
            # КРИТИЧНО ДЛЯ ОБУЧЕНИЯ AI: Получаем RSI и тренд из контекста или из глобальных данных
            rsi_value = ctx.get('rsi')
            trend_value = ctx.get('trend')
            
            # Если контекст пустой, пытаемся получить из глобальных данных RSI
            if rsi_value is None or trend_value is None:
                try:
                    with rsi_data_lock:
                        rsi_info = coins_rsi_data.get(self.symbol, {})
                        if rsi_value is None:
                            from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                            rsi_value = get_rsi_from_coin_data(rsi_info)
                        if trend_value is None:
                            trend_value = get_trend_from_coin_data(rsi_info)
                except Exception:
                    pass
            
            # Если все еще нет данных, пытаемся из rsi_data бота
            if rsi_value is None or trend_value is None:
                try:
                    with bots_data_lock:
                        bot_data = bots_data.get('bots', {}).get(self.symbol, {})
                        rsi_data = bot_data.get('rsi_data', {})
                        if rsi_value is None:
                            from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                            rsi_value = get_rsi_from_coin_data(rsi_data)
                        if trend_value is None:
                            trend_value = get_trend_from_coin_data(rsi_data)
                except Exception:
                    pass
            
            logger.info(f"[NEW_BOT_{self.symbol}] 📊 Логируем открытие позиции: RSI={rsi_value}, Trend={trend_value}, Price={price}")
            
            log_position_opened(
                bot_id=self.symbol,
                symbol=self.symbol,
                direction=direction,
                size=size,
                entry_price=price,
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
                decision_source=decision_source,
                ai_decision_id=self.ai_decision_id if decision_source == 'AI' else None,
                ai_confidence=ai_meta.get('ai_confidence'),
                ai_signal=ai_meta.get('ai_signal') or direction,
                rsi=rsi_value,
                trend=trend_value,
                is_simulated=False  # КРИТИЧНО: реальные боты - это НЕ симуляция!
            )
            # Помечаем, что логирование выполнено (для предотвращения дублирования)
            self._position_logged = True
        except Exception as log_error:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА логирования открытия позиции: {log_error}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Сбрасываем контекст, но ID оставляем до закрытия
            self._last_entry_context = {}
            if self._last_decision_source != 'AI':
                self._last_ai_decision_meta = None
            
    def _get_effective_protection_config(self) -> Dict:
        try:
            base_config = get_auto_bot_config().copy()
        except Exception:
            base_config = {}
        merged = dict(base_config)
        overrides = self.config or {}
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
        return merged

    def _build_protection_state(self) -> ProtectionState:
        entry_price = self._safe_float(self.entry_price, 0.0) or 0.0
        position_side = (self.position_side or '').upper() or 'LONG'

        entry_time = None
        if isinstance(self.position_start_time, datetime):
            entry_time = self.position_start_time.timestamp()
        elif self.config.get('entry_timestamp'):
            entry_time = self._safe_float(self.config.get('entry_timestamp'), None)
            if entry_time and entry_time > 1e12:
                entry_time = entry_time / 1000.0

        quantity = self._get_position_quantity() or None

        notional_usdt = None
        if quantity and entry_price:
            notional_usdt = quantity * entry_price
        elif isinstance(self.volume_value, (int, float)):
            notional_usdt = float(self.volume_value)

        return ProtectionState(
            position_side=position_side,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            notional_usdt=notional_usdt,
            max_profit_percent=self.max_profit_achieved or 0.0,
            break_even_activated=bool(self.break_even_activated),
            break_even_stop_price=self._safe_float(self.break_even_stop_price),
            break_even_stop_set=bool(self.break_even_stop_set),
            trailing_active=bool(self.trailing_active),
            trailing_reference_price=self._safe_float(self.trailing_reference_price),
            trailing_stop_price=self._safe_float(self.trailing_stop_price),
            trailing_take_profit_price=self._safe_float(self.trailing_take_profit_price),
            trailing_last_update_ts=self._safe_float(self.trailing_last_update_ts, 0.0) or 0.0,
        )

    def _apply_protection_state(self, state: ProtectionState) -> None:
        self.max_profit_achieved = state.max_profit_percent
        self.break_even_activated = state.break_even_activated
        self.break_even_stop_set = state.break_even_stop_set
        self.break_even_stop_price = state.break_even_stop_price
        self.trailing_active = state.trailing_active
        self.trailing_reference_price = state.trailing_reference_price
        self.trailing_stop_price = state.trailing_stop_price
        self.trailing_take_profit_price = state.trailing_take_profit_price
        self.trailing_last_update_ts = state.trailing_last_update_ts

    def _evaluate_protection_decision(self, current_price: float):
        try:
            config = self._get_effective_protection_config()
        except Exception:
            config = {}
        state = self._build_protection_state()
        realized = self._safe_float(self.realized_pnl, 0.0) or 0.0
        decision = evaluate_protections(
            current_price=current_price,
            config=config,
            state=state,
            realized_pnl=realized,
            now_ts=time.time(),
        )
        if decision.state:
            self._apply_protection_state(decision.state)
        return decision

    
    def should_open_long(self, rsi, trend, candles):
        """Проверяет, нужно ли открывать LONG позицию"""
        try:
            # ✅ ПРОВЕРКА ДЕЛИСТИНГА: Проверяем ДО всех остальных проверок
            from bots_modules.sync_and_cache import load_delisted_coins
            delisted_data = load_delisted_coins()
            delisted_coins = delisted_data.get('delisted_coins', {})
            
            if self.symbol in delisted_coins:
                delisting_info = delisted_coins[self.symbol]
                logger.warning(f"[NEW_BOT_{self.symbol}] 🚨 ДЕЛИСТИНГ! Не открываем LONG - {delisting_info.get('reason', 'Delisting detected')}")
                return False
            
            # Получаем настройки из конфига (ВАЖНО: сначала индивидуальные настройки бота, потом глобальные)
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                # Используем индивидуальные настройки из self.config если есть, иначе из auto_config
                rsi_long_threshold = self.config.get('rsi_long_threshold') or auto_config.get('rsi_long_threshold', 29)
                # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
                avoid_down_trend = self.config.get('avoid_down_trend') if 'avoid_down_trend' in self.config else auto_config.get('avoid_down_trend', False)
                rsi_time_filter_enabled = self.config.get('rsi_time_filter_enabled') if 'rsi_time_filter_enabled' in self.config else auto_config.get('rsi_time_filter_enabled', True)
                rsi_time_filter_candles = self.config.get('rsi_time_filter_candles') or auto_config.get('rsi_time_filter_candles', 8)
                rsi_time_filter_lower = self.config.get('rsi_time_filter_lower') or auto_config.get('rsi_time_filter_lower', 35)
                ai_enabled = auto_config.get('ai_enabled', False)  # Включение AI
                ai_override = auto_config.get('ai_override_original', True)
            
            # 🤖 ПРОВЕРКА AI ПРЕДСКАЗАНИЯ (если включено)
            self._set_decision_source('SCRIPT')
            if ai_enabled:
                try:
                    from bot_engine.ai.ai_integration import should_open_position_with_ai
                    
                    # Получаем текущую цену
                    current_price = 0
                    if candles and len(candles) > 0:
                        current_price = candles[-1].get('close', 0)
                    
                    if current_price > 0:
                        ai_result = should_open_position_with_ai(
                            symbol=self.symbol,
                            direction='LONG',
                            rsi=rsi,
                            trend=trend,
                            price=current_price,
                            config=auto_config
                        )
                        
                        if ai_result.get('ai_used'):
                            if ai_result.get('should_open'):
                                logger.info(f"[NEW_BOT_{self.symbol}] 🤖 AI подтверждает LONG (уверенность: {ai_result.get('ai_confidence', 0):.2%})")
                                # Сохраняем ID решения AI для последующего отслеживания результатов
                                self.ai_decision_id = ai_result.get('ai_decision_id')
                                self._set_decision_source('AI', ai_result)
                            else:
                                logger.info(f"[NEW_BOT_{self.symbol}] 🤖 AI блокирует LONG: {ai_result.get('reason', 'AI prediction')}")
                                if ai_override:
                                    return False
                                logger.info(f"[NEW_BOT_{self.symbol}] ⚖️ Используем скриптовые правила (AI только рекомендателен)")
                                self._set_decision_source('SCRIPT')
                except ImportError:
                    # AI модуль недоступен - продолжаем без него
                    pass
                except Exception:
                    pass
            
            # 1. Проверка RSI
            if rsi > rsi_long_threshold:
                return False
            
            # 2. Проверка тренда
            if avoid_down_trend and trend == 'DOWN':
                return False
            
            # 3. RSI временной фильтр
            if rsi_time_filter_enabled:
                time_filter_result = self.check_rsi_time_filter_for_long(candles, rsi, rsi_time_filter_candles, rsi_time_filter_lower)
                if not time_filter_result['allowed']:
                    return False
            
            # ✅ КРИТИЧНО: Проверка наличия открытой позиции - если позиция уже открыта, фильтр НЕ применяется
            # Фильтр защиты от повторных входов работает ТОЛЬКО при попытке открыть НОВУЮ позицию
            is_in_position = (self.status == BOT_STATUS['IN_POSITION_LONG'] or 
                             self.status == BOT_STATUS['IN_POSITION_SHORT'] or 
                             self.position_side is not None)
            
            # 4. Проверка защиты от повторных входов после убыточных закрытий
            # ⚠️ ПРИМЕНЯЕМ ТОЛЬКО если позиция НЕ открыта
            if not is_in_position:
                loss_reentry_result = self.check_loss_reentry_protection(candles)
                # ✅ КРИТИЧНО: Строгая проверка - если allowed == False, БЛОКИРУЕМ вход
                if loss_reentry_result.get('allowed') is False:
                    logger.error(f"[NEW_BOT_{self.symbol}] 🚫🚫🚫 ЗАЩИТА ОТ ПОВТОРНЫХ ВХОДОВ БЛОКИРУЕТ LONG! 🚫🚫🚫 Причина: {loss_reentry_result.get('reason', 'Unknown')}")
                    return False
            else:
                pass
            
            logger.info(f"[NEW_BOT_{self.symbol}] ✅ Открываем LONG (RSI: {rsi:.1f})")
            self._remember_entry_context(rsi, trend)
            return True
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки LONG: {e}")
            return False
    
    def should_open_short(self, rsi, trend, candles):
        """Проверяет, нужно ли открывать SHORT позицию"""
        try:
            # ✅ ПРОВЕРКА ДЕЛИСТИНГА: Проверяем ДО всех остальных проверок
            from bots_modules.sync_and_cache import load_delisted_coins
            delisted_data = load_delisted_coins()
            delisted_coins = delisted_data.get('delisted_coins', {})
            
            if self.symbol in delisted_coins:
                delisting_info = delisted_coins[self.symbol]
                logger.warning(f"[NEW_BOT_{self.symbol}] 🚨 ДЕЛИСТИНГ! Не открываем SHORT - {delisting_info.get('reason', 'Delisting detected')}")
                return False
            
            # Получаем настройки из конфига (ВАЖНО: сначала индивидуальные настройки бота, потом глобальные)
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                # Используем индивидуальные настройки из self.config если есть, иначе из auto_config
                rsi_short_threshold = self.config.get('rsi_short_threshold') or auto_config.get('rsi_short_threshold', 71)
                # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
                avoid_up_trend = self.config.get('avoid_up_trend') if 'avoid_up_trend' in self.config else auto_config.get('avoid_up_trend', False)
                rsi_time_filter_enabled = self.config.get('rsi_time_filter_enabled') if 'rsi_time_filter_enabled' in self.config else auto_config.get('rsi_time_filter_enabled', True)
                rsi_time_filter_candles = self.config.get('rsi_time_filter_candles') or auto_config.get('rsi_time_filter_candles', 8)
                rsi_time_filter_upper = auto_config.get('rsi_time_filter_upper', 65)
                ai_enabled = auto_config.get('ai_enabled', False)  # Включение AI
                ai_override = auto_config.get('ai_override_original', True)
            
            # 🤖 ПРОВЕРКА AI ПРЕДСКАЗАНИЯ (если включено)
            self._set_decision_source('SCRIPT')
            if ai_enabled:
                try:
                    from bot_engine.ai.ai_integration import should_open_position_with_ai
                    
                    # Получаем текущую цену
                    current_price = 0
                    if candles and len(candles) > 0:
                        current_price = candles[-1].get('close', 0)
                    
                    if current_price > 0:
                        ai_result = should_open_position_with_ai(
                            symbol=self.symbol,
                            direction='SHORT',
                            rsi=rsi,
                            trend=trend,
                            price=current_price,
                            config=auto_config
                        )
                        
                        if ai_result.get('ai_used'):
                            if ai_result.get('should_open'):
                                logger.info(f"[NEW_BOT_{self.symbol}] 🤖 AI подтверждает SHORT (уверенность: {ai_result.get('ai_confidence', 0):.2%})")
                                # Сохраняем ID решения AI для последующего отслеживания результатов
                                self.ai_decision_id = ai_result.get('ai_decision_id')
                                self._set_decision_source('AI', ai_result)
                            else:
                                logger.info(f"[NEW_BOT_{self.symbol}] 🤖 AI блокирует SHORT: {ai_result.get('reason', 'AI prediction')}")
                                if ai_override:
                                    return False
                                logger.info(f"[NEW_BOT_{self.symbol}] ⚖️ AI советует WAIT, но используем базовую стратегию")
                                self._set_decision_source('SCRIPT')
                except ImportError:
                    # AI модуль недоступен - продолжаем без него
                    pass
                except Exception:
                    pass
            
            # 1. Проверка RSI
            if rsi < rsi_short_threshold:
                return False
            
            # 2. Проверка тренда
            if avoid_up_trend and trend == 'UP':
                return False
            
            # 3. RSI временной фильтр
            if rsi_time_filter_enabled:
                time_filter_result = self.check_rsi_time_filter_for_short(candles, rsi, rsi_time_filter_candles, rsi_time_filter_upper)
                if not time_filter_result['allowed']:
                    return False
            
            # ✅ КРИТИЧНО: Проверка наличия открытой позиции - если позиция уже открыта, фильтр НЕ применяется
            # Фильтр защиты от повторных входов работает ТОЛЬКО при попытке открыть НОВУЮ позицию
            is_in_position = (self.status == BOT_STATUS['IN_POSITION_LONG'] or 
                             self.status == BOT_STATUS['IN_POSITION_SHORT'] or 
                             self.position_side is not None)
            
            # 4. Проверка защиты от повторных входов после убыточных закрытий
            # ⚠️ ПРИМЕНЯЕМ ТОЛЬКО если позиция НЕ открыта
            if not is_in_position:
                loss_reentry_result = self.check_loss_reentry_protection(candles)
                # ✅ КРИТИЧНО: Строгая проверка - если allowed == False, БЛОКИРУЕМ вход
                if loss_reentry_result.get('allowed') is False:
                    logger.error(f"[NEW_BOT_{self.symbol}] 🚫🚫🚫 ЗАЩИТА ОТ ПОВТОРНЫХ ВХОДОВ БЛОКИРУЕТ SHORT! 🚫🚫🚫 Причина: {loss_reentry_result.get('reason', 'Unknown')}")
                    return False
            else:
                pass
            
            logger.info(f"[NEW_BOT_{self.symbol}] ✅ Открываем SHORT (RSI: {rsi:.1f})")
            self._remember_entry_context(rsi, trend)
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
    
    def check_loss_reentry_protection(self, candles):
        """
        ДОПОЛНИТЕЛЬНЫЙ ФИЛЬТР: Защита от повторных входов после убыточных закрытий
        
        Логика фильтра:
        1. Берет последние N закрытых сделок по текущей монете (self.symbol)
        2. Проверяет, все ли они были с отрицательным результатом (pnl < 0)
        3. Если все последние N сделок в минус:
           - Проверяет время закрытия последней убыточной сделки
           - Считает количество свечей, прошедших с момента закрытия
           - Если прошло МЕНЬШЕ X свечей - БЛОКИРУЕТ вход
           - Если прошло X свечей или больше - РАЗРЕШАЕТ вход
        4. Если не все последние N сделок в минус (есть хотя бы одна прибыльная) - РАЗРЕШАЕТ вход
        
        Returns:
            dict: {'allowed': bool, 'reason': str}
        """
        try:
            # Убеждаемся, что bots_data_lock доступен
            try:
                from bots_modules.imports_and_globals import bots_data, bots_data_lock
            except ImportError:
                # Если импорт не удался, используем глобальные переменные из начала файла
                pass
            
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
            
            # ✅ КРИТИЧНО: ВСЕГДА проверяем, прошла ли минимум 1 свеча (6ч) с последнего закрытия позиции
            # Это нужно, чтобы история успела подгрузиться в БД после закрытия
            # Работает НЕЗАВИСИМО от настройки loss_reentry_protection (как просил пользователь)
            last_close_timestamp = self.config.get('last_position_close_timestamp')
            
            # Также проверяем глобальный словарь (для случаев когда бот был удален после закрытия)
            if not last_close_timestamp:
                try:
                    # Повторный импорт не нужен, переменные уже должны быть доступны
                    with bots_data_lock:
                        last_close_timestamps = bots_data.get('last_close_timestamps', {})
                        last_close_timestamp = last_close_timestamps.get(self.symbol)
                except Exception as e:
                    logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка чтения глобального словаря: {e}")
            
            if last_close_timestamp:
                try:
                    from datetime import datetime
                    current_timestamp = datetime.now().timestamp()
                    time_since_close = current_timestamp - float(last_close_timestamp)
                    min_wait_seconds = 6 * 3600  # 1 свеча 6h минимум
                    
                    if time_since_close < min_wait_seconds:
                        wait_remaining = min_wait_seconds - time_since_close
                        wait_remaining_hours = wait_remaining / 3600
                        logger.warning(
                            f"[NEW_BOT_{self.symbol}] 🚫🚫🚫 ЗАБЛОКИРОВАНО (1 свеча 6ч задержка): После закрытия позиции прошло только {time_since_close:.0f} секунд "
                            f"(требуется {min_wait_seconds} сек = 1 свеча 6ч). Осталось ждать: {wait_remaining_hours:.1f} ч"
                        )
                        return {
                            'allowed': False,
                            'reason': f'Minimum 1 candle (6h) wait after position close (only {time_since_close:.0f}s passed, need {min_wait_seconds}s)'
                        }
                    else:
                        logger.info(f"[NEW_BOT_{self.symbol}] ✅ Прошло {time_since_close/3600:.2f} ч с последнего закрытия - продолжаем проверку фильтра")
                except Exception as timestamp_check_error:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка проверки timestamp закрытия: {timestamp_check_error}")
            
            # Получаем настройки (сначала из индивидуальных, потом из глобальных)
            loss_reentry_protection_enabled = self.config.get('loss_reentry_protection') if 'loss_reentry_protection' in self.config else auto_config.get('loss_reentry_protection', True)
            loss_reentry_count = self.config.get('loss_reentry_count') or auto_config.get('loss_reentry_count', 1)
            loss_reentry_candles = self.config.get('loss_reentry_candles') or auto_config.get('loss_reentry_candles', 3)
            
            # Если защита выключена - разрешаем вход (но только если прошла 1 свеча 6ч!)
            if not loss_reentry_protection_enabled:
                return {'allowed': True, 'reason': 'Protection disabled'}
            
            # Получаем последние N закрытых сделок для этого символа
            try:
                from bot_engine.bots_database import get_bots_database
                bots_db = get_bots_database()
                
                # ✅ КРИТИЧНО: Получаем последние N закрытых сделок по текущей монете
                # Сначала пробуем из bot_trades_history (сделки ботов)
                closed_trades = bots_db.get_bot_trades_history(
                    bot_id=None,
                    symbol=self.symbol,  # ⬅️ Только для текущей монеты
                    status='CLOSED',
                    decision_source=None,
                    limit=loss_reentry_count,  # ⬅️ Последние N сделок
                    offset=0
                )
                
                if closed_trades:
                    for i, trade in enumerate(closed_trades):
                        pnl = trade.get('pnl')
                        exit_timestamp = trade.get('exit_timestamp')
                
                # ✅ КРИТИЧНО: Если нет сделок в bot_trades_history или недостаточно, дополняем из closed_pnl_history (UI сделки)
                if not closed_trades or len(closed_trades) < loss_reentry_count:
                    try:
                        from app.app_database import get_app_database
                        app_db = get_app_database()
                        if app_db:
                            # Читаем из closed_pnl_history
                            all_closed_pnl = app_db.load_closed_pnl_history(sort_by='time', period='all')
                            symbol_closed_pnl = [t for t in all_closed_pnl if t.get('symbol') == self.symbol]
                            # Сортируем по close_timestamp DESC (новые первыми)
                            symbol_closed_pnl.sort(key=lambda x: x.get('close_timestamp', 0), reverse=True)
                            
                            # Если closed_trades пустой - создаем список
                            if not closed_trades:
                                closed_trades = []
                            
                            # Дополняем до нужного количества
                            needed_count = loss_reentry_count - len(closed_trades)
                            for pnl_trade in symbol_closed_pnl[:needed_count]:
                                trade = {
                                    'pnl': pnl_trade.get('closed_pnl'),  # ⬅️ В closed_pnl_history поле называется closed_pnl
                                    'exit_time': pnl_trade.get('close_time'),
                                    'exit_timestamp': pnl_trade.get('close_timestamp'),
                                    'close_reason': 'MANUAL_CLOSE',
                                    'is_simulated': False
                                }
                                closed_trades.append(trade)
                            
                            # Сортируем объединенный список по exit_timestamp DESC
                            closed_trades.sort(key=lambda x: x.get('exit_timestamp') or 0, reverse=True)
                            # Берем только последние N
                            closed_trades = closed_trades[:loss_reentry_count]
                    except Exception as app_db_error:
                        logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка загрузки из closed_pnl_history: {app_db_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # ✅ КРИТИЧНО: Если нет закрытых сделок или недостаточно - РАЗРЕШАЕМ вход
                # (фильтр не применяется, если недостаточно истории)
                if not closed_trades or len(closed_trades) < loss_reentry_count:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Защита от повторных входов: недостаточно закрытых сделок ({len(closed_trades) if closed_trades else 0} < {loss_reentry_count}) - РАЗРЕШАЕМ вход")
                    return {'allowed': True, 'reason': f'Not enough closed trades ({len(closed_trades) if closed_trades else 0} < {loss_reentry_count})'}
                
                # ✅ ИСПРАВЛЕНО: Проверяем, все ли последние N сделок были с отрицательным результатом (pnl < 0)
                # Важно: проверяем именно ПОСЛЕДНИЕ N сделок по времени закрытия (уже отсортированы DESC)
                all_losses = True
                pnl_details = []
                for idx, trade in enumerate(closed_trades):
                    pnl = trade.get('pnl', 0)
                    exit_time = trade.get('exit_time') or trade.get('exit_timestamp')
                    exit_timestamp = trade.get('exit_timestamp')
                    close_reason = trade.get('close_reason')
                    is_simulated = trade.get('is_simulated', False)
                    
                    # ✅ КРИТИЧНО: Проверяем что PnL определен и действительно отрицательный (строго < 0)
                    try:
                        pnl_float = float(pnl) if pnl is not None else 0.0
                        pnl_details.append(f"#{idx+1}: PnL={pnl_float:.4f} USDT, exit_time={exit_time}, simulated={is_simulated}")
                        
                        # Если хотя бы одна сделка >= 0 (прибыльная или безубыточная) - не все в минус
                        if pnl_float >= 0:
                            all_losses = False
                            break
                    except (ValueError, TypeError) as e:
                        # Если не удалось преобразовать PnL - считаем что не убыточная
                        logger.error(f"[NEW_BOT_{self.symbol}] ❌ ОШИБКА преобразования PnL для сделки #{idx+1}: pnl={pnl}, тип={type(pnl)}, ошибка={e} - РАЗРЕШАЕМ вход")
                        all_losses = False
                        break
                
                # ✅ КРИТИЧНО: Если НЕ ВСЕ последние N сделок в минус - РАЗРЕШАЕМ вход (фильтр НЕ работает)
                if not all_losses:
                    logger.info(f"[NEW_BOT_{self.symbol}] ✅ Защита от повторных входов: НЕ все последние {loss_reentry_count} сделок в минус. PnL: {', '.join(pnl_details)}")
                    return {'allowed': True, 'reason': f'Not all last {loss_reentry_count} trades were losses'}
                
                # ✅ Все сделки в минус - логируем детали
                logger.warning(f"[NEW_BOT_{self.symbol}] 🚫 ВСЕ {loss_reentry_count} последних сделок в МИНУС! PnL: {', '.join(pnl_details)}")
                
                # ✅ Все последние N сделок были в минус - проверяем количество прошедших свечей
                # Берем самую последнюю закрытую убыточную сделку (первая в списке после сортировки по exit_timestamp DESC)
                last_trade = closed_trades[0]
                
                # Получаем timestamp закрытия последней сделки
                exit_timestamp = last_trade.get('exit_timestamp')
                if not exit_timestamp:
                    # Если нет exit_timestamp, пытаемся получить из exit_time
                    exit_time_str = last_trade.get('exit_time')
                    if exit_time_str:
                        try:
                            from datetime import datetime
                            if isinstance(exit_time_str, str):
                                exit_dt = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                                exit_timestamp = int(exit_dt.timestamp())
                            else:
                                exit_timestamp = int(exit_time_str)
                        except:
                            logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось получить timestamp закрытия сделки")
                            # Если не удалось получить timestamp - разрешаем вход (безопаснее)
                            return {'allowed': True, 'reason': 'Cannot get close timestamp'}
                    else:
                        # Нет данных о времени закрытия - разрешаем вход
                        return {'allowed': True, 'reason': 'No close timestamp'}
                
                # Если exit_timestamp в миллисекундах, конвертируем в секунды
                if exit_timestamp > 1e12:
                    exit_timestamp = exit_timestamp / 1000
                
                # Получаем текущее время
                current_time = time.time()
                
                # Подсчитываем количество свечей, прошедших с момента закрытия
                # Свечи 6h, значит одна свеча = 6 часов = 21600 секунд
                # Получаем текущий таймфрейм динамически
                try:
                    from bot_engine.bot_config import get_current_timeframe
                    current_timeframe = get_current_timeframe()
                    # Конвертируем таймфрейм в секунды
                    timeframe_to_seconds = {
                        '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                        '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800,
                        '12h': 43200, '1d': 86400, '3d': 259200, '1w': 604800, '1M': 2592000
                    }
                    CANDLE_INTERVAL_SECONDS = timeframe_to_seconds.get(current_timeframe, 21600)  # По умолчанию 6h
                except:
                    CANDLE_INTERVAL_SECONDS = 6 * 3600  # Fallback: 6 часов
                
                # Находим последнюю свечу (самую новую) в переданных candles
                if not candles or len(candles) == 0:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Нет свечей для проверки защиты от повторных входов")
                    return {'allowed': True, 'reason': 'No candles provided'}
                
                # ✅ ИСПРАВЛЕНО: Получаем timestamp последней свечи (проверяем оба варианта: 'timestamp' и 'time')
                last_candle = candles[-1]  # Последняя свеча - самая новая
                last_candle_timestamp = last_candle.get('timestamp') or last_candle.get('time', 0)
                
                # Если timestamp в миллисекундах, конвертируем в секунды
                if last_candle_timestamp > 1e12:
                    last_candle_timestamp = last_candle_timestamp / 1000
                
                # ✅ ИСПРАВЛЕНО: Подсчитываем количество свечей с момента закрытия
                # Свечи уже отсортированы по времени (старые -> новые)
                candles_passed = 0
                
                # ✅ ИСПРАВЛЕНО: Ищем первую свечу, которая ПОЛНОСТЬЮ позже времени закрытия
                # Проверяем оба варианта ключа: 'timestamp' и 'time'
                for i, candle in enumerate(candles):
                    candle_timestamp = candle.get('timestamp') or candle.get('time', 0)
                    if candle_timestamp > 1e12:
                        candle_timestamp = candle_timestamp / 1000
                    
                    # Если начало свечи >= времени закрытия, считаем эту и все последующие свечи
                    if candle_timestamp >= exit_timestamp:
                        candles_passed = len(candles) - i
                        break
                
                # ✅ ИСПРАВЛЕНО: Если не нашли через перебор, считаем по времени (более надежно)
                if candles_passed == 0:
                    time_diff_seconds = last_candle_timestamp - exit_timestamp
                    if time_diff_seconds > 0:
                        # Считаем количество полных 6-часовых интервалов (минимум 1)
                        candles_passed = max(1, int(time_diff_seconds / CANDLE_INTERVAL_SECONDS))
                
                # ✅ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА
                if candles_passed == 0 and last_candle_timestamp > exit_timestamp:
                    candles_passed = 1
                
                # ✅ ИСПРАВЛЕНО: Конвертируем loss_reentry_candles в int для корректного сравнения
                try:
                    loss_reentry_candles_int = int(loss_reentry_candles) if loss_reentry_candles is not None else 3
                except (ValueError, TypeError):
                    loss_reentry_candles_int = 3
                
                # ✅ КРИТИЧЕСКАЯ ПРОВЕРКА: Если прошло МЕНЬШЕ X свечей - БЛОКИРУЕМ вход
                # Если прошло X свечей или больше - РАЗРЕШАЕМ вход
                if candles_passed < loss_reentry_candles_int:
                    logger.error(
                        f"[NEW_BOT_{self.symbol}] 🚫🚫🚫 ФИЛЬТР ЗАБЛОКИРОВАЛ ВХОД! 🚫🚫🚫\n"
                        f"  - Последние {loss_reentry_count} сделок в минус\n"
                        f"  - Прошло только {candles_passed} свечей (требуется {loss_reentry_candles_int})\n"
                        f"  - ВХОД ЗАБЛОКИРОВАН!"
                    )
                    return {
                        'allowed': False,  # ⬅️ БЛОКИРУЕМ вход
                        'reason': f'Last {loss_reentry_count} trades were losses, only {candles_passed} candles passed (need {loss_reentry_candles_int})'
                    }
                else:
                    logger.info(
                        f"[NEW_BOT_{self.symbol}] ✅ Прошло достаточно свечей ({candles_passed} >= {loss_reentry_candles_int}) - фильтр РАЗРЕШАЕТ вход"
                    )
                
                # ✅ Прошло X свечей или больше - РАЗРЕШАЕМ вход
                return {
                    'allowed': True,  # ⬅️ РАЗРЕШАЕМ вход
                    'reason': f'{candles_passed} candles passed since last loss (required: {loss_reentry_candles_int})'
                }
                
            except Exception as db_error:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки защиты от повторных входов: {db_error}")
                # При ошибке разрешаем вход (безопаснее)
                return {'allowed': True, 'reason': f'Error checking protection: {str(db_error)}'}
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки защиты от повторных входов: {e}")
            # При ошибке разрешаем вход (безопаснее)
            return {'allowed': True, 'reason': f'Exception: {str(e)}'}
    
    @staticmethod
    def check_should_close_by_rsi(symbol, rsi, position_side):
        """
        Статическая функция проверки закрытия позиции по RSI (без создания объекта бота)
        
        Args:
            symbol: Символ монеты
            rsi: Текущее значение RSI
            position_side: Сторона позиции ('LONG' или 'SHORT')
        
        Returns:
            tuple: (should_close: bool, reason: str или None)
        """
        try:
            if position_side not in ['LONG', 'SHORT']:
                logger.error(f"[RSI_CHECK_{symbol}] ❌ Неизвестная сторона позиции: {position_side}")
                return False, None
            
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {})
                bot_data = bots_data.get('bots', {}).get(symbol, {})
                entry_trend = bot_data.get('entry_trend', None)
                try:
                    from bots_modules.filters import get_individual_coin_settings
                    individual_settings = get_individual_coin_settings(symbol)
                except Exception:
                    individual_settings = {}
                
                def _thresh(coin_key, config_key, default):
                    return (individual_settings.get(coin_key) if individual_settings else None) or bot_data.get(config_key) or auto_config.get(config_key, default)
                
                if position_side == 'LONG':
                    if entry_trend == 'UP':
                        config_key = 'rsi_exit_long_with_trend'
                        threshold = _thresh('rsi_exit_long_with_trend', config_key, 65)
                    else:
                        config_key = 'rsi_exit_long_against_trend'
                        threshold = _thresh('rsi_exit_long_against_trend', config_key, 60)
                    
                    condition_func = lambda r, t: r >= t  # RSI >= порог для LONG
                    condition_str = ">="
                    
                else:  # SHORT
                    if entry_trend == 'DOWN':
                        config_key = 'rsi_exit_short_with_trend'
                        threshold = _thresh('rsi_exit_short_with_trend', config_key, 35)
                    else:
                        config_key = 'rsi_exit_short_against_trend'
                        threshold = _thresh('rsi_exit_short_against_trend', config_key, 40)
                    
                    condition_func = lambda r, t: r <= t  # RSI <= порог для SHORT
                    condition_str = "<="
            
            # КРИТИЧНО: Если значение не найдено - это ОШИБКА КОНФИГУРАЦИИ!
            if threshold is None:
                logger.error(f"[RSI_CHECK_{symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА: {config_key} не найден в конфигурации! Позиция НЕ будет закрыта!")
                logger.error(f"[RSI_CHECK_{symbol}] ❌ Проверьте конфигурацию auto_bot_config в bots_data!")
                return False, None
            
            condition_result = condition_func(rsi, threshold)
            
            if condition_result:
                return True, 'RSI_EXIT'
            
            return False, None
            
        except Exception as e:
            logger.error(f"[RSI_CHECK_{symbol}] ❌ Ошибка проверки закрытия {position_side}: {e}")
            return False, None
    
    def should_close_position(self, rsi, current_price, position_side=None):
        """
        Универсальная функция проверки закрытия позиции по RSI
        
        Args:
            rsi: Текущее значение RSI
            current_price: Текущая цена (не используется, но оставлен для совместимости)
            position_side: Сторона позиции ('LONG' или 'SHORT'). Если None, берется из self.position_side
        
        Returns:
            tuple: (should_close: bool, reason: str или None)
        """
        # Используем статический метод для проверки
        if position_side is None:
            position_side = self.position_side
        return self.check_should_close_by_rsi(self.symbol, rsi, position_side)
    
    # Обратная совместимость - оставляем старые методы для совместимости
    def should_close_long(self, rsi, current_price):
        """Проверяет, нужно ли закрывать LONG позицию (обертка для совместимости)"""
        return self.should_close_position(rsi, current_price, 'LONG')
    
    def should_close_short(self, rsi, current_price):
        """Проверяет, нужно ли закрывать SHORT позицию (обертка для совместимости)"""
        return self.should_close_position(rsi, current_price, 'SHORT')
    
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
            
            # ✅ КРИТИЧНО: Для закрытия по RSI используем таймфрейм ВХОДА бота (entry_timeframe). 1m-бот — по 1m RSI.
            if self.entry_timeframe and self.status in [
                BOT_STATUS.get('IN_POSITION_LONG'),
                BOT_STATUS.get('IN_POSITION_SHORT')
            ]:
                timeframe_to_use = self.entry_timeframe
            else:
                from bot_engine.bot_config import get_current_timeframe
                timeframe_to_use = get_current_timeframe()
            
            # Получаем RSI данные с учетом таймфрейма бота
            try:
                # Проверяем, определен ли rsi_data_lock
                if 'rsi_data_lock' in globals():
                    with rsi_data_lock:
                        coin_data = coins_rsi_data['coins'].get(self.symbol)
                        if coin_data:
                            from bot_engine.bot_config import get_rsi_from_coin_data
                            # ✅ Используем таймфрейм бота для получения RSI
                            current_rsi = get_rsi_from_coin_data(coin_data, timeframe=timeframe_to_use)
                            current_price = coin_data.get('price')
                            if not current_trend:
                                from bot_engine.bot_config import get_trend_from_coin_data
                                # ✅ Используем таймфрейм бота для получения тренда
                                current_trend = get_trend_from_coin_data(coin_data, timeframe=timeframe_to_use)
                else:
                    # Fallback если lock не определен
                    coin_data = coins_rsi_data['coins'].get(self.symbol)
                    if coin_data:
                        from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                        # ✅ Используем таймфрейм бота для получения RSI
                        current_rsi = get_rsi_from_coin_data(coin_data, timeframe=timeframe_to_use)
                        current_price = coin_data.get('price')
                        if not current_trend:
                            # ✅ Используем таймфрейм бота для получения тренда
                            current_trend = get_trend_from_coin_data(coin_data, timeframe=timeframe_to_use)
            except Exception as e:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка получения RSI данных: {e}")
                # Fallback если lock не определен
                coin_data = coins_rsi_data['coins'].get(self.symbol)
                if coin_data:
                    from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                    # ✅ Используем таймфрейм бота для получения RSI
                    current_rsi = get_rsi_from_coin_data(coin_data, timeframe=timeframe_to_use)
                    current_price = coin_data.get('price')
                    if not current_trend:
                        # ✅ Используем таймфрейм бота для получения тренда
                        current_trend = get_trend_from_coin_data(coin_data, timeframe=timeframe_to_use)
            
            if current_rsi is None or current_price is None:
                logger.warning(f"[NEW_BOT_{self.symbol}] ❌ Нет RSI данных")
                return {'success': False, 'error': 'No RSI data'}
            
            # ✅ Получаем свечи для анализа с учетом таймфрейма бота
            chart_response = self.exchange.get_chart_data(self.symbol, timeframe_to_use, '30d')
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
                return {'success': True, 'status': self.status}
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления: {e}")
            return {'success': False, 'error': str(e)}

    def _get_market_price(self, fallback_price: float = None) -> float:
        """Возвращает актуальную цену из биржи (last/mark), если доступна"""
        if not self.exchange:
            return fallback_price
        try:
            ticker = self.exchange.get_ticker(self.symbol)
            if not ticker:
                return fallback_price

            candidates = (
                ticker.get('last'),
                ticker.get('markPrice'),
                ticker.get('price'),
                ticker.get('lastPrice'),
                ticker.get('mark'),
            )
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    value = float(candidate)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
        except Exception:
            pass
        return fallback_price

    def _open_position_on_exchange(self, direction: str, price: Optional[float] = None) -> bool:
        """Открывает позицию через TradingBot и логирует результат. Автовход — всегда по рынку."""
        try:
            result = self.enter_position(direction, force_market_entry=True)
            return bool(result and result.get('success'))
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка открытия позиции {direction}: {e}")
            return False

    def _handle_idle_state(self, rsi, trend, candles, price):
        """Бот в списке = проверки пройдены → по рынку заходим по условиям КОНФИГА (rsi_long_threshold, rsi_short_threshold)."""
        try:
            with bots_data_lock:
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            if not auto_bot_enabled:
                return {'success': True, 'status': self.status}
            # Направление и момент входа — только по настройкам конфига (should_open_long / should_open_short)
            if self.should_open_long(rsi, trend, candles):
                logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Вход по рынку LONG (условия конфига)")
                if self._open_position_on_exchange('LONG', price):
                    self.update_status(BOT_STATUS['IN_POSITION_LONG'], price, 'LONG')
                    return {'success': True, 'action': 'OPEN_LONG', 'status': self.status}
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть LONG позицию")
                return {'success': False, 'error': 'Failed to open LONG position'}
            if self.should_open_short(rsi, trend, candles):
                logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Вход по рынку SHORT (условия конфига)")
                if self._open_position_on_exchange('SHORT', price):
                    self.update_status(BOT_STATUS['IN_POSITION_SHORT'], price, 'SHORT')
                    return {'success': True, 'action': 'OPEN_SHORT', 'status': self.status}
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть SHORT позицию")
                return {'success': False, 'error': 'Failed to open SHORT position'}
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
            
            # Обновляем цену из биржи, чтобы trailing работал по реальному значению
            market_price = self._get_market_price(price)
            if market_price and market_price > 0:
                if price and abs(market_price - price) / max(price, 1e-9) >= 0.01:
                    pass  # обновили цену по бирже
                price = market_price

            self.current_price = price

            # 1. Проверяем защитные механизмы
            protection_result = self.check_protection_mechanisms(price)
            if protection_result['should_close']:
                logger.info(f"[NEW_BOT_{self.symbol}] 🛡️ Закрываем: {protection_result['reason']}")
                self._close_position_on_exchange(protection_result['reason'])
                return {'success': True, 'action': f"CLOSE_{self.position_side}", 'reason': protection_result['reason']}
            
            # 2. Проверяем условия закрытия по RSI (универсальная функция)
            if self.position_side in ['LONG', 'SHORT']:
                should_close, reason = self.should_close_position(rsi, price, self.position_side)
                if should_close:
                    logger.info(f"[NEW_BOT_{self.symbol}] 🔴 Закрываем {self.position_side} по RSI")
                    close_success = self._close_position_on_exchange(reason)
                    if close_success:
                        logger.info(f"[NEW_BOT_{self.symbol}] ✅ {self.position_side} закрыта")
                        return {'success': True, 'action': f'CLOSE_{self.position_side}', 'reason': reason}
                    else:
                        logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось закрыть {self.position_side} позицию на бирже!")
                        return {'success': False, 'error': 'Failed to close position on exchange', 'action': f'CLOSE_{self.position_side}_FAILED', 'reason': reason}
                else:
                    pass
            
            return {'success': True, 'status': self.status, 'position_side': self.position_side}
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка в позиции: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_trailing_by_margin(self, _profit_percent: float, current_price: float):
        """(DEPRECATED) Совместимость с устаревшими вызовами."""
        return {
            'active': False,
            'stop_price': None,
            'activation_profit_usdt': 0.0,
            'activation_threshold_usdt': 0.0,
            'locked_profit_usdt': 0.0,
            'margin_usdt': 0.0,
            'profit_usdt': 0.0,
            'profit_usdt_max': 0.0,
            'trailing_step_usdt': 0.0,
            'trailing_step_price': 0.0,
            'steps': 0
        }

    def _get_position_quantity(self) -> float:
        """Возвращает количество монет в позиции"""
        quantity = self.position_size_coins
        try:
            if quantity is not None:
                quantity = float(quantity)
        except (TypeError, ValueError):
            quantity = None

        if not quantity and self.position_size and self.entry_price:
            try:
                quantity = abs(float(self.position_size) / float(self.entry_price))
            except (TypeError, ValueError, ZeroDivisionError):
                quantity = None

        if not quantity and self.volume_value and self.entry_price:
            try:
                quantity = abs(float(self.volume_value) / float(self.entry_price))
            except (TypeError, ValueError, ZeroDivisionError):
                quantity = None

        try:
            return abs(float(quantity)) if quantity is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _calculate_break_even_stop_price(self, current_price: Optional[float] = None) -> Optional[float]:
        """Рассчитывает цену стоп-лосса для безубыточности на основе realized_pnl * 2.5"""
        if not self.entry_price or self.position_side not in ('LONG', 'SHORT'):
            return None

        quantity = self._get_position_quantity()
        if quantity <= 0:
            return None

        try:
            entry_price = float(self.entry_price)
        except (TypeError, ValueError):
            return None

        # ✅ Рассчитываем стоп от abs(realized_pnl) * 2.5 (в USDT)
        # Примечание: realized_pnl обычно отрицательный (комиссии при открытии позиции)
        # Берем по модулю (abs) и умножаем на 2.5, где:
        # - *2 = комиссия за открытие + комиссия за закрытие (по половине на каждую операцию)
        # - +0.5 = запас на проскальзывание при закрытии сделки
        try:
            realized_pnl_usdt = float(self.realized_pnl or 0.0)
        except (TypeError, ValueError):
            realized_pnl_usdt = 0.0
        
        # ✅ ИСПРАВЛЕНО: Берем по модулю БЕЗ УЧЕТА ЗНАКА (+ или -)
        # realized_pnl может быть как отрицательным (комиссии), так и положительным (прибыль)
        # ВАЖНО: Всегда берем по модулю, чтобы защитить от комиссий независимо от знака
        fee_usdt = abs(realized_pnl_usdt)
        
        price = float(current_price) if current_price is not None else None
        
        # ✅ СТРОГАЯ ЛОГИКА: Защищаем от комиссий в размере abs(realized_pnl) * 2.5
        # Если realized_pnl = 0, то fee_usdt = 0, и protected_profit_per_coin = 0, стоп будет на entry_price
        # ВАЖНО: Умножаем на 2.5 (BREAK_EVEN_FEE_MULTIPLIER) - это обязательно!
        protected_profit_usdt = fee_usdt * self.BREAK_EVEN_FEE_MULTIPLIER
        
        # Преобразуем защищаемую прибыль (USDT) в цену на монету
        protected_profit_per_coin = protected_profit_usdt / quantity if quantity > 0 else 0.0

        if self.position_side == 'LONG':
            # ✅ Для LONG: стоп на уровне entry_price + protected_profit_per_coin
            # Если realized_pnl = 0, то protected_profit_per_coin = 0, стоп = entry_price (базовая защита)
            # Если есть realized_pnl, стоп = entry_price + (abs(realized_pnl) * 2.5 / quantity)
            stop_price = entry_price + protected_profit_per_coin
            if price:
                # Не устанавливаем стоп выше текущей цены
                stop_price = min(stop_price, price)
            # Минимально стоп не ниже уровня входа (базовая защита)
            stop_price = max(stop_price, entry_price)
            
            # Break-even LONG расчёт
            price_str = f"{price:.6f}" if price is not None else "None"
            pass  # debug break-even
        else:  # SHORT
            # ✅ Для SHORT: стоп на уровне entry_price - protected_profit_per_coin
            # Если realized_pnl = 0, то protected_profit_per_coin = 0, стоп = entry_price (базовая защита)
            # Если есть realized_pnl, стоп = entry_price - (abs(realized_pnl) * 2.5 / quantity)
            stop_price = entry_price - protected_profit_per_coin
            if price:
                # Не устанавливаем стоп ниже текущей цены (для SHORT стоп выше текущей цены = убыток)
                stop_price = max(stop_price, price)
            # Максимально стоп не выше уровня входа (базовая защита для SHORT)
            stop_price = min(stop_price, entry_price)
            
        return stop_price

    def _ensure_break_even_stop(self, current_price: Optional[float], force: bool = False) -> None:
        """
        Устанавливает стоп-лосс для безубыточности ОДИН РАЗ при активации.
        После установки на бирже, защитный стоп больше не обновляется, чтобы не сбивать трейлинг-стоп.
        """
        if not self.exchange or self.position_side not in ('LONG', 'SHORT'):
            pass
            return

        stop_price = self._calculate_break_even_stop_price(current_price)
        if stop_price is None:
            pass
            return

        # ✅ ИСПРАВЛЕНО: Проверяем наличие стопа на бирже
        # Логика:
        # 1. Если флаг True И стоп есть на бирже И он близок к нашему - пропускаем (бот уже установил)
        # 2. Если флаг False И стоп есть на бирже - устанавливаем наш защитный стоп (это не наш стоп!)
        # 3. Если флаг True И стопа нет на бирже - сбрасываем флаг и устанавливаем стоп
        # 4. Если флаг False И стопа нет на бирже - устанавливаем стоп
        if not force:
            try:
                # Получаем все позиции и находим нужную по символу
                positions = self.exchange.get_positions()
                if isinstance(positions, tuple):
                    positions_list = positions[0] if positions else []
                else:
                    positions_list = positions if positions else []
                
                position = None
                for pos in positions_list:
                    pos_symbol = pos.get('symbol', '').replace('USDT', '')
                    if pos_symbol == self.symbol:
                        # Проверяем сторону позиции
                        pos_side = pos.get('side', '')
                        expected_side = 'Long' if self.position_side == 'LONG' else 'Short' if self.position_side == 'SHORT' else ''
                        if pos_side == expected_side and abs(float(pos.get('size', 0))) > 0:
                            position = pos
                            break
                
                if position:
                    exchange_stop_loss = position.get('stop_loss') or position.get('stopLoss') or position.get('stopLossPrice')
                    # Проверяем, что stop_loss не пустая строка и не None
                    if exchange_stop_loss and str(exchange_stop_loss).strip():
                        try:
                            existing_stop = float(exchange_stop_loss)
                            tolerance = abs(stop_price * 0.001)  # 0.1% tolerance
                            stop_is_close = abs(existing_stop - stop_price) <= tolerance
                            
                            if self.break_even_stop_set:
                                # Флаг True - бот должен был установить стоп
                                if stop_is_close:
                                    # Стоп есть на бирже и близок к нашему - это наш стоп, пропускаем
                                    return
                                else:
                                    # Флаг True, но стоп на бирже отличается - возможно, был изменен вручную
                                    # Устанавливаем наш защитный стоп
                                    logger.warning(
                                        f"[NEW_BOT_{self.symbol}] ⚠️ Флаг break_even_stop_set=True, но стоп на бирже отличается: "
                                        f"exchange_stop={existing_stop:.6f}, break_even={stop_price:.6f}, "
                                        f"устанавливаем наш защитный стоп"
                                    )
                            else:
                                # Флаг False - бот НЕ устанавливал стоп
                                # Если стоп есть на бирже - это НЕ наш защитный стоп, нужно установить свой!
                                logger.info(
                                    f"[NEW_BOT_{self.symbol}] 🔧 Стоп есть на бирже, но флаг break_even_stop_set=False: "
                                    f"exchange_stop={existing_stop:.6f}, break_even={stop_price:.6f}, "
                                    f"устанавливаем наш защитный стоп"
                                )
                        except (TypeError, ValueError):
                            # Не удалось распарсить стоп - устанавливаем наш
                            pass
                    else:
                        # Стопа нет на бирже
                        if self.break_even_stop_set:
                            # Флаг True, но стопа нет - сбрасываем флаг
                            logger.warning(
                                f"[NEW_BOT_{self.symbol}] ⚠️ Флаг break_even_stop_set=True, но стопа нет на бирже! "
                                f"Сбрасываем флаг и устанавливаем стоп."
                            )
                            self.break_even_stop_set = False
                        # Устанавливаем стоп (флаг уже False или был сброшен)
                else:
                    # Не удалось получить позицию - устанавливаем стоп на всякий случай
                    pass
            except Exception:
                # Ошибка при проверке - устанавливаем стоп на всякий случай
                pass

        try:
            previous_stop = self.break_even_stop_price
            result = self.exchange.update_stop_loss(self.symbol, stop_price, self.position_side)
            if result and result.get('success'):
                is_update = previous_stop is not None
                self.break_even_stop_price = stop_price
                # ✅ ИСПРАВЛЕНО: Устанавливаем флаг, что защитный стоп установлен на бирже
                # После этого защитный стоп больше не будет обновляться, чтобы не сбивать трейлинг-стоп
                self.break_even_stop_set = True
                logger.info(
                    f"[NEW_BOT_{self.symbol}] 🛡️ Break-even стоп {'обновлён' if is_update else 'установлен'} "
                    f"(один раз): {stop_price:.6f}"
                )
                # Логируем в историю
                try:
                    from bot_engine.bot_history import log_stop_loss_set
                    log_stop_loss_set(
                        bot_id=self.symbol,
                        symbol=self.symbol,
                        stop_price=stop_price,
                        position_side=self.position_side or 'LONG',
                        is_update=is_update,
                        previous_price=previous_stop
                    )
                except Exception:
                    pass
            else:
                if (result or {}).get('zero_position'):
                    pass
                else:
                    logger.warning(
                        f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось установить break-even стоп: "
                        f"result={result}, message={(result or {}).get('message', 'Unknown')}"
                    )
        except Exception as exc:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка установки break-even стопа: {exc}", exc_info=True)

    def check_protection_mechanisms(self, current_price):
        """Проверяет все защитные механизмы"""
        try:
            decision = self._evaluate_protection_decision(current_price)
            # ✅ ИСПРАВЛЕНО: Обновляем трейлинг-стоплоссы на бирже
            self._update_protection_mechanisms(current_price)
            return {
                'should_close': bool(decision.should_close),
                'reason': decision.reason
            }
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка проверки защитных механизмов: {e}")
            return {'should_close': False, 'reason': None}

    def _update_protection_mechanisms(self, current_price):
        """Обновляет защитные механизмы"""
        try:
            entry_price = self._safe_float(self.entry_price)
            current_price = self._safe_float(current_price)
            if entry_price is None or current_price is None or entry_price == 0:
                return

            # Рассчитываем процент изменения цены (для max_profit_achieved и других целей)
            if self.position_side == 'LONG':
                price_change_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                price_change_percent = ((entry_price - current_price) / entry_price) * 100

            if price_change_percent > self.max_profit_achieved:
                self.max_profit_achieved = price_change_percent
                pass

            # ✅ ИСПРАВЛЕНО: Рассчитываем profit_percent как процент от СТОИМОСТИ СДЕЛКИ (position_value)
            # Триггер защиты безубыточности - это процент от стоимости сделки, а не от цены
            position_size_coins = self._get_position_quantity()
            position_value = 0.0
            profit_usdt = 0.0
            
            # ✅ ИСПРАВЛЕНО: Используем нереализованный P&L напрямую, если доступен
            # Это более точно, чем расчет через цену
            if self.unrealized_pnl_usdt is not None:
                profit_usdt = float(self.unrealized_pnl_usdt)
            elif self.unrealized_pnl is not None:
                profit_usdt = float(self.unrealized_pnl)
            
            # Если нереализованный P&L не доступен, рассчитываем через цену
            if profit_usdt == 0.0 and position_size_coins > 0:
                if self.position_side == 'LONG':
                    profit_usdt = position_size_coins * (current_price - entry_price)
                else:
                    profit_usdt = position_size_coins * (entry_price - current_price)
            
            # Рассчитываем стоимость сделки (position_value)
            # ✅ ИСПРАВЛЕНО: Используем position_size (стоимость позиции в USDT) напрямую, если доступен
            if self.position_size:
                try:
                    position_value = float(self.position_size)
                except (TypeError, ValueError):
                    position_value = 0.0
            elif position_size_coins > 0:
                position_value = entry_price * position_size_coins
            else:
                position_value = 0.0
            
            # Рассчитываем profit_percent от стоимости сделки
            if position_value > 0:
                profit_percent = (profit_usdt / position_value) * 100
            else:
                profit_percent = 0.0

            # ✅ ИСПРАВЛЕНО: Добавлен вызов установки break-even стопа
            # Проверяем, активирована ли защита безубыточности
            config = self._get_effective_protection_config()
            break_even_enabled = bool(config.get('break_even_protection', True))
            break_even_trigger = self._safe_float(
                config.get('break_even_trigger_percent', config.get('break_even_trigger')),
                0.0
            ) or 0.0
            
            if break_even_enabled and break_even_trigger > 0:
                # ✅ ИСПРАВЛЕНО: Если прибыль достигла триггера, активируем защиту
                # Проверяем, нужно ли активировать защиту (даже если она уже была активирована ранее)
                if profit_percent >= break_even_trigger:
                    if not self.break_even_activated:
                        self.break_even_activated = True
                        logger.info(
                            f"[NEW_BOT_{self.symbol}] 🛡️ Защита безубыточности активирована "
                            f"(прибыль {profit_percent:.2f}% >= триггер {break_even_trigger:.2f}%)"
                        )
                    
                    # Если защита активирована, устанавливаем/обновляем стоп
                    self._ensure_break_even_stop(current_price, force=False)
                else:
                    # ✅ ИСПРАВЛЕНО: Если прибыль упала ниже триггера, но защита уже была активирована,
                    # защита остается активной (не деактивируем, чтобы защитить уже достигнутую прибыль)
                    if self.break_even_activated:
                        # Защита остается активной, обновляем стоп
                        self._ensure_break_even_stop(current_price, force=False)
            else:
                # Если защита отключена, деактивируем
                if self.break_even_activated:
                    self.break_even_activated = False
                    self.break_even_stop_price = None
                    self.break_even_stop_set = False
                    logger.info(f"[NEW_BOT_{self.symbol}] 🛡️ Защита безубыточности деактивирована (отключена в конфиге)")

            # ✅ ИСПРАВЛЕНО: Для trailing используем profit_percent (процент от стоимости сделки) для активации
            # Это аналогично break-even защите - триггер активации должен быть процентом от стоимости сделки
            self._update_trailing_stops(current_price, profit_percent, price_change_percent)

        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления защитных механизмов: {e}")

    def _update_trailing_stops(self, current_price: float, profit_percent: float, price_change_percent: float = None) -> Dict[str, Optional[str]]:
        """
        Обновляет трейлинг-стоп и резервный трейлинг-тейк. Возвращает решение о закрытии позиции.
        
        Args:
            current_price: Текущая цена
            profit_percent: Процент прибыли от стоимости сделки (используется для активации)
            price_change_percent: Процент изменения цены (для обратной совместимости и логирования)
        """
        result = {'should_close': False, 'reason': None}

        try:
            config = get_auto_bot_config()
        except Exception:
            config = {}

        activation = self._safe_float(config.get('trailing_stop_activation'), 0.0) or 0.0
        stop_distance = max(0.0, self._safe_float(config.get('trailing_stop_distance'), 0.0) or 0.0)
        take_distance = max(0.0, self._safe_float(config.get('trailing_take_distance'), 0.0) or 0.0)
        update_interval = max(0.0, self._safe_float(config.get('trailing_update_interval'), 0.0) or 0.0)

        self.trailing_activation_threshold = activation
        self.trailing_step_usdt = 0.0
        self.trailing_step_price = 0.0
        self.trailing_steps = 0

        if stop_distance <= 0 or self.position_side not in ('LONG', 'SHORT'):
            self.trailing_active = False
            return result

        current_price = self._safe_float(current_price)
        entry_price = self._safe_float(self.entry_price, current_price)
        if current_price is None or entry_price is None or current_price <= 0:
            return result

        now_ts = time.time()
        tolerance = 1e-8

        # ✅ ИСПРАВЛЕНО: Используем profit_percent (процент от стоимости сделки) для проверки активации
        # Это аналогично break-even защите - триггер активации должен быть процентом от стоимости сделки
        if activation > 0 and profit_percent < activation and not self.trailing_active:
            self.trailing_reference_price = self._safe_float(self.trailing_reference_price, entry_price)
            return result

        # ✅ ИСПРАВЛЕНО: Активируем trailing, если profit_percent >= activation
        # Аналогично break-even - если прибыль достигла триггера, активируем защиту
        if not self.trailing_active:
            if activation > 0 and profit_percent >= activation:
                self.trailing_active = True
                self.trailing_reference_price = current_price
                logger.info(
                    f"[NEW_BOT_{self.symbol}] 🌀 Trailing активирован "
                    f"(прибыль {profit_percent:.2f}% >= триггер {activation:.2f}%)"
                )
            else:
                return result
        else:
            reference = self._safe_float(self.trailing_reference_price, entry_price)
            if self.position_side == 'LONG':
                reference = max(reference or entry_price, current_price)
            else:
                reference = min(reference or entry_price, current_price)
            self.trailing_reference_price = reference

        reference_price = self._safe_float(self.trailing_reference_price, entry_price)

        stop_price = None
        if self.position_side == 'LONG':
            stop_price = reference_price * (1 - stop_distance / 100.0)
            stop_price = max(stop_price, entry_price)
            # ✅ ИСПРАВЛЕНО: Трейлинг-стоп должен быть не хуже защитного стопа (если он установлен)
            # Но защитный стоп не должен сбивать трейлинг-стоп - трейлинг-стоп проверяется всегда
            if self.break_even_stop_price is not None and self.break_even_stop_set:
                # Защитный стоп уже установлен на бирже - трейлинг-стоп должен быть лучше (выше)
                stop_price = max(stop_price, self.break_even_stop_price)
        else:
            stop_price = reference_price * (1 + stop_distance / 100.0)
            stop_price = min(stop_price, entry_price)
            # ✅ ИСПРАВЛЕНО: Трейлинг-стоп должен быть не хуже защитного стопа (если он установлен)
            # Но защитный стоп не должен сбивать трейлинг-стоп - трейлинг-стоп проверяется всегда
            if self.break_even_stop_price is not None and self.break_even_stop_set:
                # Защитный стоп уже установлен на бирже - трейлинг-стоп должен быть лучше (ниже)
                stop_price = min(stop_price, self.break_even_stop_price)

        stop_price = self._safe_float(stop_price)
        previous_stop = self._safe_float(self.trailing_stop_price)

        should_update_stop = False
        if self.position_side == 'LONG':
            if stop_price is not None and (previous_stop is None or stop_price > previous_stop + tolerance):
                should_update_stop = True
        else:
            if stop_price is not None and (previous_stop is None or stop_price < previous_stop - tolerance):
                should_update_stop = True

        can_update_now = update_interval <= 0 or (now_ts - (self.trailing_last_update_ts or 0.0)) >= update_interval

        if should_update_stop and self.exchange and can_update_now:
            try:
                response = self.exchange.update_stop_loss(self.symbol, stop_price, self.position_side)
                if response and response.get('success'):
                    is_update = previous_stop is not None
                    self.trailing_stop_price = stop_price
                    self.trailing_last_update_ts = now_ts
                    logger.info(
                        f"[NEW_BOT_{self.symbol}] 🔁 Trailing стоп {'обновлён' if is_update else 'установлен'}: ref={reference_price:.6f}, stop={stop_price:.6f}"
                    )
                    # Логируем в историю
                    try:
                        from bot_engine.bot_history import log_stop_loss_set
                        log_stop_loss_set(
                            bot_id=self.symbol,
                            symbol=self.symbol,
                            stop_price=stop_price,
                            position_side=self.position_side or 'LONG',
                            is_update=is_update,
                            previous_price=previous_stop
                        )
                    except Exception as log_err:
                        pass
                else:
                    logger.warning(
                        f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось обновить trailing stop: "
                        f"{(response or {}).get('message', 'Unknown error')}"
                    )
            except Exception as exc:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка обновления trailing stop: {exc}")
        elif should_update_stop and not can_update_now:
            pass

        tp_price = None
        if take_distance > 0:
            if self.position_side == 'LONG':
                tp_price = reference_price * (1 - take_distance / 100.0)
                tp_price = max(tp_price, entry_price)
                if stop_price is not None:
                    tp_price = max(tp_price, stop_price + tolerance)
            else:
                tp_price = reference_price * (1 + take_distance / 100.0)
                tp_price = min(tp_price, entry_price)
                if stop_price is not None:
                    tp_price = min(tp_price, stop_price - tolerance)

            tp_price = self._safe_float(tp_price)
            previous_tp = self._safe_float(self.trailing_take_profit_price)

            update_take = False
            if self.position_side == 'LONG':
                if tp_price is not None and (previous_tp is None or tp_price > previous_tp + tolerance):
                    update_take = True
            else:
                if tp_price is not None and (previous_tp is None or tp_price < previous_tp - tolerance):
                    update_take = True

            if update_take and self.exchange and can_update_now:
                try:
                    response = self.exchange.update_take_profit(self.symbol, tp_price, self.position_side)
                    if response and response.get('success'):
                        is_update = previous_tp is not None
                        self.trailing_take_profit_price = tp_price
                        self.trailing_last_update_ts = now_ts
                        logger.info(
                            f"[NEW_BOT_{self.symbol}] 🎯 Trailing тейк {'обновлён' if is_update else 'установлен'}: ref={reference_price:.6f}, take={tp_price:.6f}"
                        )
                        # Логируем в историю
                        try:
                            from bot_engine.bot_history import log_take_profit_set
                            log_take_profit_set(
                                bot_id=self.symbol,
                                symbol=self.symbol,
                                take_profit_price=tp_price,
                                position_side=self.position_side or 'LONG',
                                is_update=is_update,
                                previous_price=previous_tp
                            )
                        except Exception as log_err:
                            pass
                    else:
                        logger.warning(
                            f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось обновить trailing тейк: "
                            f"{(response or {}).get('message', 'Unknown error')}"
                        )
                except Exception as exc:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка обновления trailing тейка: {exc}")

        self.trailing_max_profit_usdt = max(self.trailing_max_profit_usdt, profit_percent)
        if stop_price and reference_price:
            if self.position_side == 'LONG':
                self.trailing_locked_profit = max(0.0, reference_price - stop_price)
            else:
                self.trailing_locked_profit = max(0.0, stop_price - reference_price)

        effective_stop = stop_price if stop_price is not None else previous_stop
        if effective_stop is None:
            return result

        if self.position_side == 'LONG' and current_price <= effective_stop:
            logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Trailing stop (LONG) достигнут: {effective_stop:.6f}")
            result['should_close'] = True
            result['reason'] = f'TRAILING_STOP_{profit_percent:.2f}%'
        elif self.position_side == 'SHORT' and current_price >= effective_stop:
            logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Trailing stop (SHORT) достигнут: {effective_stop:.6f}")
            result['should_close'] = True
            result['reason'] = f'TRAILING_STOP_{profit_percent:.2f}%'

        return result

    def _close_position_on_exchange(self, reason):
        """Закрывает позицию на бирже"""
        try:
            if not self.exchange:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Биржа не инициализирована")
                return False
            
            if not self.position_side:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА: position_side не установлен! Невозможно закрыть позицию!")
                return False
            
            # Получаем актуальный размер позиции с биржи
            position_size = None
            expected_side = 'Long' if self.position_side == 'LONG' else 'Short' if self.position_side == 'SHORT' else self.position_side
            
            try:
                positions = self.exchange.get_positions()
                if isinstance(positions, tuple):
                    positions_list = positions[0] if positions else []
                else:
                    positions_list = positions if positions else []
                
                for pos in positions_list:
                    symbol_name = pos.get('symbol', '')
                    normalized_symbol = symbol_name.replace('USDT', '')
                    if normalized_symbol == self.symbol or symbol_name == self.symbol:
                        pos_side = 'Long' if pos.get('side') in ['Buy', 'Long'] else 'Short'
                        if pos_side == expected_side and abs(float(pos.get('size', 0))) > 0:
                            position_size = abs(float(pos.get('size', 0)))
                            # Сохраняем актуальный размер
                            self.position_size = position_size
                            self.position_size_coins = position_size
                            break
            except Exception as e:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка получения размера позиции с биржи: {e}")
            
            # Если с биржи получить не удалось — используем кешированные значения как fallback
            if position_size is None or position_size <= 0:
                cached_sizes = [
                    self.position_size_coins,
                    self.position_size,
                    (self.volume_value / self.entry_price) if self.entry_price else None
                ]
                for cached_value in cached_sizes:
                    try:
                        if cached_value and abs(float(cached_value)) > 0:
                            position_size = abs(float(cached_value))
                            logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Используем кешированный размер позиции: {position_size}")
                            break
                    except (TypeError, ValueError):
                        continue
            
            if position_size is None or position_size <= 0:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось определить размер позиции для закрытия!")
                return False
            
            # ✅ КРИТИЧНО: Преобразуем side в формат, который ожидает биржа ('Long'/'Short')
            side_for_exchange = 'Long' if self.position_side == 'LONG' else 'Short' if self.position_side == 'SHORT' else self.position_side
            
            
            # Закрываем позицию на бирже
            close_result = self.exchange.close_position(
                symbol=self.symbol,
                size=position_size,
                side=side_for_exchange  # ✅ Используем правильный формат
            )
            
            
            if close_result and close_result.get('success'):
                
                # Сохраняем историю закрытия позиции (для обучения ИИ)
                try:
                    self._log_position_closed(reason, close_result)
                except Exception as log_error:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка логирования закрытия: {log_error}")
                
                # 🎓 Обратная связь для обучения ИИ (если есть backtest_result)
                if hasattr(self, '_last_backtest_result') and self._last_backtest_result:
                    try:
                        self._evaluate_ai_prediction(reason, close_result)
                    except Exception as ai_error:
                        logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка оценки ИИ: {ai_error}")
                
                # КРИТИЧНО: Обновляем статус бота
                old_status = self.status
                self.update_status(BOT_STATUS['IDLE'])
                self.position_side = None
                self.entry_price = None
                self.unrealized_pnl = 0
                self.break_even_stop_price = None
                self.break_even_stop_set = False
                
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ Статус бота обновлен: {old_status} → {BOT_STATUS['IDLE']}")
                
                # КРИТИЧНО: Сохраняем состояние бота в bots_data
                try:
                    with bots_data_lock:
                        if self.symbol in bots_data['bots']:
                            bots_data['bots'][self.symbol] = self.to_dict()
                            logger.info(f"[NEW_BOT_{self.symbol}] ✅ Состояние бота сохранено в bots_data")
                except Exception as save_error:
                    logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка сохранения состояния бота: {save_error}")
                
                return True
            else:
                error = close_result.get('error', 'Unknown error') if close_result else 'No response'
                error_msg = close_result.get('message', error) if close_result else error
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ НЕ УДАЛОСЬ закрыть позицию на бирже!")
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка: {error_msg}")
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Полный ответ: {close_result}")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА при закрытии позиции: {e}")
            import traceback
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Traceback: {traceback.format_exc()}")
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
            
            # Получаем размер позиции
            position_size = None
            if self.position_size:
                position_size = self.position_size
            else:
                # Получаем размер позиции с биржи
                try:
                    positions = self.exchange.get_positions()
                    if isinstance(positions, tuple):
                        positions_list = positions[0] if positions else []
                    else:
                        positions_list = positions if positions else []
                    
                    for pos in positions_list:
                        if pos.get('symbol', '').replace('USDT', '') == self.symbol:
                            pos_side = 'Long' if pos.get('side') == 'Buy' else 'Short'
                            expected_side = 'Long' if self.position_side == 'LONG' else 'Short' if self.position_side == 'SHORT' else self.position_side
                            if pos_side == expected_side and abs(float(pos.get('size', 0))) > 0:
                                position_size = abs(float(pos.get('size', 0)))
                                break
                except Exception as e:
                    logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка получения размера позиции: {e}")
            
            if not position_size:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось определить размер позиции для экстренного закрытия")
                return False
            
            # Преобразуем side в формат биржи
            side_for_exchange = 'Long' if self.position_side == 'LONG' else 'Short' if self.position_side == 'SHORT' else self.position_side
            
            # Экстренное закрытие рыночным ордером
            emergency_result = self.exchange.close_position(
                symbol=self.symbol,
                size=position_size,
                side=side_for_exchange,
                order_type='Market'  # Принудительно рыночный ордер
            )
            
            if emergency_result and emergency_result.get('success'):
                logger.warning(f"[NEW_BOT_{self.symbol}] ✅ ЭКСТРЕННОЕ ЗАКРЫТИЕ УСПЕШНО: Позиция закрыта рыночным ордером")
                
                # Логируем закрытие позиции
                try:
                    self._log_position_closed('DELISTING_EMERGENCY', emergency_result)
                except Exception as log_error:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка логирования экстренного закрытия: {log_error}")
                
                self.update_status(BOT_STATUS['IDLE'])
                
                # Дополнительно обнуляем все данные позиции
                self.position_side = None
                self.entry_price = None
                self.unrealized_pnl = 0.0
                self.max_profit_achieved = 0.0
                self.trailing_stop_price = None
                self.break_even_activated = False
                self.trailing_active = False
                self.trailing_activation_profit = 0.0
                self.trailing_activation_threshold = 0.0
                self.trailing_locked_profit = 0.0
                self.trailing_max_profit_usdt = 0.0
                self.trailing_step_usdt = 0.0
                self.trailing_step_price = 0.0
                self.trailing_steps = 0
                self.break_even_stop_price = None
                self.break_even_stop_set = False
                
                return True
            else:
                error = emergency_result.get('error', 'Unknown error') if emergency_result else 'No response'
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ ЭКСТРЕННОЕ ЗАКРЫТИЕ НЕУДАЧНО: {error}")
                return False
                
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ КРИТИЧЕСКАЯ ОШИБКА ЭКСТРЕННОГО ЗАКРЫТИЯ: {e}")
            return False
    
    def calculate_dynamic_take_profit(self, side, actual_entry_price, actual_leverage, actual_qty, tp_percent=None):
        """
        Рассчитывает Take Profit от маржи с учетом плеча
        
        Args:
            side (str): Направление позиции ('LONG' или 'SHORT')
            actual_entry_price (float): Реальная цена входа
            actual_leverage (float): Реальное плечо
            actual_qty (float): Реальное количество монет
            tp_percent (float, optional): TP процент от маржи. Если не указан, берется из конфига.
            
        Returns:
            float: Цена Take Profit
        """
        try:
            # Если tp_percent не указан, получаем из конфига (по умолчанию 100%)
            if tp_percent is None:
                auto_bot_config = get_auto_bot_config()
                tp_percent = auto_bot_config.get('take_profit_percent', 100.0)
            
            # Рассчитываем маржу и прибыль
            position_value = abs(actual_qty) * actual_entry_price if actual_qty else self.volume_value
            margin = position_value / actual_leverage
            target_profit_usdt = margin * (tp_percent / 100)
            
            # Рассчитываем прибыль на монету
            profit_per_coin = target_profit_usdt / abs(actual_qty) if actual_qty and abs(actual_qty) > 0 else (target_profit_usdt / (self.volume_value / actual_entry_price))
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🎯 TP CALC: side={side}, entry={actual_entry_price}, leverage={actual_leverage}x, margin={margin:.4f} USDT, target_profit={target_profit_usdt:.4f} USDT (+{tp_percent}%)")
            
            if side == 'LONG':
                # Для LONG: TP выше
                tp_price = actual_entry_price + profit_per_coin
                logger.info(f"[NEW_BOT_{self.symbol}] ✅ TP для LONG: {actual_entry_price:.6f} → {tp_price:.6f} (+{tp_percent}% от маржи)")
                return tp_price
                
            elif side == 'SHORT':
                # Для SHORT: TP ниже
                tp_price = actual_entry_price - profit_per_coin
                logger.info(f"[NEW_BOT_{self.symbol}] 📉 TP для SHORT: {actual_entry_price:.6f} → {tp_price:.6f} (+{tp_percent}% от маржи)")
                return tp_price
            
            return None
            
        except Exception as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка расчета TP: {e}")
            return None
    
    def _log_position_closed(self, reason, close_result):
        """Сохраняет детальные данные о закрытии позиции (для обучения ИИ)"""
        try:
            from bot_engine.bot_history import bot_history_manager
            
            # Получаем данные о закрытии
            exit_price = close_result.get('price', self.entry_price) if close_result else self.entry_price
            
            # КРИТИЧНО: Рассчитываем PnL из цен входа и выхода, а не из накопленного realized_pnl
            # Накопленный realized_pnl из кошелька - это сумма всех сделок, а не конкретной сделки!
            if self.entry_price and exit_price and self.entry_price > 0:
                # Получаем размер позиции (в USDT)
                # volume_value - размер позиции в USDT
                # position_size_coins - размер позиции в монетах (если есть)
                position_size = getattr(self, 'volume_value', None)
                position_size_coins = getattr(self, 'position_size_coins', None) or getattr(self, 'position_size', None)
                if (not position_size or position_size <= 0) and position_size_coins and self.entry_price:
                    position_size = position_size_coins * self.entry_price  # Конвертируем в USDT
                
                # Если все еще нет размера, используем значение по умолчанию
                if not position_size or position_size <= 0:
                    position_size = 10.0  # Значение по умолчанию
                    position_size_coins = position_size / self.entry_price if self.entry_price else None
                
                # Рассчитываем ROI (процент изменения цены)
                if self.position_side == 'LONG':
                    roi_percent = (exit_price - self.entry_price) / self.entry_price
                else:  # SHORT
                    roi_percent = (self.entry_price - exit_price) / self.entry_price
                
                # Рассчитываем PnL в USDT
                # Если position_size в USDT (размер позиции), то PnL = roi_percent * position_size
                if position_size and position_size > 0:
                    pnl = roi_percent * position_size
                else:
                    # Если нет размера позиции, используем ROI в процентах
                    pnl = roi_percent * 100
                
                # ROI в процентах
                pnl_pct = roi_percent * 100
            else:
                # Fallback: используем значения из close_result или unrealized_pnl
                pnl = close_result.get('realized_pnl', self.unrealized_pnl) if close_result else self.unrealized_pnl
                pnl_pct = close_result.get('roi', 0) if close_result else 0
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось рассчитать PnL из цен, используем fallback: pnl={pnl}")
            
            # КРИТИЧНО ДЛЯ ОБУЧЕНИЯ AI: Сохраняем данные ВСЕГДА, не только для стопов!
            # Получаем RSI и тренд на момент входа из истории или из сохраненного контекста
            entry_rsi = None
            entry_trend = None
            
            # Пытаемся получить из сохраненного контекста или из истории
            try:
                # Ищем в истории открытия позиции для этого бота
                from bot_engine.bot_history import bot_history_manager
                history = bot_history_manager.get_bot_history(symbol=self.symbol, action_type='POSITION_OPENED', limit=1)
                if history:
                    entry_rsi = history[0].get('rsi')
                    entry_trend = history[0].get('trend')
            except Exception as e:
                pass
            
            # Если не нашли в истории, пытаемся из глобальных данных
            if entry_rsi is None or entry_trend is None:
                try:
                    with rsi_data_lock:
                        rsi_info = coins_rsi_data.get(self.symbol, {})
                        from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                        if entry_rsi is None:
                            entry_rsi = get_rsi_from_coin_data(rsi_info)
                        if entry_trend is None:
                            entry_trend = get_trend_from_coin_data(rsi_info)
                except Exception:
                    pass
            
            # Получаем входные данные для обучения ИИ (ВСЕГДА, не только для стопов!)
            entry_data = {
                'entry_price': self.entry_price,
                'rsi': entry_rsi,  # КРИТИЧНО: RSI на момент входа
                'volatility': getattr(self, 'entry_volatility', None),
                'trend': entry_trend or getattr(self, 'entry_trend', None),  # Тренд на момент входа
                'duration_hours': (self.position_start_time and 
                                 (datetime.now() - self.position_start_time).total_seconds() / 3600) if self.position_start_time else 0,
                'max_profit_achieved': self.max_profit_achieved,
                'position_size_usdt': position_size,
                'position_size_coins': position_size_coins,
                'position_leverage': getattr(self, 'leverage', None)
            }
            
            # Получаем текущие рыночные данные при выходе
            exit_rsi = None
            exit_trend = None
            try:
                with rsi_data_lock:
                    rsi_info = coins_rsi_data.get(self.symbol, {})
                    from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                    exit_rsi = get_rsi_from_coin_data(rsi_info)
                    exit_trend = get_trend_from_coin_data(rsi_info)
            except Exception:
                pass
            
            market_data = {
                'exit_price': exit_price,
                'rsi': exit_rsi,  # RSI на момент выхода
                'volatility': None,  # TODO: Получить текущую волатильность
                'trend': exit_trend,  # Тренд на момент выхода
                'price_movement': ((exit_price - self.entry_price) / self.entry_price * 100) if self.entry_price and self.entry_price > 0 else 0
            }
            
            logger.info(f"[NEW_BOT_{self.symbol}] 📊 Логируем закрытие: Entry RSI={entry_rsi}, Entry Trend={entry_trend}, Exit RSI={exit_rsi}, Exit Trend={exit_trend}")
            
            # Сохраняем в историю (bot_history.json и ai_data.db)
            bot_history_manager.log_position_closed(
                bot_id=self.symbol,
                symbol=self.symbol,
                direction=self.position_side,
                exit_price=exit_price,
                pnl=pnl,
                roi=pnl_pct,
                reason=reason,
                entry_data=entry_data,
                market_data=market_data,
                is_simulated=False  # КРИТИЧНО: реальные боты - это НЕ симуляция!
            )
            
            # КРИТИЧНО: Также сохраняем в bots_data.db для истории торговли ботов
            try:
                from bot_engine.bots_database import get_bots_database
                bots_db = get_bots_database()
                
                # Формируем данные для сохранения
                trade_data = {
                    'bot_id': self.symbol,
                    'symbol': self.symbol,
                    'direction': self.position_side,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'entry_time': self.position_start_time.isoformat() if self.position_start_time else None,
                    'exit_time': datetime.now().isoformat(),
                    'entry_timestamp': self.position_start_time.timestamp() * 1000 if self.position_start_time else None,
                    'exit_timestamp': datetime.now().timestamp() * 1000,
                    'position_size_usdt': position_size,
                    'position_size_coins': position_size_coins,
                    'pnl': pnl,
                    'roi': pnl_pct,
                    'status': 'CLOSED',
                    'close_reason': reason,
                    'decision_source': getattr(self, 'decision_source', 'SCRIPT'),
                    'ai_decision_id': getattr(self, 'ai_decision_id', None),
                    'ai_confidence': getattr(self, 'ai_confidence', None),
                    'entry_rsi': entry_rsi,
                    'exit_rsi': exit_rsi,
                    'entry_trend': entry_trend or getattr(self, 'entry_trend', None),
                    'exit_trend': exit_trend,
                    'entry_volatility': entry_data.get('volatility'),
                    'entry_volume_ratio': None,  # TODO: получить из entry_data если есть
                    'is_successful': pnl > 0,
                    'is_simulated': False,
                    'source': 'bot',
                    'order_id': close_result.get('order_id') if close_result else None,
                    'extra_data': {
                        'entry_data': entry_data,
                        'market_data': market_data
                    }
                }
                
                trade_id = bots_db.save_bot_trade_history(trade_data)
                if trade_id:
                    logger.info(f"[NEW_BOT_{self.symbol}] ✅ История сделки сохранена в bots_data.db (ID: {trade_id})")
                else:
                    logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось сохранить историю сделки в bots_data.db")
            except Exception as bots_db_error:
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка сохранения истории в bots_data.db: {bots_db_error}")
            
            # ✅ КРИТИЧНО: Сохраняем timestamp последнего закрытия для задержки перед следующим входом
            try:
                current_timestamp = datetime.now().timestamp()
                self.config['last_position_close_timestamp'] = current_timestamp
                
                # Также обновляем в bots_data для персистентности
                from bots_modules.imports_and_globals import bots_data, bots_data_lock
                with bots_data_lock:
                    if self.symbol in bots_data['bots']:
                        bots_data['bots'][self.symbol]['last_position_close_timestamp'] = current_timestamp
                
                try:
                    from bot_engine.bot_config import get_current_timeframe
                    _tf = get_current_timeframe()
                except Exception:
                    _tf = '?'
                logger.info(f"[NEW_BOT_{self.symbol}] ⏰ Сохранен timestamp последнего закрытия: {current_timestamp} (через 1 свечу {_tf} разрешим новый вход)")
            except Exception as timestamp_error:
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Ошибка сохранения timestamp закрытия: {timestamp_error}")
            
            # ВАЖНО: Обновляем результат решения AI для переобучения
            if hasattr(self, 'ai_decision_id') and self.ai_decision_id:
                try:
                    from bot_engine.ai.ai_integration import update_ai_decision_result
                    is_successful = pnl > 0
                    update_ai_decision_result(self.ai_decision_id, pnl, pnl_pct, is_successful)
                    pass
                    self.ai_decision_id = None
                except Exception as ai_track_error:
                    pass
            
        except Exception as e:
            pass
    
    def _evaluate_ai_prediction(self, reason, close_result):
        """Оценивает предсказание ИИ и сохраняет для обучения"""
        try:
            from bot_engine.ai.smart_risk_manager import SmartRiskManager
            from bot_engine.bot_history import bot_history_manager
            
            # Получаем данные о реальном результате
            exit_price = close_result.get('price', self.entry_price) if close_result else self.entry_price
            pnl = close_result.get('realized_pnl', self.unrealized_pnl) if close_result else self.unrealized_pnl
            pnl_pct = close_result.get('roi', 0) if close_result else 0
            
            actual_outcome = {
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'roi': pnl_pct,
                'reason': reason
            }
            
            # Оцениваем предсказание
            smart_risk = SmartRiskManager()
            evaluation = smart_risk.evaluate_prediction(
                self.symbol,
                self._last_backtest_result,
                actual_outcome
            )
            
            logger.info(f"[NEW_BOT_{self.symbol}] 🎓 ИИ оценен: score={evaluation.get('score', 0):.2f}")
            
        except Exception as e:
            pass
    
    def to_dict(self):
        """Преобразует бота в словарь для сохранения"""
        # Получаем дополнительные данные из конфига если есть
        bot_id = self.config.get('id', f"{self.symbol}_{int(datetime.now().timestamp())}")
        
        return {
            'id': bot_id,
            'symbol': self.symbol,
            'status': self.status,
            'auto_managed': self.config.get('auto_managed', False),
            'volume_mode': self.volume_mode,
            'volume_value': self.volume_value,
            'position': None,  # Для совместимости
            'entry_price': self.entry_price,
            'entry_time': self.position_start_time.isoformat() if self.position_start_time else None,
            'position_side': self.position_side,
            'position_size': self.position_size,  # ✅ Размер позиции в монетах
            'position_size_coins': self.position_size_coins,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_usdt': self.unrealized_pnl_usdt,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage,
            'margin_usdt': self.margin_usdt,
            'created_at': self.created_at,
            'last_signal_time': self.last_signal_time,
            'last_bar_timestamp': None,  # Для совместимости
            'max_profit_achieved': self.max_profit_achieved,
            'trailing_stop_price': self.trailing_stop_price,
            'trailing_activation_threshold': self.trailing_activation_threshold,
            'trailing_activation_profit': self.trailing_activation_profit,
            'trailing_locked_profit': self.trailing_locked_profit,
            'trailing_active': self.trailing_active,
            'trailing_max_profit_usdt': self.trailing_max_profit_usdt,
            'trailing_step_usdt': self.trailing_step_usdt,
            'trailing_step_price': self.trailing_step_price,
            'trailing_steps': self.trailing_steps,
            'trailing_reference_price': self.trailing_reference_price,
            'trailing_last_update_ts': self.trailing_last_update_ts,
            'trailing_take_profit_price': self.trailing_take_profit_price,
            'break_even_activated': self.break_even_activated,
            'break_even_stop_set': self.break_even_stop_set,
            'break_even_stop_price': self.break_even_stop_price,
            'position_start_time': self.position_start_time.isoformat() if self.position_start_time else None,
            'order_id': self.order_id,
            'entry_timestamp': self.entry_timestamp,
            'opened_by_autobot': self.opened_by_autobot,
            'entry_trend': self.entry_trend,  # ✅ Сохраняем тренд при входе
            'entry_timeframe': self.entry_timeframe,  # ✅ Сохраняем таймфрейм при входе
            'scaling_enabled': False,  # Для совместимости
            'scaling_levels': [],  # Для совместимости
            'scaling_current_level': 0,  # Для совместимости
            'scaling_group_id': None,  # Для совместимости
            # Добавляем стопы и тейки если они есть
            'stop_loss': getattr(self, 'stop_loss', None) or self.config.get('stop_loss'),
            'take_profit': getattr(self, 'take_profit', None) or self.config.get('take_profit'),
            'current_price': getattr(self, 'current_price', None) or self.config.get('current_price'),
            'ai_decision_id': getattr(self, 'ai_decision_id', None)
        }

    def _build_trading_bot_bridge_config(self):
        """Формирует конфиг для TradingBot при ручном открытии позиции."""
        try:
            with bots_data_lock:
                auto_config = dict(bots_data.get('auto_bot_config', {}))
        except Exception:
            auto_config = {}

        config = {
            'auto_managed': True,
            'status': 'idle',
            'volume_mode': self.volume_mode,
            'volume_value': self.volume_value,
            'max_loss_percent': self.config.get('max_loss_percent', auto_config.get('max_loss_percent', 10)),
            'take_profit_percent': self.config.get('take_profit_percent', auto_config.get('take_profit_percent', 20)),
            'break_even_protection': self.config.get('break_even_protection', auto_config.get('break_even_protection', True)),
            'break_even_trigger': self.config.get('break_even_trigger', auto_config.get('break_even_trigger', 20)),
            'trailing_stop_activation': self.config.get('trailing_stop_activation', auto_config.get('trailing_stop_activation', 30)),
            'trailing_stop_distance': self.config.get('trailing_stop_distance', auto_config.get('trailing_stop_distance', 5)),
            'trailing_take_distance': self.config.get('trailing_take_distance', auto_config.get('trailing_take_distance', 0.5)),
            'trailing_update_interval': self.config.get('trailing_update_interval', auto_config.get('trailing_update_interval', 3)),
        }

        # Переносим дополнительные индивидуальные параметры, если они были сохранены в self.config
        for key in ('rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                    'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                    'entry_trend'):
            if key in self.config:
                config[key] = self.config[key]

        return config

    def enter_position(self, direction: str, force_market_entry: bool = True):
        """
        Открывает позицию через TradingBot, используя текущие настройки бота.
        Args:
            direction: 'LONG' или 'SHORT'
            force_market_entry: True — автовход, всегда по рынку (лимитные ордера не используются).
        """
        if not direction:
            raise ValueError("Direction is required")

        side = direction.upper()
        if side not in ('LONG', 'SHORT'):
            raise ValueError(f"Unsupported direction {direction}")

        if not self.exchange:
            raise RuntimeError("Exchange is not initialized")

        try:
            from bot_engine.trading_bot import TradingBot
            from bot_engine.bot_config import BotStatus
        except ImportError as e:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось импортировать TradingBot: {e}")
            raise

        bridge_config = self._build_trading_bot_bridge_config()

        trading_bot = TradingBot(self.symbol, self.exchange, bridge_config)
        
        # ✅ Логируем перед входом в позицию для диагностики
        logger.info(f"[NEW_BOT_{self.symbol}] 🚀 Вызываем _enter_position({side}) для входа в позицию" + (" по рынку" if force_market_entry else ""))
        
        result = trading_bot._enter_position(side, force_market_entry=force_market_entry)
        
        # ✅ Логируем результат
        if result:
            logger.info(f"[NEW_BOT_{self.symbol}] 📊 Результат входа: success={result.get('success')}, action={result.get('action')}, error={result.get('error')}")

        if not result.get('success'):
            error_msg = result.get('message') or result.get('error') or 'Unknown error'
            # Блокировка фильтрами - это нормальная работа системы, логируем как WARNING
            if result.get('error') == 'filters_blocked' or 'заблокирован фильтрами' in error_msg:
                logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось открыть позицию {side}: {error_msg}")
            else:
                logger.error(f"[NEW_BOT_{self.symbol}] ❌ Не удалось открыть позицию {side}: {error_msg}")
            raise RuntimeError(error_msg)

        # ✅ КРИТИЧНО: Получаем тренд на момент входа для правильного определения порога выхода
        # Получаем тренд из сохраненного контекста или из глобальных данных RSI
        entry_trend_value = None
        ctx = getattr(self, '_last_entry_context', {}) or {}
        entry_trend_value = ctx.get('trend')
        
        # Если контекст пустой, пытаемся получить из глобальных данных RSI
        if entry_trend_value is None:
            try:
                from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
                with rsi_data_lock:
                    rsi_info = coins_rsi_data.get('coins', {}).get(self.symbol, {})
                    from bot_engine.bot_config import get_trend_from_coin_data
                    entry_trend_value = get_trend_from_coin_data(rsi_info) or rsi_info.get('trend')
            except Exception as e:
                pass
        
        # Обновляем состояние текущего бота
        self.entry_price = result.get('entry_price')
        self.position_side = side
        self.position_size = result.get('quantity')
        self.position_size_coins = result.get('quantity')
        self.position_start_time = datetime.now()
        self.entry_timestamp = datetime.now().timestamp()
        self.entry_trend = entry_trend_value  # ✅ Сохраняем тренд при входе
        
        # ✅ КРИТИЧНО: Сохраняем таймфрейм при входе в позицию
        # Это позволяет боту работать со своим таймфреймом независимо от системного
        from bot_engine.bot_config import get_current_timeframe
        self.entry_timeframe = get_current_timeframe()
        
        target_status = BOT_STATUS['IN_POSITION_LONG'] if side == 'LONG' else BOT_STATUS['IN_POSITION_SHORT']
        self.update_status(target_status, entry_price=self.entry_price, position_side=side)

        try:
            with bots_data_lock:
                bots_data['bots'][self.symbol] = self.to_dict()
        except Exception as save_error:
            logger.error(f"[NEW_BOT_{self.symbol}] ❌ Ошибка сохранения состояния после входа: {save_error}")

        # ✅ КРИТИЧНО: Сразу сохраняем бота в БД с entry_timeframe (1m/6h и т.д.), чтобы при перезапуске лонги 1m не закрывались по 6h
        try:
            from bots_modules.sync_and_cache import save_bots_state
            save_bots_state()
        except Exception as persist_err:
            logger.warning(f"[NEW_BOT_{self.symbol}] ⚠️ Не удалось сразу сохранить состояние в БД: {persist_err}")

        logger.info(f"[NEW_BOT_{self.symbol}] ✅ Позиция {side} открыта: qty={self.position_size} price={self.entry_price}")
        if result.get('success'):
            self._on_position_opened(
                direction=side,
                entry_price=self.entry_price,
                position_size=self._get_position_quantity()
            )
        return result

