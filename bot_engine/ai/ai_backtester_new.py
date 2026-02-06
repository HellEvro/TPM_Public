#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль бэктестинга стратегий

Тестирует торговые стратегии на исторических данных
"""

import os
import json
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from bot_engine.protections import ProtectionState, evaluate_protections
from bot_engine.ai.filter_utils import apply_entry_filters
from bot_engine.utils.rsi_utils import calculate_rsi_history

logger = logging.getLogger('AI.Backtester')

_individual_settings_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _get_cached_individual_settings(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    """Фолбек для получения индивидуальных настроек, когда bots_modules недоступен."""
    if not symbol:
        return None
    normalized = symbol.upper()
    global _individual_settings_cache  # noqa: WPS420
    try:
        if _individual_settings_cache is None:
            from bot_engine.storage import load_individual_coin_settings  # noqa: WPS433,E402

            _individual_settings_cache = load_individual_coin_settings() or {}
        settings = _individual_settings_cache.get(normalized)
        return deepcopy(settings) if settings else None
    except Exception as exc:  # pragma: no cover - резервный путь
        pass
        return None


def _get_config_snapshot(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращает срез настроек Auto Bot (глобальный + overrides), используется тренером и бэктестером.
    """
    try:
        from bots_modules.imports_and_globals import get_config_snapshot  # noqa: WPS433,E402

        return get_config_snapshot(symbol)
    except Exception as exc:  # pragma: no cover - fallback при отсутствии сервиса ботов
        pass
        try:
            from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG  # noqa: WPS433,E402

            global_config = deepcopy(DEFAULT_AUTO_BOT_CONFIG)
        except Exception:
            global_config = {}
        individual_config = _get_cached_individual_settings(symbol) if symbol else None
        merged_config = deepcopy(global_config)
        if individual_config:
            merged_config.update(individual_config)
        return {
            'global': global_config,
            'individual': individual_config,
            'merged': merged_config,
            'symbol': symbol.upper() if symbol else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def _normalize_timestamp(raw_ts: Any) -> Optional[float]:
    """Преобразует таймстамп (мс/с/iso) в секунды."""
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        value = float(raw_ts)
        if value > 1e12:
            return value / 1000.0
        return value
    if isinstance(raw_ts, str):
        try:
            return datetime.fromisoformat(raw_ts.replace('Z', '')).timestamp()
        except ValueError:
            try:
                value = float(raw_ts)
                return _normalize_timestamp(value)
            except ValueError:
                return None
    return None


def _create_protection_state(direction: str, entry_price: float, notional_usdt: float, entry_ts: Any) -> ProtectionState:
    safe_price = float(entry_price) if entry_price else 0.0
    quantity = None
    if safe_price > 0 and notional_usdt:
        quantity = notional_usdt / safe_price
    return ProtectionState(
        position_side=direction,
        entry_price=safe_price,
        entry_time=_normalize_timestamp(entry_ts),
        quantity=quantity,
        notional_usdt=notional_usdt,
    )


def _determine_trend(closes: List[float], index: int, window: int) -> str:
    if not closes or index <= 0:
        return 'NEUTRAL'
    lookback = max(1, min(window or 1, index))
    base_price = closes[index - lookback]
    current_price = closes[index]
    if current_price > base_price:
        return 'UP'
    if current_price < base_price:
        return 'DOWN'
    return 'NEUTRAL'


class AIBacktester:
    """
    Класс для бэктестинга торговых стратегий
    """
    
    def __init__(self):
        """Инициализация бэктестера"""
        # УДАЛЕНО: self.results_dir - результаты теперь сохраняются в БД (backtest_results)
        self.data_dir = 'data/ai'
        config_snapshot = _get_config_snapshot()
        self.auto_bot_config = config_snapshot.get('global', {})
        
        # Создаем только основную директорию (для БД и моделей)
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("✅ AIBacktester инициализирован")
    
    def _load_market_data(self) -> Dict:
        """
        Загрузить рыночные данные
        
        ВАЖНО: Использует ТОЛЬКО БД (таблица candles_history)
        Свечи загружаются через пагинацию по 2000 свечей для каждой монеты
        """
        try:
            market_data = {'latest': {'candles': {}}}
            candles_data = {}
            
            # Загружаем ТОЛЬКО из БД
            try:
                from bot_engine.ai.ai_database import get_ai_database
                ai_db = get_ai_database()
                if not ai_db:
                    logger.warning("⚠️ AI Database не доступна")
                    return market_data
                
                # Ограничиваем загрузку (при AI_MEMORY_LIMIT_MB лимиты из AILauncherConfig)
                from bot_engine.config_loader import get_current_timeframe
                try:
                    from bot_engine.ai.ai_launcher_config import AILauncherConfig
                    _max_sym = min(30, AILauncherConfig.MAX_SYMBOLS_FOR_CANDLES)
                    _max_candles = AILauncherConfig.MAX_CANDLES_PER_SYMBOL
                except Exception:
                    _max_sym, _max_candles = 30, 1000
                candles_data = ai_db.get_all_candles_dict(
                    timeframe=get_current_timeframe(),
                    max_symbols=_max_sym,
                    max_candles_per_symbol=_max_candles
                )
                if candles_data:
                    total_candles = sum(len(c) for c in candles_data.values())
                    logger.info(f"✅ Загружено {len(candles_data)} монет из БД ({total_candles:,} свечей, ограничено для экономии памяти)")
                else:
                    logger.warning("⚠️ БД пуста, ожидаем загрузки свечей...")
                    return market_data
            except Exception as db_error:
                logger.error(f"❌ Ошибка загрузки из БД: {db_error}")
                import traceback
                logger.error(traceback.format_exc())
                return market_data
            
            if candles_data:
                logger.info(f"✅ Загружено полной истории для {len(candles_data)} монет")
                
                if 'latest' not in market_data:
                    market_data['latest'] = {}
                if 'candles' not in market_data['latest']:
                    market_data['latest']['candles'] = {}
                
                for symbol, candle_info in candles_data.items():
                    if isinstance(candle_info, dict):
                        candles = candle_info.get('candles', [])
                    else:
                        candles = candle_info if isinstance(candle_info, list) else []
                    
                    if candles:
                        market_data['latest']['candles'][symbol] = {
                            'candles': candles,
                            'timeframe': get_current_timeframe(),
                            'last_update': datetime.now().isoformat(),
                            'count': len(candles),
                            'source': 'ai_data.db'
                        }
                
                logger.info(f"✅ Обработано: {len(market_data['latest']['candles'])} монет")
            else:
                logger.warning("⚠️ Нет данных свечей для бэктеста")
            
            return market_data
            
            # 2. Получаем индикаторы через API
            try:
                import requests
                response = requests.get('http://127.0.0.1:5001/api/bots/coins-with-rsi', timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        coins_data = data.get('coins', {})
                        logger.info(f"✅ Загружено индикаторов для {len(coins_data)} монет через API")
                        
                        # Получаем RSI и тренд с учетом текущего таймфрейма
                        from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
                        
                        for symbol, coin_data in coins_data.items():
                            market_data['latest']['indicators'][symbol] = {
                                'rsi': get_rsi_from_coin_data(coin_data),
                                'trend': get_trend_from_coin_data(coin_data),
                                'price': coin_data.get('price'),
                                'signal': coin_data.get('signal'),
                                'volume': coin_data.get('volume')
                            }
            except Exception as api_error:
                pass
            
            return market_data
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки рыночных данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _load_history_data(self, with_rsi_only: bool = False) -> List[Dict]:
        """
        Загрузить историю трейдов.

        ПРИОРИТЕТ: БД (если доступна), затем bot_history.json
        history_data.json больше не используется, так как все данные в БД.

        Args:
            with_rsi_only: Если True, включать только сделки с заполненным RSI (для бэктеста).
        """
        trades = []
        
        # 1. ПРИОРИТЕТ: Загружаем из БД (если доступна)
        try:
            from bot_engine.ai.ai_database import get_ai_database
            ai_db = get_ai_database()
            if ai_db:
                # ВАЖНО: Включаем симуляции для обучения ИИ на разных параметрах
                db_trades = ai_db.get_trades_for_training(
                    include_simulated=True,  # ВКЛЮЧАЕМ симуляции для обучения!
                    include_real=True,
                    include_exchange=True,  # ВАЖНО: Включаем сделки с биржи!
                    min_trades=0,  # КРИТИЧНО: 0 чтобы получить все сделки, не фильтровать по символам
                    limit=None
                )
                if db_trades:
                    # Конвертируем формат БД в формат для обучения
                    for trade in db_trades:
                        if with_rsi_only:
                            rsi_val = trade.get('rsi') if trade.get('rsi') is not None else trade.get('entry_rsi')
                            if rsi_val is None:
                                continue
                        converted_trade = {
                            'id': f"db_{trade.get('symbol')}_{trade.get('timestamp', '')}",
                            'timestamp': trade.get('timestamp') or trade.get('entry_time'),
                            'bot_id': trade.get('bot_id', trade.get('symbol')),
                            'symbol': trade.get('symbol'),
                            'direction': trade.get('direction'),
                            'entry_price': trade.get('entry_price'),
                            'exit_price': trade.get('exit_price'),
                            'pnl': trade.get('pnl'),
                            'roi': trade.get('roi'),
                            'status': 'CLOSED',
                            'decision_source': trade.get('decision_source', 'SCRIPT'),
                            'rsi': trade.get('rsi') or trade.get('entry_rsi'),
                            'entry_rsi': trade.get('entry_rsi'),
                            'exit_rsi': trade.get('exit_rsi'),
                            'trend': trade.get('trend'),
                            'close_timestamp': trade.get('close_timestamp') or trade.get('exit_time'),
                            'close_reason': trade.get('close_reason'),
                            'is_successful': trade.get('is_successful', False),
                            'is_simulated': False,
                            'entry_data': trade.get('entry_data') or {'rsi': trade.get('entry_rsi')},
                            'exit_market_data': trade.get('exit_market_data') or {'rsi': trade.get('exit_rsi')},
                        }
                        trades.append(converted_trade)
                    
                    if trades:
                        return trades
        except Exception as e:
            pass
        
        # 2. Fallback: загружаем из bot_history.json или API
        try:
            history_file = os.path.join(self.data_dir, 'history_data.json')
            if not os.path.exists(history_file):
                pass
                # Пробуем получить через API
                try:
                    import requests
                    response = requests.get('http://127.0.0.1:5001/api/bots/trades?limit=1000', timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            return data.get('trades', [])
                except:
                    pass
                return []
            
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as json_error:
                logger.warning(f"⚠️ Файл истории поврежден (JSON ошибка на строке {json_error.lineno}, колонка {json_error.colno}): {history_file}")
                # Пробуем получить через API как резервный вариант
                try:
                    import requests
                    response = requests.get('http://127.0.0.1:5001/api/bots/trades?limit=1000', timeout=5)
                    if response.status_code == 200:
                        api_data = response.json()
                        if api_data.get('success'):
                            return api_data.get('trades', [])
                except:
                    pass
                return []
            
            trades = []
            latest = data.get('latest', {})
            history = data.get('history', [])
            
            if latest:
                trades.extend(latest.get('trades', []))
            
            for entry in history:
                trades.extend(entry.get('trades', []))
            
            # Убираем дубликаты по ID
            seen_ids = set()
            unique_trades = []
            for trade in trades:
                trade_id = trade.get('id')
                if trade_id and trade_id not in seen_ids:
                    seen_ids.add(trade_id)
                    unique_trades.append(trade)
            
            # Убрано: logger.debug(f"📊 Загружено {len(unique_trades)} уникальных сделок из истории") - слишком шумно
            return unique_trades
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки истории: {e}")
            return []
    
    def _backtest_on_candles(self, strategy_params: Dict, period_days: int = 30) -> Dict:
        """
        Бэктест стратегии на основе свечей (когда нет истории сделок)
        
        Args:
            strategy_params: Параметры стратегии
            period_days: Период для бэктеста в днях
        
        Returns:
            Результаты бэктеста
        """
        logger.info("📊 Бэктест на основе свечей...")
        
        try:
            market_data = self._load_market_data()
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для бэктеста")
                return {'error': 'No candles available for backtesting'}
            
            base_config = self.auto_bot_config or {}
            rsi_period = int(base_config.get('rsi_period', 14) or 14)
            initial_balance = 10000.0
            balance = initial_balance
            closed_trades: List[Dict[str, Any]] = []
            total_positions_opened = 0
            
            def close_position(position: Optional[Dict[str, Any]], exit_price: float, exit_time: Any, reason: str):
                nonlocal balance
                if not position or exit_price <= 0:
                    return None
                entry_price = position['entry_price']
                direction = position['direction']
                size = position['size']
                if direction == 'LONG':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                pnl_usdt = size * (pnl_pct / 100)
                balance += size + pnl_usdt
                closed_trades.append({
                    'symbol': position['symbol'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl_usdt,
                    'pnl_pct': pnl_pct,
                    'exit_reason': reason,
                    'entry_time': position['entry_time'],
                    'exit_time': exit_time
                })
                return None
            
            processed_symbols = 0
            for symbol, candle_info in candles_data.items():
                candles = candle_info.get('candles', [])
                if len(candles) < rsi_period + 5:
                    continue
                
                symbol_config = _get_config_snapshot(symbol).get('merged', base_config)
                position_size_pct = strategy_params.get('position_size_pct')
                if position_size_pct is None:
                    if symbol_config.get('default_position_mode') == 'percent':
                        position_size_pct = symbol_config.get('default_position_size', 10.0)
                    else:
                        position_size_pct = 10.0
                
                closes = [float(c.get('close', 0) or 0) for c in candles]
                times = [c.get('time') for c in candles]
                if len(closes) <= rsi_period + 1 or any(price <= 0 for price in closes):
                    continue
                
                rsi_history = calculate_rsi_history(closes, period=rsi_period)
                if not rsi_history:
                    continue
                
                position = None
                for i in range(rsi_period, len(closes)):
                    rsi_index = i - rsi_period
                    if rsi_index >= len(rsi_history):
                        break
                    
                    current_price = closes[i]
                    current_time = times[i]
                    # Безопасное получение RSI с проверкой на None и нечисловые значения
                    current_rsi_raw = rsi_history[rsi_index]
                    try:
                        current_rsi = float(current_rsi_raw) if current_rsi_raw is not None else 50.0
                        if not isinstance(current_rsi, (int, float)) or current_rsi < 0 or current_rsi > 100:
                            current_rsi = 50.0  # Значение по умолчанию если вне диапазона
                    except (TypeError, ValueError):
                        current_rsi = 50.0  # Значение по умолчанию при ошибке преобразования
                    trend_window = int(symbol_config.get('trend_analysis_period', 30) or 30)
                    trend = _determine_trend(closes, i, trend_window)
                    
                    if position:
                        decision = evaluate_protections(
                            current_price=current_price,
                            config=symbol_config,
                            state=position['protection_state'],
                            realized_pnl=0.0,
                            now_ts=_normalize_timestamp(current_time)
                        )
                        position['protection_state'] = decision.state
                        if decision.should_close and decision.reason:
                            position = close_position(position, current_price, current_time, decision.reason)
                            continue
                        
                        if position:
                            if position['direction'] == 'LONG':
                                if position['entry_trend'] == 'UP':
                                    rsi_exit_raw = symbol_config.get('rsi_exit_long_with_trend', base_config.get('rsi_exit_long_with_trend', 65))
                                else:
                                    rsi_exit_raw = symbol_config.get('rsi_exit_long_against_trend', base_config.get('rsi_exit_long_against_trend', 60))
                                try:
                                    rsi_exit = float(rsi_exit_raw) if rsi_exit_raw is not None else 65.0
                                    if not isinstance(rsi_exit, (int, float)) or rsi_exit < 0 or rsi_exit > 100:
                                        rsi_exit = 65.0
                                except (TypeError, ValueError):
                                    rsi_exit = 65.0
                                if current_rsi >= rsi_exit:
                                    position = close_position(position, current_price, current_time, 'RSI_EXIT')
                                    continue
                            else:
                                if position['entry_trend'] == 'DOWN':
                                    rsi_exit_raw = symbol_config.get('rsi_exit_short_with_trend', base_config.get('rsi_exit_short_with_trend', 35))
                                else:
                                    rsi_exit_raw = symbol_config.get('rsi_exit_short_against_trend', base_config.get('rsi_exit_short_against_trend', 40))
                                try:
                                    rsi_exit = float(rsi_exit_raw) if rsi_exit_raw is not None else 35.0
                                    if not isinstance(rsi_exit, (int, float)) or rsi_exit < 0 or rsi_exit > 100:
                                        rsi_exit = 35.0
                                except (TypeError, ValueError):
                                    rsi_exit = 35.0
                                if current_rsi <= rsi_exit:
                                    position = close_position(position, current_price, current_time, 'RSI_EXIT')
                                    continue
                    
                    if position:
                        continue
                    
                    # Безопасное получение пороговых значений RSI с проверкой на None
                    rsi_long_entry_raw = strategy_params.get('rsi_long_entry', symbol_config.get('rsi_long_threshold', 29))
                    rsi_short_entry_raw = strategy_params.get('rsi_short_entry', symbol_config.get('rsi_short_threshold', 71))
                    try:
                        rsi_long_entry = float(rsi_long_entry_raw) if rsi_long_entry_raw is not None else 29.0
                        if not isinstance(rsi_long_entry, (int, float)) or rsi_long_entry < 0 or rsi_long_entry > 100:
                            rsi_long_entry = 29.0
                    except (TypeError, ValueError):
                        rsi_long_entry = 29.0
                    try:
                        rsi_short_entry = float(rsi_short_entry_raw) if rsi_short_entry_raw is not None else 71.0
                        if not isinstance(rsi_short_entry, (int, float)) or rsi_short_entry < 0 or rsi_short_entry > 100:
                            rsi_short_entry = 71.0
                    except (TypeError, ValueError):
                        rsi_short_entry = 71.0
                    
                    should_enter_long = current_rsi <= rsi_long_entry
                    should_enter_short = current_rsi >= rsi_short_entry
                    
                    if not (should_enter_long or should_enter_short):
                        continue
                    
                    filters_allowed, filters_reason = apply_entry_filters(
                        symbol,
                        candles[:i + 1],
                        current_rsi,
                        'ENTER_LONG' if should_enter_long else 'ENTER_SHORT',
                        symbol_config,
                        trend=trend,
                    )
                    if not filters_allowed:
                        continue
                    
                    direction = 'LONG' if should_enter_long else 'SHORT'
                    position_size_usdt = balance * (position_size_pct / 100.0)
                    if position_size_usdt <= 0:
                        continue
                    
                    position = {
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'entry_rsi': current_rsi,
                        'entry_trend': trend,
                        'size': position_size_usdt,
                        'protection_state': _create_protection_state(direction, current_price, position_size_usdt, current_time)
                    }
                    balance -= position_size_usdt
                    total_positions_opened += 1
                
                if position:
                    position = close_position(position, closes[-1], times[-1], 'FORCED_EXIT_END')
                
                processed_symbols += 1
                if processed_symbols >= 10:
                    break
            
            if len(closed_trades) == 0:
                logger.warning("⚠️ Не удалось открыть позиции на основе текущих данных свечей")
                return {
                    'strategy_params': strategy_params,
                    'period_days': period_days,
                    'initial_balance': initial_balance,
                    'final_balance': initial_balance,
                    'total_return': 0.0,
                    'total_pnl': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Не удалось открыть позиции (нужна история сделок для полного анализа)'
                }
            
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]
            total_pnl = sum(t['pnl'] for t in closed_trades)
            win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0.0
            avg_win = float(np.mean([t['pnl'] for t in winning_trades])) if winning_trades else 0.0
            avg_loss = float(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0.0
            final_balance = balance
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            results = {
                'strategy_params': strategy_params,
                'period_days': period_days,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
                'timestamp': datetime.now().isoformat(),
                'note': 'Расширенный бэктест на свечах (Protection Engine)',
                'positions_opened': total_positions_opened,
                'closed_trades': closed_trades
            }
            
            logger.info(
                f"✅ Бэктест на свечах: {len(closed_trades)} сделок, "
                f"Return={total_return:.2f}%, WinRate={win_rate:.2f}%"
            )
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Ошибка бэктеста на свечах: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def backtest_strategy(self, strategy_params: Dict, period_days: int = 30) -> Dict:
        """
        Бэктест стратегии с заданными параметрами
        
        Args:
            strategy_params: Параметры стратегии (RSI вход/выход, стоп-лосс и т.д.)
            period_days: Период для бэктеста в днях
        
        Returns:
            Результаты бэктеста
        """
        strategy_name = strategy_params.get('name', 'Unknown')
        logger.info(f"📈 Бэктест стратегии '{strategy_name}' с параметрами: {strategy_params}")
        
        try:
            # Загружаем исторические данные (только сделки с RSI — иначе бэктест не сможет оценить вход/выход)
            trades = self._load_history_data(with_rsi_only=True)
            
            logger.info(f"📊 Загружено {len(trades)} сделок из истории (с RSI)")
            
            # Если нет сделок с RSI или мало, используем свечи для симуляции
            if len(trades) < 10:
                logger.info("⚠️ Недостаточно сделок для бэктеста, используем свечи для симуляции...")
                return self._backtest_on_candles(strategy_params, period_days)
            
            # Фильтруем сделки по периоду
            cutoff_date = datetime.now() - timedelta(days=period_days)
            filtered_trades = []
            
            for trade in trades:
                try:
                    trade_time = datetime.fromisoformat(trade.get('timestamp', '').replace('Z', ''))
                    if trade_time >= cutoff_date:
                        filtered_trades.append(trade)
                except:
                    continue
            
            logger.info(f"📊 Отфильтровано {len(filtered_trades)} сделок за последние {period_days} дней")
            
            if len(filtered_trades) < 10:
                logger.info("⚠️ Недостаточно сделок за период, используем свечи для симуляции...")
                return self._backtest_on_candles(strategy_params, period_days)
            
            base_config = _get_config_snapshot().get('global', {})

            # Симулируем торговлю с новыми параметрами
            initial_balance = 10000.0
            balance = initial_balance
            positions = []
            closed_trades = []
            entered_count = 0  # сколько раз открыли позицию по стратегии (для диагностики)

            # Параметры стратегии (с проверкой на None)
            rsi_long_entry_raw = strategy_params.get(
                'rsi_long_entry',
                base_config.get('rsi_long_threshold', 29)
            )
            try:
                rsi_long_entry = float(rsi_long_entry_raw) if rsi_long_entry_raw is not None else 29.0
                if not isinstance(rsi_long_entry, (int, float)) or rsi_long_entry < 0 or rsi_long_entry > 100:
                    rsi_long_entry = 29.0
            except (TypeError, ValueError):
                rsi_long_entry = 29.0
            
            rsi_long_exit_raw = strategy_params.get(
                'rsi_long_exit',
                base_config.get('rsi_exit_long_with_trend', 65)
            )
            try:
                rsi_long_exit = float(rsi_long_exit_raw) if rsi_long_exit_raw is not None else 65.0
                if not isinstance(rsi_long_exit, (int, float)) or rsi_long_exit < 0 or rsi_long_exit > 100:
                    rsi_long_exit = 65.0
            except (TypeError, ValueError):
                rsi_long_exit = 65.0
            
            rsi_short_entry_raw = strategy_params.get(
                'rsi_short_entry',
                base_config.get('rsi_short_threshold', 71)
            )
            try:
                rsi_short_entry = float(rsi_short_entry_raw) if rsi_short_entry_raw is not None else 71.0
                if not isinstance(rsi_short_entry, (int, float)) or rsi_short_entry < 0 or rsi_short_entry > 100:
                    rsi_short_entry = 71.0
            except (TypeError, ValueError):
                rsi_short_entry = 71.0
            
            rsi_short_exit_raw = strategy_params.get(
                'rsi_short_exit',
                base_config.get('rsi_exit_short_with_trend', 35)
            )
            try:
                rsi_short_exit = float(rsi_short_exit_raw) if rsi_short_exit_raw is not None else 35.0
                if not isinstance(rsi_short_exit, (int, float)) or rsi_short_exit < 0 or rsi_short_exit > 100:
                    rsi_short_exit = 35.0
            except (TypeError, ValueError):
                rsi_short_exit = 35.0
            stop_loss_pct = strategy_params.get(
                'stop_loss_pct',
                base_config.get('max_loss_percent', 2.0)
            )
            take_profit_pct = strategy_params.get(
                'take_profit_pct',
                base_config.get('take_profit_percent', 20.0)
            )
            position_size_pct = strategy_params.get('position_size_pct', 10.0)
            
            # Симулируем каждую сделку
            for trade in filtered_trades:
                entry_data = trade.get('entry_data', {})
                exit_market_data = trade.get('exit_market_data', {})
                # RSI может быть в entry_data.rsi или в полях сделки (entry_rsi / rsi)
                entry_rsi_raw = entry_data.get('rsi') or trade.get('entry_rsi') or trade.get('rsi')
                try:
                    entry_rsi = float(entry_rsi_raw) if entry_rsi_raw is not None else 50.0
                    if not isinstance(entry_rsi, (int, float)) or entry_rsi < 0 or entry_rsi > 100:
                        entry_rsi = 50.0  # Значение по умолчанию если вне диапазона
                except (TypeError, ValueError):
                    entry_rsi = 50.0  # Значение по умолчанию при ошибке преобразования
                
                exit_rsi_raw = (
                    (exit_market_data.get('rsi') if exit_market_data else None)
                    or trade.get('exit_rsi')
                )
                try:
                    exit_rsi = float(exit_rsi_raw) if exit_rsi_raw is not None else entry_rsi
                    if not isinstance(exit_rsi, (int, float)) or exit_rsi < 0 or exit_rsi > 100:
                        exit_rsi = entry_rsi
                except (TypeError, ValueError):
                    exit_rsi = entry_rsi
                
                direction = trade.get('direction', 'LONG')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                
                if entry_price == 0 or exit_price == 0:
                    continue
                
                # Проверяем условия входа по новой стратегии
                should_enter = False
                
                if direction == 'LONG':
                    should_enter = entry_rsi <= rsi_long_entry
                elif direction == 'SHORT':
                    should_enter = entry_rsi >= rsi_short_entry
                
                if not should_enter:
                    continue

                entered_count += 1
                # Открываем позицию
                position_size = balance * (position_size_pct / 100.0)
                position = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'size': position_size,
                    'entry_rsi': entry_rsi,
                    'entry_time': trade.get('timestamp'),
                    'protection_state': _create_protection_state(
                        direction,
                        entry_price,
                        position_size,
                        trade.get('timestamp')
                    )
                }
                positions.append(position)
                balance -= position_size
                
                # Проверяем условия выхода
                should_exit = False
                exit_reason = None
                protection_decision = evaluate_protections(
                    current_price=exit_price,
                    config=base_config,
                    state=position.get('protection_state'),
                    realized_pnl=0.0,
                    now_ts=_normalize_timestamp(
                        exit_market_data.get('time')
                        if exit_market_data
                        else trade.get('exit_time')
                    )
                )
                position['protection_state'] = protection_decision.state
                if protection_decision.should_close and protection_decision.reason:
                    should_exit = True
                    exit_reason = protection_decision.reason
                
                if not should_exit and direction == 'LONG':
                    if exit_price <= entry_price * (1 - stop_loss_pct / 100.0):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                    elif exit_price >= entry_price * (1 + take_profit_pct / 100.0):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                    elif exit_rsi >= rsi_long_exit:
                        should_exit = True
                        exit_reason = 'RSI_EXIT'
                
                elif not should_exit and direction == 'SHORT':
                    if exit_price >= entry_price * (1 + stop_loss_pct / 100.0):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                    elif exit_price <= entry_price * (1 - take_profit_pct / 100.0):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                    elif exit_rsi <= rsi_short_exit:
                        should_exit = True
                        exit_reason = 'RSI_EXIT'
                
                if should_exit:
                    # Закрываем позицию
                    if direction == 'LONG':
                        pnl = (exit_price - entry_price) / entry_price * position_size
                    else:
                        pnl = (entry_price - exit_price) / entry_price * position_size
                    
                    balance += position_size + pnl
                    
                    closed_trades.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'roi': (pnl / position_size) * 100
                    })
                    
                    positions.remove(position)
            
            # Рассчитываем статистику
            if len(closed_trades) == 0:
                # Возвращаем валидный результат без error, чтобы оптимизатор видел стратегии (0 сделок)
                logger.warning(
                    f"⚠️ По стратегии '{strategy_name}' не закрыто ни одной сделки "
                    f"(открыто по условиям входа: {entered_count} из {len(filtered_trades)}). "
                    "Возможные причины: в истории нет RSI (entry_data/entry_rsi), или условия выхода не сработали."
                )
                return {
                    'strategy_params': strategy_params,
                    'period_days': period_days,
                    'initial_balance': initial_balance,
                    'final_balance': initial_balance,
                    'total_return': 0.0,
                    'total_pnl': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'timestamp': datetime.now().isoformat(),
                }

            total_pnl = sum(t['pnl'] for t in closed_trades)
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            final_balance = balance + sum(p['size'] for p in positions)
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            results = {
                'strategy_params': strategy_params,
                'period_days': period_days,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Сохраняем результаты
            # Сохраняем в БД вместо файла
            try:
                from bot_engine.ai.ai_database import get_ai_database
                ai_db = get_ai_database()
                if ai_db:
                    backtest_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    ai_db.save_backtest_result(
                        results=results,
                        backtest_name=backtest_name,
                        symbol=symbol
                    )
                    pass
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить результаты бэктеста в БД: {e}")
            
            logger.info(f"✅ Бэктест завершен: Return={total_return:.2f}%, Win Rate={win_rate:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка бэктеста: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def backtest_strategies(self, period_days: int = 30) -> List[Dict]:
        """
        Бэктест нескольких стратегий для сравнения
        
        Args:
            period_days: Период для бэктеста
        
        Returns:
            Список результатов бэктеста
        """
        logger.info(f"📈 Бэктест нескольких стратегий за {period_days} дней...")
        
        # Разные варианты параметров стратегии
        strategies = [
            {
                'name': 'Conservative',
                'rsi_long_entry': 25,
                'rsi_long_exit': 60,
                'rsi_short_entry': 75,
                'rsi_short_exit': 40,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 15.0,
                'position_size_pct': 5.0
            },
            {
                'name': 'Moderate',
                'rsi_long_entry': 29,
                'rsi_long_exit': 65,
                'rsi_short_entry': 71,
                'rsi_short_exit': 35,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 20.0,
                'position_size_pct': 10.0
            },
            {
                'name': 'Aggressive',
                'rsi_long_entry': 30,
                'rsi_long_exit': 70,
                'rsi_short_entry': 70,
                'rsi_short_exit': 30,
                'stop_loss_pct': 3.0,
                'take_profit_pct': 30.0,
                'position_size_pct': 15.0
            }
        ]
        
        results = []
        
        for strategy in strategies:
            try:
                result = self.backtest_strategy(strategy, period_days)
                if 'error' not in result:
                    result['strategy_name'] = strategy['name']
                    results.append(result)
            except Exception as e:
                logger.error(f"❌ Ошибка бэктеста стратегии {strategy['name']}: {e}")
        
        # Сортируем по доходности
        results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        logger.info(f"✅ Бэктест завершен: протестировано {len(results)} стратегий")
        
        return results
    
    def compare_with_current_strategy(self, period_days: int = 30) -> Dict:
        """
        Сравнение текущей стратегии с оптимизированными вариантами
        
        Args:
            period_days: Период для сравнения
        
        Returns:
            Словарь с результатами сравнения
        """
        logger.info("📊 Сравнение стратегий...")
        
        # Текущие параметры стратегии (из конфига)
        current_strategy = {
            'name': 'Current',
            'rsi_long_entry': 29,
            'rsi_long_exit': 65,
            'rsi_short_entry': 71,
            'rsi_short_exit': 35,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 20.0,
            'position_size_pct': 10.0
        }
        
        # Бэктест текущей стратегии
        current_result = self.backtest_strategy(current_strategy, period_days)
        
        # Бэктест оптимизированных стратегий
        optimized_results = self.backtest_strategies(period_days)
        
        comparison = {
            'current_strategy': current_result,
            'optimized_strategies': optimized_results,
            'best_strategy': optimized_results[0] if optimized_results else None,
            'improvement': None
        }
        
        if optimized_results and 'total_return' in current_result:
            best_return = optimized_results[0].get('total_return', 0)
            current_return = current_result.get('total_return', 0)
            improvement = best_return - current_return
            
            comparison['improvement'] = {
                'return_improvement': improvement,
                'return_improvement_pct': (improvement / abs(current_return)) * 100 if current_return != 0 else 0
            }
        
        return comparison

