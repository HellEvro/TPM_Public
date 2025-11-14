#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль бэктестинга стратегий

Тестирует торговые стратегии на исторических данных
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger('AI.Backtester')


class AIBacktester:
    """
    Класс для бэктестинга торговых стратегий
    """
    
    def __init__(self):
        """Инициализация бэктестера"""
        self.results_dir = 'data/ai/backtest_results'
        self.data_dir = 'data/ai'
        
        # Создаем директории
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("✅ AIBacktester инициализирован")
    
    def _load_market_data(self) -> Dict:
        """
        Загрузить рыночные данные
        
        ВАЖНО: Использует ТОЛЬКО полную историю свечей из data/ai/candles_full_history.json
        (загруженную через пагинацию по 2000 свечей для каждой монеты)
        
        НЕ использует candles_cache.json - только полная история для качественного бэктеста!
        """
        try:
            # ВАЖНО: Загружаем ТОЛЬКО из полной истории свечей (data/ai/candles_full_history.json)
            # НЕ используем market_data.json - свечи всегда из candles_full_history.json!
            # НЕ используем candles_cache.json - только полная история!
            # Если файла нет - возвращаем пустые данные (не fallback на кэш!)
            full_history_file = os.path.join('data', 'ai', 'candles_full_history.json')
            market_data = {'latest': {'candles': {}}}
            
            # Загружаем свечи напрямую из полной истории (ВСЕГДА)
            if not os.path.exists(full_history_file):
                logger.error("=" * 80)
                logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ НЕ НАЙДЕН!")
                logger.error("=" * 80)
                logger.error(f"   📁 Файл: {full_history_file}")
                logger.error("   💡 Файл должен быть загружен через load_full_candles_history()")
                logger.error("   ⏳ ДОЖДИТЕСЬ пока файл не будет создан и загружен")
                logger.error("   ❌ НЕ используем candles_cache.json - только полная история!")
                logger.error("   ⏸️ Бэктест будет пропущен до загрузки файла")
                logger.error("=" * 80)
                return market_data
            
            # Читаем ТОЛЬКО из полной истории свечей
            try:
                logger.info(f"📖 Загрузка полной истории свечей из {full_history_file}...")
                logger.info("   💡 Это файл загружен через пагинацию по 2000 свечей для каждой монеты")
                logger.info("   ✅ Используем ТОЛЬКО полную историю (не используем candles_cache.json)")
                
                with open(full_history_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                # Извлекаем свечи из структуры с метаданными
                candles_data = {}
                if 'candles' in full_data:
                    candles_data = full_data['candles']
                elif isinstance(full_data, dict) and not full_data.get('metadata'):
                    candles_data = full_data
                else:
                    logger.warning("⚠️ Неожиданная структура файла candles_full_history.json")
                    candles_data = {}
                
                if candles_data:
                    logger.info(f"✅ Загружено полной истории для {len(candles_data)} монет")
                    
                    if 'latest' not in market_data:
                        market_data['latest'] = {}
                    if 'candles' not in market_data['latest']:
                        market_data['latest']['candles'] = {}
                    
                    for symbol, candle_info in candles_data.items():
                        candles = candle_info.get('candles', []) if isinstance(candle_info, dict) else []
                        if candles:
                            market_data['latest']['candles'][symbol] = {
                                'candles': candles,
                                'timeframe': candle_info.get('timeframe', '6h') if isinstance(candle_info, dict) else '6h',
                                'last_update': candle_info.get('last_update') or candle_info.get('loaded_at') if isinstance(candle_info, dict) else None
                            }
                else:
                    logger.error("=" * 80)
                    logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ ПУСТ ИЛИ ПОВРЕЖДЕН!")
                    logger.error("=" * 80)
                    logger.error(f"   📁 Файл: {full_history_file}")
                    logger.error("   ⏳ Дождитесь перезагрузки файла через load_full_candles_history()")
                    logger.error("   ⏸️ Бэктест будет пропущен до загрузки файла")
                    logger.error("=" * 80)
                    
            except json.JSONDecodeError as json_error:
                logger.error("=" * 80)
                logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ ПОВРЕЖДЕН!")
                logger.error("=" * 80)
                logger.error(f"   📁 Файл: {full_history_file}")
                logger.error(f"   ⚠️ JSON ошибка на позиции {json_error.pos}")
                logger.error("   🗑️ Удаляем поврежденный файл, он будет пересоздан при следующей загрузке")
                try:
                    os.remove(full_history_file)
                    logger.info("   ✅ Поврежденный файл удален")
                except Exception as del_error:
                    logger.debug(f"   ⚠️ Не удалось удалить файл: {del_error}")
                logger.error("   ⏳ Дождитесь перезагрузки файла через load_full_candles_history()")
                logger.error("   ⏸️ Бэктест будет пропущен до загрузки файла")
                logger.error("=" * 80)
            except Exception as e:
                logger.error(f"❌ Ошибка чтения candles_full_history.json: {e}")
                logger.error("   ⏸️ Бэктест будет пропущен до загрузки файла")
            
            # 2. Получаем индикаторы через API
            try:
                import requests
                response = requests.get('http://127.0.0.1:5001/api/bots/coins-with-rsi', timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        coins_data = data.get('coins', {})
                        logger.info(f"✅ Загружено индикаторов для {len(coins_data)} монет через API")
                        
                        for symbol, coin_data in coins_data.items():
                            market_data['latest']['indicators'][symbol] = {
                                'rsi': coin_data.get('rsi6h'),
                                'trend': coin_data.get('trend6h'),
                                'price': coin_data.get('price'),
                                'signal': coin_data.get('signal'),
                                'volume': coin_data.get('volume')
                            }
            except Exception as api_error:
                logger.debug(f"⚠️ Не удалось получить индикаторы через API: {api_error}")
            
            return market_data
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки рыночных данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _load_history_data(self) -> List[Dict]:
        """Загрузить историю трейдов"""
        try:
            history_file = os.path.join(self.data_dir, 'history_data.json')
            if not os.path.exists(history_file):
                logger.debug("📊 Файл истории не найден, пробуем получить через API...")
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
            
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
            
            logger.debug(f"📊 Загружено {len(unique_trades)} уникальных сделок из истории")
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
            # Загружаем рыночные данные (свечи)
            market_data = self._load_market_data()
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для бэктеста")
                return {'error': 'No candles available for backtesting'}
            
            # Параметры стратегии
            rsi_long_entry = strategy_params.get('rsi_long_entry', 29)
            rsi_long_exit = strategy_params.get('rsi_long_exit', 65)
            rsi_short_entry = strategy_params.get('rsi_short_entry', 71)
            rsi_short_exit = strategy_params.get('rsi_short_exit', 35)
            stop_loss_pct = strategy_params.get('stop_loss_pct', 2.0)
            take_profit_pct = strategy_params.get('take_profit_pct', 20.0)
            position_size_pct = strategy_params.get('position_size_pct', 10.0)
            
            # Симулируем торговлю на свечах
            initial_balance = 10000.0
            balance = initial_balance
            positions = []
            closed_trades = []
            
            # Обрабатываем свечи каждой монеты
            processed_symbols = 0
            for symbol, candle_info in candles_data.items():
                candles = candle_info.get('candles', [])
                if len(candles) < 50:  # Нужно минимум свечей для анализа
                    continue
                
                indicators = latest.get('indicators', {}).get(symbol, {})
                current_rsi = indicators.get('rsi', 50)
                
                # Простая симуляция: проверяем условия входа/выхода на основе RSI
                # В реальном бэктесте нужно анализировать каждую свечу последовательно
                
                # Проверяем условия входа
                should_enter_long = current_rsi <= rsi_long_entry
                should_enter_short = current_rsi >= rsi_short_entry
                
                if should_enter_long or should_enter_short:
                    # Получаем текущую цену из последней свечи
                    if candles:
                        current_price = candles[-1].get('close', 0)
                        if current_price > 0:
                            direction = 'LONG' if should_enter_long else 'SHORT'
                            position_size = balance * (position_size_pct / 100.0)
                            
                            position = {
                                'symbol': symbol,
                                'direction': direction,
                                'entry_price': current_price,
                                'size': position_size,
                                'entry_rsi': current_rsi,
                                'entry_time': candles[-1].get('time')
                            }
                            positions.append(position)
                            balance -= position_size
                
                processed_symbols += 1
                if processed_symbols >= 10:  # Ограничиваем количество монет для теста
                    break
            
            # Упрощенная симуляция выхода (в реальности нужно отслеживать каждую позицию)
            # Для демонстрации просто возвращаем базовые результаты
            
            if len(positions) == 0:
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
            
            # Базовые результаты
            final_balance = balance + sum(p['size'] for p in positions)
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            results = {
                'strategy_params': strategy_params,
                'period_days': period_days,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'total_pnl': final_balance - initial_balance,
                'total_trades': len(positions),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'timestamp': datetime.now().isoformat(),
                'note': 'Упрощенный бэктест на свечах (нужна история сделок для полного анализа)',
                'positions_opened': len(positions)
            }
            
            logger.info(f"✅ Бэктест на свечах завершен: открыто {len(positions)} позиций, баланс: {final_balance:.2f}")
            
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
            # Загружаем исторические данные
            trades = self._load_history_data()
            
            logger.info(f"📊 Загружено {len(trades)} сделок из истории")
            
            # Если нет сделок, используем свечи для симуляции
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
            
            # Симулируем торговлю с новыми параметрами
            initial_balance = 10000.0
            balance = initial_balance
            positions = []
            closed_trades = []
            
            # Параметры стратегии
            rsi_long_entry = strategy_params.get('rsi_long_entry', 29)
            rsi_long_exit = strategy_params.get('rsi_long_exit', 65)
            rsi_short_entry = strategy_params.get('rsi_short_entry', 71)
            rsi_short_exit = strategy_params.get('rsi_short_exit', 35)
            stop_loss_pct = strategy_params.get('stop_loss_pct', 2.0)
            take_profit_pct = strategy_params.get('take_profit_pct', 20.0)
            position_size_pct = strategy_params.get('position_size_pct', 10.0)
            
            # Симулируем каждую сделку
            for trade in filtered_trades:
                entry_data = trade.get('entry_data', {})
                exit_market_data = trade.get('exit_market_data', {})
                
                entry_rsi = entry_data.get('rsi', 50)
                exit_rsi = exit_market_data.get('rsi', 50) if exit_market_data else entry_rsi
                
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
                
                # Открываем позицию
                position_size = balance * (position_size_pct / 100.0)
                position = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'size': position_size,
                    'entry_rsi': entry_rsi,
                    'entry_time': trade.get('timestamp')
                }
                positions.append(position)
                balance -= position_size
                
                # Проверяем условия выхода
                should_exit = False
                exit_reason = None
                
                if direction == 'LONG':
                    # Проверяем стоп-лосс
                    if exit_price <= entry_price * (1 - stop_loss_pct / 100.0):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                    # Проверяем тейк-профит
                    elif exit_price >= entry_price * (1 + take_profit_pct / 100.0):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                    # Проверяем RSI выход
                    elif exit_rsi >= rsi_long_exit:
                        should_exit = True
                        exit_reason = 'RSI_EXIT'
                
                elif direction == 'SHORT':
                    # Проверяем стоп-лосс
                    if exit_price >= entry_price * (1 + stop_loss_pct / 100.0):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                    # Проверяем тейк-профит
                    elif exit_price <= entry_price * (1 - take_profit_pct / 100.0):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                    # Проверяем RSI выход
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
                return {'error': 'No trades executed'}
            
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
            results_file = os.path.join(
                self.results_dir,
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
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

