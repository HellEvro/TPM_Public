#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль оптимизации торговых стратегий

Анализирует результаты торговли и оптимизирует параметры стратегии
"""

import os
import json
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger('AI.StrategyOptimizer')


DEFAULT_PARAMETER_GENOMES: Dict[str, Dict[str, Any]] = {
    'rsi_long_threshold': {'min': 20, 'max': 35, 'step': 1, 'type': 'int'},
    'rsi_short_threshold': {'min': 65, 'max': 80, 'step': 1, 'type': 'int'},
    'rsi_exit_long_with_trend': {'min': 55, 'max': 75, 'step': 2, 'type': 'int'},
    'rsi_exit_short_with_trend': {'min': 25, 'max': 45, 'step': 2, 'type': 'int'},
    'max_loss_percent': {'min': 8.0, 'max': 25.0, 'step': 1.0, 'precision': 1},
    'take_profit_percent': {'min': 10.0, 'max': 40.0, 'step': 2.5, 'precision': 1},
    'trailing_stop_activation': {'min': 10.0, 'max': 70.0, 'step': 5.0, 'precision': 1},
    'trailing_stop_distance': {'min': 5.0, 'max': 40.0, 'step': 2.5, 'precision': 1},
    'trailing_take_distance': {'min': 0.2, 'max': 2.0, 'step': 0.1, 'precision': 2},
    'trailing_update_interval': {'min': 1.0, 'max': 8.0, 'step': 0.5, 'precision': 1},
    'break_even_trigger': {'min': 30.0, 'max': 250.0, 'step': 10.0, 'precision': 1},
    'max_position_hours': {'min': 12, 'max': 336, 'step': 12, 'type': 'int'},
}

DEFAULT_MAX_TESTS = 200


class AIStrategyOptimizer:
    """
    Класс для оптимизации торговых стратегий
    """
    
    def __init__(self):
        """Инициализация оптимизатора"""
        self.results_dir = 'data/ai/optimization_results'
        self.data_dir = 'data/ai'
        
        # Создаем директории
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.parameter_genomes, self.parameter_genomes_meta = self._load_parameter_genomes()
        self.max_genome_tests = int(self.parameter_genomes_meta.get('max_tests', DEFAULT_MAX_TESTS))
        
        logger.info("✅ AIStrategyOptimizer инициализирован")

    def _load_parameter_genomes(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Загружает пользовательские геномы параметров, объединяя с дефолтами."""
        path = os.path.join(self.data_dir, 'optimizer_genomes.json')
        merged = deepcopy(DEFAULT_PARAMETER_GENOMES)
        meta: Dict[str, Any] = {'version': 'default', 'source': 'defaults', 'max_tests': DEFAULT_MAX_TESTS}

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as fp:
                    payload = json.load(fp)
                file_params = payload.get('parameters') if isinstance(payload, dict) else payload
                if isinstance(file_params, dict):
                    for name, overrides in file_params.items():
                        if isinstance(overrides, dict):
                            merged[name] = {**merged.get(name, {}), **overrides}
                meta.update({k: v for k, v in payload.items() if k != 'parameters'})
                meta['source'] = os.path.relpath(path)
            except Exception as genome_error:
                logger.warning(f"⚠️ Не удалось загрузить optimizer_genomes.json: {genome_error}")

        return merged, meta

    def _build_range_from_genome(self, parameter_name: str) -> List[float]:
        """Создает диапазон значений на основе описания генома."""
        genome = self.parameter_genomes.get(parameter_name, {})

        if 'values' in genome and genome['values']:
            return list(genome['values'])

        min_value = genome.get('min')
        max_value = genome.get('max')
        step = genome.get('step')

        if min_value is None or max_value is None or step is None:
            raise ValueError(f"В геноме {parameter_name} отсутствуют min/max/step")

        values = list(np.arange(min_value, max_value + step * 0.5, step))
        precision = genome.get('precision')
        value_type = genome.get('type', 'float')

        if precision is not None:
            values = [round(v, precision) for v in values]

        if value_type == 'int':
            values = [int(round(v)) for v in values]

        return values

    def _log_parameter_ranges(self, ranges: Dict[str, List[Any]]):
        logger.info("   🧬 Конфигурация геномов оптимизатора:")
        logger.info(
            f"      версия: {self.parameter_genomes_meta.get('version', 'default')}, "
            f"источник: {self.parameter_genomes_meta.get('source', 'defaults')}, "
            f"max_tests: {self.max_genome_tests}"
        )
        for key, values in ranges.items():
            preview = values
            if len(values) > 10:
                preview = [values[0], values[1], '...', values[-2], values[-1]]
            logger.info(f"      {key}: {preview} (всего {len(values)})")

    def _log_param_changes(self, symbol: str, new_params: Dict[str, Any]):
        try:
            from bots_modules.imports_and_globals import get_individual_coin_settings  # noqa: WPS433,E402
            previous = get_individual_coin_settings(symbol) or {}
        except Exception:
            previous = {}

        changes = []
        for key, value in new_params.items():
            prev_value = previous.get(key)
            if prev_value != value:
                changes.append((key, prev_value, value))

        if not changes:
            logger.info(f"      📄 Изменения параметров для {symbol}: отсутствуют (значения совпадают)")
            return

        logger.info(f"      📄 Изменения параметров для {symbol}:")
        for key, prev_value, next_value in changes:
            logger.info(f"         - {key}: {prev_value} → {next_value}")
    
    def _load_history_data(self) -> List[Dict]:
        """Загрузить историю трейдов"""
        trades = []
        
        # 1. Пробуем загрузить из data/ai/history_data.json (данные собранные через API)
        try:
            history_file = os.path.join(self.data_dir, 'history_data.json')
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                latest = data.get('latest', {})
                history = data.get('history', [])
                
                if latest:
                    trades.extend(latest.get('trades', []))
                
                for entry in history:
                    trades.extend(entry.get('trades', []))
                
                logger.debug(f"📊 Загружено {len(trades)} сделок из history_data.json")
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки history_data.json: {e}")
        
        # 2. Пробуем загрузить напрямую из data/bot_history.json (основной файл bots.py)
        try:
            bot_history_file = os.path.join('data', 'bot_history.json')
            if os.path.exists(bot_history_file):
                with open(bot_history_file, 'r', encoding='utf-8') as f:
                    bot_history_data = json.load(f)
                
                # Извлекаем сделки из bot_history.json
                bot_trades = bot_history_data.get('trades', [])
                if bot_trades:
                    # Добавляем только новые сделки (избегаем дубликатов)
                    existing_ids = {t.get('id') for t in trades if t.get('id')}
                    for trade in bot_trades:
                        trade_id = trade.get('id') or trade.get('timestamp')
                        if trade_id not in existing_ids:
                            trades.append(trade)
                    
                    logger.debug(f"📊 Добавлено {len(bot_trades)} сделок из bot_history.json")
        except json.JSONDecodeError as json_error:
            logger.warning(f"⚠️ Файл bot_history.json поврежден (JSON ошибка на позиции {json_error.pos})")
            logger.info("🗑️ Удаляем поврежденный файл, bots.py пересоздаст его автоматически")
            try:
                bot_history_file = os.path.join('data', 'bot_history.json')
                os.remove(bot_history_file)
                logger.info("✅ Поврежденный файл удален")
            except Exception as del_error:
                logger.debug(f"⚠️ Не удалось удалить файл: {del_error}")
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки bot_history.json: {e}")
        
        # 3. Фильтруем только закрытые сделки с PnL
        closed_trades = [
            t for t in trades
            if t.get('status') == 'CLOSED' and t.get('pnl') is not None
        ]
        
        if len(closed_trades) > 0:
            logger.info(f"✅ Загружено {len(closed_trades)} закрытых сделок для анализа (всего {len(trades)} сделок)")
        
        return closed_trades
    
    def analyze_trade_patterns(self) -> Dict:
        """
        Анализ паттернов торговли
        
        Определяет какие условия приводят к прибыльным сделкам
        """
        logger.info("=" * 80)
        logger.info("🔍 АНАЛИЗ ПАТТЕРНОВ ТОРГОВЛИ")
        logger.info("=" * 80)
        
        try:
            trades = self._load_history_data()
            
            logger.info(f"📊 Загружено {len(trades)} сделок для анализа")
            
            if len(trades) < 10:
                logger.warning("⚠️ Недостаточно данных для анализа (нужно минимум 10 сделок)")
                logger.info("💡 Возвращаем базовые паттерны...")
                return {
                    'total_trades': len(trades),
                    'profitable_trades': len([t for t in trades if t.get('pnl', 0) > 0]),
                    'losing_trades': len([t for t in trades if t.get('pnl', 0) <= 0]),
                    'win_rate': 0,
                    'rsi_analysis': {},
                    'trend_analysis': {},
                    'time_analysis': {},
                    'note': 'Недостаточно данных для полного анализа'
                }
            
            # Анализируем прибыльные и убыточные сделки
            profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            patterns = {
                'total_trades': len(trades),
                'profitable_trades': len(profitable_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(profitable_trades) / len(trades) * 100 if trades else 0,
                'rsi_analysis': {},
                'trend_analysis': {},
                'time_analysis': {}
            }
            
            # Анализ по RSI
            profitable_rsi = []
            losing_rsi = []
            
            for trade in profitable_trades:
                entry_data = trade.get('entry_data', {})
                rsi = entry_data.get('rsi')
                if rsi:
                    profitable_rsi.append(rsi)
            
            for trade in losing_trades:
                entry_data = trade.get('entry_data', {})
                rsi = entry_data.get('rsi')
                if rsi:
                    losing_rsi.append(rsi)
            
            if profitable_rsi:
                patterns['rsi_analysis']['profitable_avg'] = np.mean(profitable_rsi)
                patterns['rsi_analysis']['profitable_min'] = np.min(profitable_rsi)
                patterns['rsi_analysis']['profitable_max'] = np.max(profitable_rsi)
            
            if losing_rsi:
                patterns['rsi_analysis']['losing_avg'] = np.mean(losing_rsi)
                patterns['rsi_analysis']['losing_min'] = np.min(losing_rsi)
                patterns['rsi_analysis']['losing_max'] = np.max(losing_rsi)
            
            # Анализ по тренду
            trend_stats = {}
            
            for trade in trades:
                entry_data = trade.get('entry_data', {})
                trend = entry_data.get('trend', 'NEUTRAL')
                pnl = trade.get('pnl', 0)
                
                if trend not in trend_stats:
                    trend_stats[trend] = {'trades': 0, 'profitable': 0, 'total_pnl': 0}
                
                trend_stats[trend]['trades'] += 1
                if pnl > 0:
                    trend_stats[trend]['profitable'] += 1
                trend_stats[trend]['total_pnl'] += pnl
            
            patterns['trend_analysis'] = trend_stats
            
            # Сохраняем результаты анализа
            analysis_file = os.path.join(self.results_dir, 'trade_patterns.json')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(patterns, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Анализ завершен: Win Rate={patterns['win_rate']:.2f}%")
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов: {e}")
            return {}
    
    def optimize_strategy(self) -> Dict:
        """
        Оптимизация параметров стратегии
        
        Returns:
            Оптимизированные параметры стратегии
        """
        logger.info("⚙️ Оптимизация стратегии...")
        
        try:
            # Анализируем паттерны
            patterns = self.analyze_trade_patterns()
            
            if not patterns:
                logger.warning("⚠️ Недостаточно данных для оптимизации")
                return {}
            
            # Определяем оптимальные параметры на основе анализа
            optimized_params = {
                'rsi_long_entry': 29,  # По умолчанию
                'rsi_long_exit': 65,
                'rsi_short_entry': 71,
                'rsi_short_exit': 35,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 20.0
            }
            
            # Оптимизируем на основе RSI анализа
            rsi_analysis = patterns.get('rsi_analysis', {})
            
            if 'profitable_avg' in rsi_analysis:
                profitable_avg_rsi = rsi_analysis['profitable_avg']
                
                # Для LONG: если прибыльные сделки при низком RSI, используем его
                if profitable_avg_rsi < 30:
                    optimized_params['rsi_long_entry'] = max(20, int(profitable_avg_rsi - 5))
                    optimized_params['rsi_long_exit'] = min(70, int(profitable_avg_rsi + 35))
            
            if 'losing_avg' in rsi_analysis:
                losing_avg_rsi = rsi_analysis['losing_avg']
                
                # Избегаем параметров, которые приводят к убыткам
                if losing_avg_rsi < 30:
                    # Если убытки при низком RSI, повышаем порог входа
                    optimized_params['rsi_long_entry'] = max(optimized_params['rsi_long_entry'], 25)
            
            # Оптимизация на основе тренда
            trend_analysis = patterns.get('trend_analysis', {})
            
            if trend_analysis:
                # Определяем лучший тренд для торговли
                best_trend = None
                best_win_rate = 0
                
                for trend, stats in trend_analysis.items():
                    win_rate = stats['profitable'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_trend = trend
                
                optimized_params['best_trend'] = best_trend
                optimized_params['trend_win_rate'] = best_win_rate
            
            # Сохраняем оптимизированные параметры
            optimization_file = os.path.join(self.results_dir, 'optimized_params.json')
            with open(optimization_file, 'w', encoding='utf-8') as f:
                json.dump(optimized_params, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Оптимизация завершена: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации: {e}")
            return {}
    
    def optimize_bot_config(self, symbol: str) -> Dict:
        """
        Оптимизация конфигурации конкретного бота
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Оптимизированная конфигурация бота
        """
        logger.info(f"⚙️ Оптимизация конфигурации для {symbol}...")
        
        try:
            trades = self._load_history_data()
            
            # Фильтруем сделки по символу
            symbol_trades = [t for t in trades if t.get('symbol') == symbol]
            
            if len(symbol_trades) < 5:
                logger.warning(f"⚠️ Недостаточно данных для {symbol}")
                return {}
            
            # Анализируем сделки для этого символа
            profitable = [t for t in symbol_trades if t.get('pnl', 0) > 0]
            
            # Определяем оптимальные параметры для символа
            optimized_config = {
                'symbol': symbol,
                'rsi_long_entry': 29,
                'rsi_long_exit': 65,
                'rsi_short_entry': 71,
                'rsi_short_exit': 35
            }
            
            # Анализ RSI для этого символа
            profitable_rsi = []
            for trade in profitable:
                entry_data = trade.get('entry_data', {})
                rsi = entry_data.get('rsi')
                if rsi:
                    profitable_rsi.append(rsi)
            
            if profitable_rsi:
                avg_rsi = np.mean(profitable_rsi)
                optimized_config['rsi_long_entry'] = max(20, int(avg_rsi - 5))
                optimized_config['rsi_long_exit'] = min(70, int(avg_rsi + 35))
            
            logger.info(f"✅ Оптимизация для {symbol} завершена")
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации для {symbol}: {e}")
            return {}
    
    def optimize_coin_parameters_on_candles(
        self, 
        symbol: str, 
        candles: List[Dict],
        current_win_rate: float = 0.0
    ) -> Optional[Dict]:
        """
        ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ КОНКРЕТНОЙ МОНЕТЫ через Grid Search
        
        Тестирует разные комбинации параметров на исторических свечах
        и находит оптимальные для этой монеты.
        
        Args:
            symbol: Символ монеты
            candles: Список свечей для тестирования
            current_win_rate: Текущий win rate (если < 80%, запускаем оптимизацию)
        
        Returns:
            Оптимизированные параметры или None
        """
        logger.info("=" * 80)
        logger.info(f"🔍 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ {symbol}")
        logger.info("=" * 80)
        if current_win_rate >= 80.0:
            logger.info(f"   📊 Текущий Win Rate: {current_win_rate:.1f}% (>=80%, приемлемо, но оптимизируем дальше к 100%)")
        else:
            logger.info(f"   📊 Текущий Win Rate: {current_win_rate:.1f}% (<80%, оптимизируем, но НЕ сохраняем пока не достигнем >=80%)")
        logger.info(f"   📈 Свечей для тестирования: {len(candles)}")
        logger.info(f"   🧠 Используем базу знаний для улучшения торговой методики")
        
        # Загружаем базу знаний для использования опыта
        knowledge_base = {}
        successful_rsi_ranges = {}
        try:
            from bot_engine.ai.ai_continuous_learning import AIContinuousLearning
            continuous_learning = AIContinuousLearning()
            knowledge_base = continuous_learning.knowledge_base
            
            # Используем знания о успешных RSI диапазонах для приоритизации тестов
            successful_rsi_ranges = knowledge_base.get('successful_patterns', {}).get('rsi_ranges', {})
            if successful_rsi_ranges:
                best_rsi_range = max(successful_rsi_ranges.items(), key=lambda x: x[1])[0] if successful_rsi_ranges else None
                logger.info(f"   💡 База знаний: успешные входы в диапазоне RSI {best_rsi_range}")
        except Exception as kb_error:
            logger.debug(f"   ⚠️ Не удалось загрузить базу знаний: {kb_error}")
        
        try:
            # Загружаем базовые параметры из bot_config.py
            try:
                from bot_engine.bot_config import (
                    RSI_OVERSOLD, RSI_OVERBOUGHT,
                    RSI_EXIT_LONG_WITH_TREND, RSI_EXIT_LONG_AGAINST_TREND,
                    RSI_EXIT_SHORT_WITH_TREND, RSI_EXIT_SHORT_AGAINST_TREND,
                    DEFAULT_AUTO_BOT_CONFIG
                )
                base_rsi_long_entry = RSI_OVERSOLD
                base_rsi_short_entry = RSI_OVERBOUGHT
                base_rsi_long_exit_with = RSI_EXIT_LONG_WITH_TREND
                base_rsi_long_exit_against = RSI_EXIT_LONG_AGAINST_TREND
                base_rsi_short_exit_with = RSI_EXIT_SHORT_WITH_TREND
                base_rsi_short_exit_against = RSI_EXIT_SHORT_AGAINST_TREND
                base_stop_loss = DEFAULT_AUTO_BOT_CONFIG.get('max_loss_percent', 15)
                base_take_profit = DEFAULT_AUTO_BOT_CONFIG.get('take_profit_percent', 20)
            except ImportError:
                # Значения по умолчанию
                base_rsi_long_entry = 29
                base_rsi_short_entry = 71
                base_rsi_long_exit_with = 65
                base_rsi_long_exit_against = 60
                base_rsi_short_exit_with = 35
                base_rsi_short_exit_against = 40
                base_stop_loss = 15
                base_take_profit = 20
            
            parameter_ranges = {
                'rsi_long_entry': self._build_range_from_genome('rsi_long_threshold'),
                'rsi_short_entry': self._build_range_from_genome('rsi_short_threshold'),
                'rsi_long_exit': self._build_range_from_genome('rsi_exit_long_with_trend'),
                'rsi_short_exit': self._build_range_from_genome('rsi_exit_short_with_trend'),
                'stop_loss': self._build_range_from_genome('max_loss_percent'),
                'take_profit': self._build_range_from_genome('take_profit_percent'),
                'trailing_activation': self._build_range_from_genome('trailing_stop_activation'),
                'trailing_distance': self._build_range_from_genome('trailing_stop_distance'),
                'break_even_trigger': self._build_range_from_genome('break_even_trigger'),
                'trailing_take_distance': self._build_range_from_genome('trailing_take_distance'),
                'trailing_update_interval': self._build_range_from_genome('trailing_update_interval'),
            }

            self._log_parameter_ranges(parameter_ranges)

            rsi_long_entry_range = parameter_ranges['rsi_long_entry']
            rsi_short_entry_range = parameter_ranges['rsi_short_entry']
            rsi_long_exit_range = parameter_ranges['rsi_long_exit']
            rsi_short_exit_range = parameter_ranges['rsi_short_exit']
            stop_loss_range = parameter_ranges['stop_loss']
            take_profit_range = parameter_ranges['take_profit']
            trailing_activation_range = parameter_ranges['trailing_activation']
            trailing_distance_range = parameter_ranges['trailing_distance']
            break_even_trigger_range = parameter_ranges['break_even_trigger']
            trailing_take_distance_range = parameter_ranges['trailing_take_distance']
            trailing_update_interval_range = parameter_ranges['trailing_update_interval']
            
            total_combinations = (
                len(rsi_long_entry_range) * len(rsi_short_entry_range) *
                len(rsi_long_exit_range) * len(rsi_short_exit_range) *
                len(stop_loss_range) * len(take_profit_range) *
                len(trailing_activation_range) * len(trailing_distance_range) *
                len(break_even_trigger_range) * len(trailing_take_distance_range) *
                len(trailing_update_interval_range)
            )
            logger.info(f"   🔍 Тестируем до {total_combinations} комбинаций параметров (ограничение {self.max_genome_tests})")
            
            best_params = None
            best_win_rate = 0.0
            best_total_pnl = float('-inf')
            best_trades_count = 0
            
            tested_count = 0
            max_tests = self.max_genome_tests  # Настраиваемое значение через optimizer_genomes.json
            
            # Импортируем функцию расчета RSI
            try:
                from bot_engine.indicators import TechnicalIndicators
                calculate_rsi_history_func = TechnicalIndicators.calculate_rsi_history
            except ImportError:
                try:
                    from bots_modules.calculations import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
                except ImportError:
                    from bot_engine.utils.rsi_utils import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
            
            # Сортируем свечи по времени
            candles_sorted = sorted(candles, key=lambda x: x.get('time', 0))
            if len(candles_sorted) < 100:
                logger.warning(f"⚠️ Недостаточно свечей для оптимизации ({len(candles_sorted)})")
                return None
            
            # Вычисляем RSI один раз для всех свечей
            rsi_history = calculate_rsi_history_func(candles_sorted, period=14)
            if not rsi_history or len(rsi_history) < 50:
                logger.warning(f"⚠️ Недостаточно данных RSI для оптимизации")
                return None
            
            closes = [float(c.get('close', 0) or 0) for c in candles_sorted]
            
            # Тестируем комбинации (умный выбор для производительности)
            # Приоритет: сначала тестируем базовые значения, потом вариации
            for rsi_long_entry in rsi_long_entry_range[:4]:  # Берем больше значений
                for rsi_short_entry in rsi_short_entry_range[:4]:
                    for rsi_long_exit in rsi_long_exit_range[:3]:
                        for rsi_short_exit in rsi_short_exit_range[:3]:
                            for stop_loss in stop_loss_range[:4]:
                                for take_profit in take_profit_range[:4]:
                                    for trailing_activation in trailing_activation_range[:3]:
                                        for trailing_distance in trailing_distance_range[:3]:
                                            for break_even_trigger in break_even_trigger_range[:3]:
                                                for trailing_take_distance in trailing_take_distance_range[:2]:
                                                    for trailing_update_interval in trailing_update_interval_range[:2]:
                                                        if tested_count >= max_tests:
                                                            break
                                                        
                                                        tested_count += 1
                                                        
                                                        # Симулируем торговлю с этими параметрами
                                                        simulated_trades = []
                                                        current_position = None
                                                        max_profit_achieved = {}  # Для каждой позиции отслеживаем максимальную прибыль
                                                        trailing_active = {}  # Для каждой позиции отслеживаем активацию трейлинга
                                                        break_even_activated = {}  # Для каждой позиции отслеживаем безубыток
                                        
                                                        for i in range(14, len(candles_sorted)):
                                                            try:
                                                                rsi_idx = i - 14
                                                                if rsi_idx >= len(rsi_history):
                                                                    continue
                                                                
                                                                current_rsi = rsi_history[rsi_idx]
                                                                current_price = closes[i]
                                                                
                                                                # Определяем тренд
                                                                trend = 'NEUTRAL'
                                                                if i >= 50:
                                                                    ema_short = self._calculate_ema(closes[max(0, i-50):i+1], 50)
                                                                    ema_long = self._calculate_ema(closes[max(0, i-200):i+1], 200)
                                                                    if ema_short and ema_long:
                                                                        if ema_short > ema_long:
                                                                            trend = 'UP'
                                                                        elif ema_short < ema_long:
                                                                            trend = 'DOWN'
                                                                
                                                                # Проверка выхода с учетом всех защитных механизмов
                                                                if current_position:
                                                                    direction = current_position['direction']
                                                                    entry_price = current_position['entry_price']
                                                                    position_id = current_position.get('id', id(current_position))
                                                                    
                                                                    # Вычисляем текущую прибыль
                                                                    if direction == 'LONG':
                                                                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                                                                    else:  # SHORT
                                                                        profit_pct = ((entry_price - current_price) / entry_price) * 100
                                                                    
                                                                    # Обновляем максимальную прибыль
                                                                    if position_id not in max_profit_achieved:
                                                                        max_profit_achieved[position_id] = profit_pct
                                                                    else:
                                                                        max_profit_achieved[position_id] = max(max_profit_achieved[position_id], profit_pct)
                                                                    
                                                                    # Проверка Break Even
                                                                    if position_id not in break_even_activated:
                                                                        break_even_activated[position_id] = False
                                                                    
                                                                    if not break_even_activated[position_id] and profit_pct >= break_even_trigger:
                                                                        break_even_activated[position_id] = True
                                                                    
                                                                    # Если безубыток активирован и прибыль упала до 0 или ниже - закрываем
                                                                    if break_even_activated[position_id] and profit_pct <= 0:
                                                                        simulated_trades.append({
                                                                            'direction': direction,
                                                                            'entry_price': entry_price,
                                                                            'exit_price': current_price,
                                                                            'pnl_pct': profit_pct,
                                                                            'is_successful': profit_pct > 0,
                                                                            'exit_reason': 'BREAK_EVEN'
                                                                        })
                                                                        current_position = None
                                                                        continue
                                                                    
                                                                    # Проверка Trailing Stop
                                                                    if position_id not in trailing_active:
                                                                        trailing_active[position_id] = False
                                                                    
                                                                    # Активация trailing stop
                                                                    if not trailing_active[position_id] and profit_pct >= trailing_activation:
                                                                        trailing_active[position_id] = True
                                                                    
                                                                    # Если trailing stop активен, проверяем расстояние
                                                                    if trailing_active[position_id]:
                                                                        max_profit = max_profit_achieved[position_id]
                                                                        # Trailing stop срабатывает если цена откатилась на trailing_distance от максимума
                                                                        if direction == 'LONG':
                                                                            trailing_stop_price = entry_price * (1 + (max_profit - trailing_distance) / 100)
                                                                            if current_price <= trailing_stop_price:
                                                                                simulated_trades.append({
                                                                                    'direction': direction,
                                                                                    'entry_price': entry_price,
                                                                                    'exit_price': current_price,
                                                                                    'pnl_pct': profit_pct,
                                                                                    'is_successful': profit_pct > 0,
                                                                                    'exit_reason': 'TRAILING_STOP',
                                                                                    'max_profit': max_profit
                                                                                })
                                                                                current_position = None
                                                                                continue
                                                                        else:  # SHORT
                                                                            trailing_stop_price = entry_price * (1 - (max_profit - trailing_distance) / 100)
                                                                            if current_price >= trailing_stop_price:
                                                                                simulated_trades.append({
                                                                                    'direction': direction,
                                                                                    'entry_price': entry_price,
                                                                                    'exit_price': current_price,
                                                                                    'pnl_pct': profit_pct,
                                                                                    'is_successful': profit_pct > 0,
                                                                                    'exit_reason': 'TRAILING_STOP',
                                                                                    'max_profit': max_profit
                                                                                })
                                                                                current_position = None
                                                                                continue
                                                                    
                                                                    # Стандартные проверки выхода
                                                                    should_exit = False
                                                                    exit_reason = None
                                                                    
                                                                    if direction == 'LONG':
                                                                        if current_rsi >= rsi_long_exit:
                                                                            should_exit = True
                                                                            exit_reason = 'RSI_EXIT'
                                                                        elif current_price <= entry_price * (1 - stop_loss / 100):
                                                                            should_exit = True
                                                                            exit_reason = 'STOP_LOSS'
                                                                        elif current_price >= entry_price * (1 + take_profit / 100):
                                                                            should_exit = True
                                                                            exit_reason = 'TAKE_PROFIT'
                                                                    else:  # SHORT
                                                                        if current_rsi <= rsi_short_exit:
                                                                            should_exit = True
                                                                            exit_reason = 'RSI_EXIT'
                                                                        elif current_price >= entry_price * (1 + stop_loss / 100):
                                                                            should_exit = True
                                                                            exit_reason = 'STOP_LOSS'
                                                                        elif current_price <= entry_price * (1 - take_profit / 100):
                                                                            should_exit = True
                                                                            exit_reason = 'TAKE_PROFIT'
                                                                    
                                                                    if should_exit:
                                                                        simulated_trades.append({
                                                                            'direction': direction,
                                                                            'entry_price': entry_price,
                                                                            'exit_price': current_price,
                                                                            'pnl_pct': profit_pct,
                                                                            'is_successful': profit_pct > 0,
                                                                            'exit_reason': exit_reason
                                                                        })
                                                                        current_position = None
                                                                        continue
                                                                
                                                                # Проверка входа
                                                                if not current_position:
                                                                    if current_rsi <= rsi_long_entry:
                                                                        position_id = len(simulated_trades) + 1
                                                                        current_position = {
                                                                            'id': position_id,
                                                                            'direction': 'LONG',
                                                                            'entry_price': current_price,
                                                                            'entry_rsi': current_rsi,
                                                                            'entry_idx': i
                                                                        }
                                                                        max_profit_achieved[position_id] = 0
                                                                        trailing_active[position_id] = False
                                                                        break_even_activated[position_id] = False
                                                                    elif current_rsi >= rsi_short_entry:
                                                                        position_id = len(simulated_trades) + 1
                                                                        current_position = {
                                                                            'id': position_id,
                                                                            'direction': 'SHORT',
                                                                            'entry_price': current_price,
                                                                            'entry_rsi': current_rsi,
                                                                            'entry_idx': i
                                                                        }
                                                                        max_profit_achieved[position_id] = 0
                                                                        trailing_active[position_id] = False
                                                                        break_even_activated[position_id] = False
                                                            except:
                                                                continue
                                                        
                                                        # Оцениваем результаты
                                                        if len(simulated_trades) >= 5:  # Минимум 5 сделок для оценки
                                                            successful = sum(1 for t in simulated_trades if t['is_successful'])
                                                            win_rate = successful / len(simulated_trades) * 100
                                                            total_pnl = sum(t['pnl_pct'] for t in simulated_trades)
                                                            
                                                            # Выбираем лучшую комбинацию (приоритет: win_rate > total_pnl)
                                                            if win_rate > best_win_rate or (win_rate == best_win_rate and total_pnl > best_total_pnl):
                                                                best_win_rate = win_rate
                                                                best_total_pnl = total_pnl
                                                                best_trades_count = len(simulated_trades)
                                                                best_params = {
                                                                    'rsi_long_threshold': rsi_long_entry,
                                                                    'rsi_short_threshold': rsi_short_entry,
                                                                    'rsi_exit_long_with_trend': rsi_long_exit,
                                                                    'rsi_exit_long_against_trend': rsi_long_exit - 5,  # Против тренда выходим раньше
                                                                    'rsi_exit_short_with_trend': rsi_short_exit,
                                                                    'rsi_exit_short_against_trend': rsi_short_exit + 5,
                                                                    'max_loss_percent': stop_loss,
                                                                    'take_profit_percent': take_profit,
                                                                    'trailing_stop_activation': trailing_activation,
                                                                    'trailing_stop_distance': trailing_distance,
                                                                    'trailing_take_distance': trailing_take_distance,
                                                                    'trailing_update_interval': trailing_update_interval,
                                                                    'break_even_trigger': break_even_trigger,
                                                                    'break_even_protection': True,  # Всегда включен
                                                                    'optimized_at': datetime.now().isoformat(),
                                                                    'optimization_win_rate': win_rate,
                                                                    'optimization_total_pnl': total_pnl,
                                                                    'optimization_trades_count': len(simulated_trades)
                                                                }
                                                                best_params['parameter_genome_version'] = self.parameter_genomes_meta.get('version', 'default')
                                                                
                                                                # Анализ причин выхода для улучшения стратегии
                                                                exit_reasons = {}
                                                                for trade in simulated_trades:
                                                                    reason = trade.get('exit_reason', 'UNKNOWN')
                                                                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                                                                best_params['exit_reasons_analysis'] = exit_reasons
                                                        
                                                        if tested_count >= max_tests:
                                                            break
                                                    if tested_count >= max_tests:
                                                        break
                                                if tested_count >= max_tests:
                                                    break
                                            if tested_count >= max_tests:
                                                break
                                        if tested_count >= max_tests:
                                            break
                                    if tested_count >= max_tests:
                                        break
                                if tested_count >= max_tests:
                                    break
                            if tested_count >= max_tests:
                                break
                        if tested_count >= max_tests:
                            break
                    if tested_count >= max_tests:
                        break
                if tested_count >= max_tests:
                    break
            
            if best_params and best_win_rate > current_win_rate:
                logger.info(f"   ✅ Найдены оптимальные параметры!")
                logger.info(f"      📊 Win Rate: {current_win_rate:.1f}% → {best_win_rate:.1f}% (+{best_win_rate - current_win_rate:.1f}%)")
                logger.info(f"      💰 Total PnL: {best_total_pnl:.2f}%")
                logger.info(f"      🎯 Сделок: {best_trades_count}")
                logger.info(f"      📈 Параметры: RSI LONG {best_params['rsi_long_threshold']}/{best_params['rsi_exit_long_with_trend']}, SHORT {best_params['rsi_short_threshold']}/{best_params['rsi_exit_short_with_trend']}")
                logger.info(f"      🛑 SL: {best_params['max_loss_percent']}%, TP: {best_params['take_profit_percent']}%")
                logger.info(f"      🚀 Trailing Stop: активация {best_params['trailing_stop_activation']}%, расстояние {best_params['trailing_stop_distance']}%")
                logger.info(f"      🎯 Trailing Take: расстояние {best_params['trailing_take_distance']}%, интервал {best_params['trailing_update_interval']}с")
                logger.info(f"      🛡️ Break Even: триггер {best_params['break_even_trigger']}%")
                
                # Анализ причин выхода
                exit_reasons = best_params.get('exit_reasons_analysis', {})
                if exit_reasons:
                    logger.info(f"      📊 Анализ выходов: {exit_reasons}")
                try:
                    formatted_params = json.dumps(
                        {k: v for k, v in best_params.items() if k != 'exit_reasons_analysis'},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str
                    )
                    logger.info(f"      🧾 Полный набор параметров: {formatted_params}")
                except Exception:
                    logger.debug("      🧾 Не удалось сериализовать параметры для логов")
                
                # ВАЖНО: Сохраняем индивидуальные настройки ТОЛЬКО если win rate >= 80%
                if best_win_rate >= 80.0:
                    logger.info(f"      🎯 Win Rate >= 80% - СОХРАНЯЕМ индивидуальные настройки для {symbol}")
                    logger.info(f"      💡 Эти параметры будут использоваться ботами вместо глобальных")
                    self._log_param_changes(symbol, best_params)
                    
                    # Сохраняем оптимальные параметры для монеты через API bots.py
                    try:
                        import requests
                        response = requests.post(
                            'http://localhost:5001/api/bots/individual-settings/' + symbol,
                            json=best_params,
                            timeout=5
                        )
                        if response.status_code == 200:
                            logger.info(f"   💾 Оптимизированные параметры сохранены для {symbol}")
                        else:
                            logger.warning(f"   ⚠️ Не удалось сохранить параметры через API: {response.status_code}")
                            # Пробуем напрямую через импорт
                            try:
                                from bots_modules.imports_and_globals import set_individual_coin_settings
                                set_individual_coin_settings(symbol, best_params, persist=True)
                                logger.info(f"   💾 Параметры сохранены напрямую для {symbol}")
                            except Exception as direct_error:
                                logger.error(f"   ❌ Ошибка прямого сохранения: {direct_error}")
                    except Exception as save_error:
                        logger.error(f"   ❌ Ошибка сохранения параметров: {save_error}")
                else:
                    logger.info(f"      ⚠️ Win Rate {best_win_rate:.1f}% < 80% - НЕ сохраняем индивидуальные настройки")
                    logger.info(f"      💡 Продолжаем использовать глобальные настройки (скрипты) пока AI модель не достигнет >=80%")
                    logger.info(f"      💡 Параметры найдены, но будут применены только когда win rate >= 80%")
                
                logger.info("=" * 80)
                return best_params
            else:
                logger.info(f"   ⚠️ Не найдено лучших параметров (текущий: {current_win_rate:.1f}%, лучший найденный: {best_win_rate:.1f}%)")
                logger.info("=" * 80)
                return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации параметров для {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Вычисляет EMA (Exponential Moving Average)"""
        if not prices or len(prices) < period:
            return None
        
        prices_array = np.array(prices[-period:])
        multiplier = 2.0 / (period + 1)
        
        ema = prices_array[0]
        for price in prices_array[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)

