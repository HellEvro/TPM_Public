#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль постоянного обучения и улучшения торговой методики

Анализирует результаты торговли и постоянно улучшает:
- Входы и выходы из сделок
- Работу со стоп-лоссами
- Работу с тейк-профитами
- Трейлинг-стопы и трейлинг-тейки
- Изучение рынка и паттернов
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger('AI.ContinuousLearning')


class AIContinuousLearning:
    """
    Класс для постоянного обучения и улучшения торговой методики
    """
    
    def __init__(self):
        """Инициализация модуля постоянного обучения"""
        self.data_dir = 'data/ai'
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
        
        # Загружаем базу знаний из БД
        self.knowledge_base = self._load_knowledge_base()
        
        logger.info("✅ AIContinuousLearning инициализирован")
    
    def _load_knowledge_base(self) -> Dict:
        """Загрузить базу знаний о торговле из БД"""
        try:
            if self.ai_db:
                result = self.ai_db.get_knowledge_base('trading_knowledge_base')
                if result and result.get('knowledge_data'):
                    return result['knowledge_data']
        except Exception as e:
            pass
        
        # База знаний по умолчанию
        default_kb = {
            'successful_patterns': {
                'rsi_ranges': {},
                'trend_conditions': {},
                'volatility_conditions': {},
                'time_conditions': {}
            },
            'failed_patterns': {
                'rsi_ranges': {},
                'trend_conditions': {},
                'volatility_conditions': {},
                'time_conditions': {}
            },
            'optimal_parameters': {},
            'market_insights': [],
            'improvement_history': [],
            'last_update': None
        }
        
        # Сохраняем дефолтную базу в БД
        try:
            if self.ai_db:
                self.ai_db.save_knowledge_base('trading_knowledge_base', default_kb)
        except:
            pass
        
        return default_kb
    
    def _save_knowledge_base(self):
        """Сохранить базу знаний в БД"""
        try:
            if not self.ai_db:
                logger.warning("⚠️ AI Database не доступна, база знаний не сохранена")
                return
            
            self.knowledge_base['last_update'] = datetime.now().isoformat()
            self.ai_db.save_knowledge_base('trading_knowledge_base', self.knowledge_base)
            pass
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения базы знаний в БД: {e}")
    
    def _should_train_on_symbol(self, symbol: str) -> bool:
        """
        Проверяет, должна ли монета использоваться для обучения AI на основе whitelist/blacklist.
        
        Логика:
        - Если scope == 'whitelist' ИЛИ (scope == 'all' и whitelist не пуст) -> обучаться только на монетах из whitelist
        - Если scope == 'blacklist' -> исключить монеты из blacklist (но если whitelist не пуст, то использовать whitelist)
        - Если scope == 'all' и whitelist пуст -> использовать все монеты кроме blacklist
        
        Args:
            symbol: Символ монеты для проверки
            
        Returns:
            True если монета должна использоваться для обучения, False иначе
        """
        if not symbol:
            return False
        
        symbol_upper = symbol.upper()
        
        try:
            # Пробуем получить конфигурацию из bots_data
            from bots_modules.imports_and_globals import bots_data, bots_data_lock
            with bots_data_lock:
                auto_config = bots_data.get('auto_bot_config', {}) or {}
        except ImportError:
            auto_config = {}
        if not auto_config:
            # Fallback при отдельном запуске ai.py: whitelist/blacklist/scope из data/coin_filters.json
            try:
                from bot_engine.ai.bots_data_helper import get_auto_bot_config
                auto_config = get_auto_bot_config() or {}
            except Exception:
                pass
        if not auto_config:
            # Не удалось загрузить конфигурацию — используем все монеты
            return True
        
        scope = auto_config.get('scope', 'all')
        whitelist = auto_config.get('whitelist', []) or []
        blacklist = auto_config.get('blacklist', []) or []
        
        # Нормализуем списки: элементы могут быть строками или объектами { symbol, added_at, updated_at }
        whitelist = [(item if isinstance(item, str) else item.get('symbol', '')).upper() for item in whitelist if item]
        blacklist = [(item if isinstance(item, str) else item.get('symbol', '')).upper() for item in blacklist if item]
        
        # Если whitelist не пуст (независимо от scope), обучаемся только на монетах из whitelist
        if whitelist:
            return symbol_upper in whitelist
        
        # Если scope == 'whitelist' но whitelist пуст, не обучаемся ни на чем
        if scope == 'whitelist':
            return False
        
        # Если scope == 'blacklist', исключаем монеты из blacklist
        if scope == 'blacklist':
            return symbol_upper not in blacklist
        
        # scope == 'all': исключаем только blacklist
        return symbol_upper not in blacklist
    
    def analyze_trade_results(self, trades: List[Dict]) -> Dict:
        """
        Анализирует результаты сделок и извлекает знания
        
        Args:
            trades: Список сделок с результатами
        
        Returns:
            Анализ результатов торговли
        """
        logger.info("=" * 80)
        logger.info("📚 АНАЛИЗ РЕЗУЛЬТАТОВ ТОРГОВЛИ ДЛЯ УЛУЧШЕНИЯ МЕТОДИКИ")
        logger.info("=" * 80)
        
        try:
            successful_trades = [t for t in trades if t.get('pnl', 0) > 0]
            failed_trades = [t for t in trades if t.get('pnl', 0) <= 0]
            
            logger.info(f"   📊 Всего сделок: {len(trades)}")
            logger.info(f"   ✅ Успешных: {len(successful_trades)}")
            logger.info(f"   ❌ Неуспешных: {len(failed_trades)}")
            
            # Анализ успешных паттернов
            successful_patterns = self._analyze_patterns(successful_trades, 'successful')
            
            # Анализ неуспешных паттернов
            failed_patterns = self._analyze_patterns(failed_trades, 'failed')
            
            # Извлечение инсайтов о рынке
            market_insights = self._extract_market_insights(trades)
            
            # Обновление базы знаний
            self._update_knowledge_base(successful_patterns, failed_patterns, market_insights)
            
            # Рекомендации по улучшению
            recommendations = self._generate_improvement_recommendations()
            
            analysis = {
                'successful_patterns': successful_patterns,
                'failed_patterns': failed_patterns,
                'market_insights': market_insights,
                'recommendations': recommendations,
                'analyzed_at': datetime.now().isoformat()
            }
            
            logger.info("=" * 80)
            logger.info("✅ АНАЛИЗ ЗАВЕРШЕН")
            logger.info("=" * 80)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа результатов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _analyze_patterns(self, trades: List[Dict], pattern_type: str) -> Dict:
        """Анализирует паттерны в сделках"""
        patterns = {
            'rsi_ranges': defaultdict(int),
            'trend_conditions': defaultdict(int),
            'volatility_conditions': defaultdict(int),
            'exit_reasons': defaultdict(int),
            'avg_pnl': 0,
            'avg_duration': 0
        }
        
        if not trades:
            return patterns
        
        rsi_values = []
        trends = []
        exit_reasons = []
        pnl_values = []
        
        for trade in trades:
            # RSI анализ
            entry_data = trade.get('entry_data', {})
            rsi = entry_data.get('rsi')
            if rsi:
                rsi_values.append(rsi)
                # Группируем по диапазонам
                if rsi <= 25:
                    patterns['rsi_ranges']['<=25'] += 1
                elif rsi <= 30:
                    patterns['rsi_ranges']['26-30'] += 1
                elif rsi <= 35:
                    patterns['rsi_ranges']['31-35'] += 1
                elif rsi >= 70:
                    patterns['rsi_ranges']['>=70'] += 1
                elif rsi >= 65:
                    patterns['rsi_ranges']['65-69'] += 1
            
            # Тренд анализ
            trend = entry_data.get('trend', 'NEUTRAL')
            trends.append(trend)
            patterns['trend_conditions'][trend] += 1
            
            # Причина выхода
            exit_reason = trade.get('exit_reason', 'UNKNOWN')
            exit_reasons.append(exit_reason)
            patterns['exit_reasons'][exit_reason] += 1
            
            # PnL
            pnl = trade.get('pnl', 0)
            pnl_values.append(pnl)
        
        if rsi_values:
            patterns['avg_rsi'] = np.mean(rsi_values)
            patterns['min_rsi'] = np.min(rsi_values)
            patterns['max_rsi'] = np.max(rsi_values)
        
        if pnl_values:
            patterns['avg_pnl'] = np.mean(pnl_values)
            patterns['min_pnl'] = np.min(pnl_values)
            patterns['max_pnl'] = np.max(pnl_values)
        
        # Конвертируем defaultdict в обычные dict
        patterns['rsi_ranges'] = dict(patterns['rsi_ranges'])
        patterns['trend_conditions'] = dict(patterns['trend_conditions'])
        patterns['exit_reasons'] = dict(patterns['exit_reasons'])
        
        return patterns
    
    def _extract_market_insights(self, trades: List[Dict]) -> List[Dict]:
        """Извлекает инсайты о рынке из сделок"""
        insights = []
        
        if not trades:
            return insights
        
        # Анализ лучших и худших сделок
        sorted_trades = sorted(trades, key=lambda x: x.get('pnl', 0), reverse=True)
        
        if len(sorted_trades) >= 5:
            best_trades = sorted_trades[:5]
            worst_trades = sorted_trades[-5:]
            
            # Инсайт о лучших сделках
            best_rsi_avg = np.mean([t.get('entry_data', {}).get('rsi', 50) for t in best_trades])
            best_trends = [t.get('entry_data', {}).get('trend', 'NEUTRAL') for t in best_trades]
            
            insights.append({
                'type': 'best_trades_pattern',
                'description': f'Лучшие сделки при среднем RSI {best_rsi_avg:.1f}',
                'trends': dict([(t, best_trends.count(t)) for t in set(best_trends)]),
                'avg_pnl': np.mean([t.get('pnl', 0) for t in best_trades])
            })
            
            # Инсайт о худших сделках (чего избегать)
            worst_rsi_avg = np.mean([t.get('entry_data', {}).get('rsi', 50) for t in worst_trades])
            worst_trends = [t.get('entry_data', {}).get('trend', 'NEUTRAL') for t in worst_trades]
            
            insights.append({
                'type': 'worst_trades_pattern',
                'description': f'Худшие сделки при среднем RSI {worst_rsi_avg:.1f}',
                'trends': dict([(t, worst_trends.count(t)) for t in set(worst_trends)]),
                'avg_pnl': np.mean([t.get('pnl', 0) for t in worst_trades]),
                'avoid': True
            })
        
        return insights
    
    def _update_knowledge_base(self, successful_patterns: Dict, failed_patterns: Dict, market_insights: List[Dict]):
        """Обновляет базу знаний на основе анализа"""
        # Обновляем успешные паттерны
        for key, value in successful_patterns.items():
            if key in ['rsi_ranges', 'trend_conditions', 'exit_reasons']:
                if key not in self.knowledge_base['successful_patterns']:
                    self.knowledge_base['successful_patterns'][key] = {}
                
                for sub_key, count in value.items():
                    if sub_key not in self.knowledge_base['successful_patterns'][key]:
                        self.knowledge_base['successful_patterns'][key][sub_key] = 0
                    self.knowledge_base['successful_patterns'][key][sub_key] += count
        
        # Обновляем неуспешные паттерны
        for key, value in failed_patterns.items():
            if key in ['rsi_ranges', 'trend_conditions', 'exit_reasons']:
                if key not in self.knowledge_base['failed_patterns']:
                    self.knowledge_base['failed_patterns'][key] = {}
                
                for sub_key, count in value.items():
                    if sub_key not in self.knowledge_base['failed_patterns'][key]:
                        self.knowledge_base['failed_patterns'][key][sub_key] = 0
                    self.knowledge_base['failed_patterns'][key][sub_key] += count
        
        # Добавляем инсайты о рынке
        self.knowledge_base['market_insights'].extend(market_insights)
        
        # Ограничиваем историю инсайтов (последние 1000)
        if len(self.knowledge_base['market_insights']) > 1000:
            self.knowledge_base['market_insights'] = self.knowledge_base['market_insights'][-1000:]
        
        # Сохраняем базу знаний
        self._save_knowledge_base()
    
    def _generate_improvement_recommendations(self) -> List[Dict]:
        """Генерирует рекомендации по улучшению торговли"""
        recommendations = []
        
        # Анализируем успешные и неуспешные паттерны
        successful_rsi = self.knowledge_base['successful_patterns'].get('rsi_ranges', {})
        failed_rsi = self.knowledge_base['failed_patterns'].get('rsi_ranges', {})
        
        # Рекомендация по RSI диапазонам
        if successful_rsi and failed_rsi:
            best_rsi_range = max(successful_rsi.items(), key=lambda x: x[1])[0] if successful_rsi else None
            worst_rsi_range = max(failed_rsi.items(), key=lambda x: x[1])[0] if failed_rsi else None
            
            if best_rsi_range:
                recommendations.append({
                    'type': 'rsi_optimization',
                    'recommendation': f'Предпочитать входы в диапазоне RSI {best_rsi_range}',
                    'confidence': successful_rsi[best_rsi_range] / sum(successful_rsi.values()) if successful_rsi else 0
                })
            
            if worst_rsi_range:
                recommendations.append({
                    'type': 'rsi_avoidance',
                    'recommendation': f'Избегать входов в диапазоне RSI {worst_rsi_range}',
                    'confidence': failed_rsi[worst_rsi_range] / sum(failed_rsi.values()) if failed_rsi else 0
                })
        
        return recommendations
    
    def get_optimal_parameters_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Получить оптимальные параметры для символа на основе базы знаний
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Оптимальные параметры или None
        """
        return self.knowledge_base.get('optimal_parameters', {}).get(symbol)
    
    def learn_from_real_trades(self, trades: List[Dict]):
        """
        Обучение на реальных сделках с постоянным улучшением
        
        Args:
            trades: Список реальных сделок с результатами
        """
        logger.info("=" * 80)
        logger.info("🧠 ПОСТОЯННОЕ ОБУЧЕНИЕ НА РЕАЛЬНЫХ СДЕЛКАХ")
        logger.info("=" * 80)
        
        # Фильтруем сделки по whitelist/blacklist
        original_trades_count = len(trades)
        filtered_trades = []
        for trade in trades:
            symbol = trade.get('symbol', '')
            if self._should_train_on_symbol(symbol):
                filtered_trades.append(trade)
        
        trades = filtered_trades
        filtered_count = len(trades)
        skipped_by_filter = original_trades_count - filtered_count
        
        if skipped_by_filter > 0:
            logger.info(f"🎯 Фильтрация по whitelist/blacklist: {original_trades_count} → {filtered_count} сделок ({skipped_by_filter} пропущено)")
        
        if len(trades) < 10:
            logger.info(f"⏳ Недостаточно сделок для обучения (есть {len(trades)}, нужно минимум 10)")
            return
        
        # Анализируем результаты
        analysis = self.analyze_trade_results(trades)
        
        # Извлекаем уроки
        lessons = self._extract_lessons(analysis)
        
        # Применяем улучшения
        improvements = self._apply_improvements(lessons)
        
        logger.info("=" * 80)
        logger.info("✅ ПОСТОЯННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info(f"   📚 Извлечено уроков: {len(lessons)}")
        logger.info(f"   🔧 Применено улучшений: {len(improvements)}")
        logger.info("=" * 80)
    
    def _extract_lessons(self, analysis: Dict) -> List[Dict]:
        """Извлекает уроки из анализа"""
        lessons = []
        
        successful = analysis.get('successful_patterns', {})
        failed = analysis.get('failed_patterns', {})
        
        # Урок о RSI
        if successful.get('avg_rsi') and failed.get('avg_rsi'):
            successful_rsi = successful['avg_rsi']
            failed_rsi = failed['avg_rsi']
            
            if abs(successful_rsi - failed_rsi) > 5:
                lessons.append({
                    'type': 'rsi_lesson',
                    'message': f'Успешные сделки при RSI {successful_rsi:.1f}, неуспешные при {failed_rsi:.1f}',
                    'action': 'adjust_rsi_thresholds'
                })
        
        # Урок о трендах
        successful_trends = successful.get('trend_conditions', {})
        failed_trends = failed.get('trend_conditions', {})
        
        if successful_trends and failed_trends:
            best_trend = max(successful_trends.items(), key=lambda x: x[1])[0] if successful_trends else None
            worst_trend = max(failed_trends.items(), key=lambda x: x[1])[0] if failed_trends else None
            
            if best_trend and worst_trend and best_trend != worst_trend:
                lessons.append({
                    'type': 'trend_lesson',
                    'message': f'Лучший тренд: {best_trend}, худший: {worst_trend}',
                    'action': 'prefer_trend',
                    'preferred_trend': best_trend
                })
        
        return lessons
    
    def _apply_improvements(self, lessons: List[Dict]) -> List[Dict]:
        """Применяет улучшения на основе уроков"""
        improvements = []

        for lesson in lessons:
            lesson_type = lesson.get('type')

            if lesson_type == 'rsi_lesson':
                # Сохраняем рекомендацию по RSI
                improvements.append({
                    'type': 'rsi_adjustment',
                    'lesson': lesson,
                    'applied_at': datetime.now().isoformat()
                })

            elif lesson_type == 'trend_lesson':
                # Сохраняем рекомендацию по тренду
                improvements.append({
                    'type': 'trend_preference',
                    'lesson': lesson,
                    'applied_at': datetime.now().isoformat()
                })

        # НОВОЕ: Применяем улучшения к ML моделям
        if improvements:
            try:
                self._apply_learning_to_models(improvements)
            except Exception as e:
                pass

        # Сохраняем историю улучшений
        self.knowledge_base['improvement_history'].extend(improvements)

        # Ограничиваем историю (последние 500 улучшений)
        if len(self.knowledge_base['improvement_history']) > 500:
            self.knowledge_base['improvement_history'] = self.knowledge_base['improvement_history'][-500:]

        self._save_knowledge_base()

        return improvements

    def _apply_learning_to_models(self, improvements: List[Dict]) -> None:
        """
        Применяет извлеченные уроки к ML моделям для их улучшения

        Args:
            improvements: Список улучшений для применения
        """
        try:
            # Получаем AI систему
            from bot_engine.ai.ai_integration import get_ai_system
            ai_system = get_ai_system()

            if not ai_system or not ai_system.trainer:
                pass
                return

            logger.info(f"🔄 Применение {len(improvements)} улучшений к ML моделям...")

            # Анализируем улучшения и корректируем модели
            rsi_adjustments = []
            trend_preferences = []

            for improvement in improvements:
                if improvement['type'] == 'rsi_adjustment':
                    rsi_adjustments.append(improvement)
                elif improvement['type'] == 'trend_preference':
                    trend_preferences.append(improvement)

            # Применяем корректировки RSI
            if rsi_adjustments:
                self._adjust_model_for_rsi(rsi_adjustments, ai_system.trainer)

            # Применяем предпочтения трендов
            if trend_preferences:
                self._adjust_model_for_trends(trend_preferences, ai_system.trainer)

            logger.info("✅ Улучшения применены к ML моделям")

        except Exception as e:
            logger.error(f"❌ Ошибка применения улучшений к моделям: {e}")

    def _adjust_model_for_rsi(self, rsi_adjustments: List[Dict], ai_trainer) -> None:
        """
        Корректирует модель на основе уроков по RSI

        Args:
            rsi_adjustments: Корректировки RSI
            ai_trainer: Экземпляр AITrainer
        """
        try:
            # Анализируем все корректировки RSI
            successful_rsi_avg = np.mean([adj['lesson']['message'].split()[2] for adj in rsi_adjustments if 'Успешные' in adj['lesson']['message']])
            failed_rsi_avg = np.mean([adj['lesson']['message'].split()[6] for adj in rsi_adjustments if 'Успешные' in adj['lesson']['message']])

            if abs(successful_rsi_avg - failed_rsi_avg) > 2:
                # Создаем обучающие данные на основе корректировок
                correction_data = {
                    'rsi_correction': successful_rsi_avg - failed_rsi_avg,
                    'confidence': len(rsi_adjustments) / 10.0  # Уверенность на основе количества примеров
                }

                # Сохраняем корректировку для использования при следующем обучении
                self.knowledge_base['model_corrections'] = self.knowledge_base.get('model_corrections', {})
                self.knowledge_base['model_corrections']['rsi'] = correction_data

                logger.info(f"📊 Скорректирована модель RSI: успешные сделки при RSI {successful_rsi_avg:.1f}, неуспешные при {failed_rsi_avg:.1f}")

        except Exception as e:
            pass

    def _adjust_model_for_trends(self, trend_preferences: List[Dict], ai_trainer) -> None:
        """
        Корректирует модель на основе предпочтений трендов

        Args:
            trend_preferences: Предпочтения трендов
            ai_trainer: Экземпляр AITrainer
        """
        try:
            # Анализируем предпочтения трендов
            preferred_trends = {}
            avoided_trends = {}

            for pref in trend_preferences:
                preferred = pref['lesson'].get('preferred_trend')
                avoided = pref['lesson'].get('message', '').split()[-1]  # Последнее слово - худший тренд

                if preferred:
                    preferred_trends[preferred] = preferred_trends.get(preferred, 0) + 1
                if avoided:
                    avoided_trends[avoided] = avoided_trends.get(avoided, 0) + 1

            # Определяем наиболее предпочитаемый тренд
            if preferred_trends:
                best_trend = max(preferred_trends.items(), key=lambda x: x[1])[0]
                self.knowledge_base['model_corrections'] = self.knowledge_base.get('model_corrections', {})
                self.knowledge_base['model_corrections']['trend_preference'] = best_trend

                logger.info(f"📈 Предпочитаемый тренд для модели: {best_trend}")

        except Exception as e:
            pass


    def evaluate_ai_performance(self, trades: List[Dict]) -> Dict:
        """
        Оценивает производительность AI на основе сделок

        Args:
            trades: Список сделок с результатами

        Returns:
            Словарь с метриками производительности AI
        """
        try:
            logger.info("📊 Оценка производительности AI...")

            # Разделяем сделки с AI и без AI
            ai_trades = [t for t in trades if t.get('ai_used', False)]
            non_ai_trades = [t for t in trades if not t.get('ai_used', False)]

            metrics = {
                'total_trades': len(trades),
                'ai_trades': len(ai_trades),
                'non_ai_trades': len(non_ai_trades),
                'ai_trades_percentage': (len(ai_trades) / len(trades) * 100) if trades else 0,
                'evaluation_timestamp': datetime.now().isoformat()
            }

            # Оцениваем AI сделки
            if ai_trades:
                ai_successful = len([t for t in ai_trades if t.get('pnl', 0) > 0])
                ai_win_rate = ai_successful / len(ai_trades) if ai_trades else 0
                ai_avg_pnl = np.mean([t.get('pnl', 0) for t in ai_trades]) if ai_trades else 0
                ai_total_pnl = sum([t.get('pnl', 0) for t in ai_trades])

                metrics.update({
                    'ai_win_rate': ai_win_rate,
                    'ai_avg_pnl': ai_avg_pnl,
                    'ai_total_pnl': ai_total_pnl,
                    'ai_successful_trades': ai_successful,
                    'ai_failed_trades': len(ai_trades) - ai_successful
                })

            # Оцениваем не-AI сделки для сравнения
            if non_ai_trades:
                non_ai_successful = len([t for t in non_ai_trades if t.get('pnl', 0) > 0])
                non_ai_win_rate = non_ai_successful / len(non_ai_trades) if non_ai_trades else 0
                non_ai_avg_pnl = np.mean([t.get('pnl', 0) for t in non_ai_trades]) if non_ai_trades else 0
                non_ai_total_pnl = sum([t.get('pnl', 0) for t in non_ai_trades])

                metrics.update({
                    'non_ai_win_rate': non_ai_win_rate,
                    'non_ai_avg_pnl': non_ai_avg_pnl,
                    'non_ai_total_pnl': non_ai_total_pnl,
                    'non_ai_successful_trades': non_ai_successful,
                    'non_ai_failed_trades': len(non_ai_trades) - non_ai_successful
                })

            # Сравнение AI vs не-AI
            if ai_trades and non_ai_trades:
                win_rate_diff = metrics['ai_win_rate'] - metrics['non_ai_win_rate']
                avg_pnl_diff = metrics['ai_avg_pnl'] - metrics['non_ai_avg_pnl']

                metrics.update({
                    'win_rate_difference': win_rate_diff,
                    'avg_pnl_difference': avg_pnl_diff,
                    'ai_better_win_rate': win_rate_diff > 0,
                    'ai_better_avg_pnl': avg_pnl_diff > 0
                })

                # Определяем общую оценку AI
                ai_score = 0
                if win_rate_diff > 0.05:  # AI лучше на 5%+ по win rate
                    ai_score += 1
                if avg_pnl_diff > 10:  # AI лучше на $10+ в среднем
                    ai_score += 1
                if metrics['ai_win_rate'] > 0.6:  # AI имеет win rate > 60%
                    ai_score += 1

                metrics['ai_performance_score'] = ai_score  # 0-3 шкала
                metrics['ai_performance_rating'] = self._get_performance_rating(ai_score)

                logger.info("📊 Оценка AI:")
                logger.info(f"   Win Rate AI: {metrics['ai_win_rate']:.1%} vs Без AI: {metrics['non_ai_win_rate']:.1%} (разница: {win_rate_diff:.1%})")
                logger.info(f"   Avg PnL AI: ${metrics['ai_avg_pnl']:.2f} vs Без AI: ${metrics['non_ai_avg_pnl']:.2f} (разница: ${avg_pnl_diff:.2f})")
                logger.info(f"   Рейтинг AI: {metrics['ai_performance_rating']} (балл: {ai_score}/3)")

            # Сохраняем метрики в knowledge base
            self.knowledge_base['performance_metrics'] = self.knowledge_base.get('performance_metrics', [])
            self.knowledge_base['performance_metrics'].append(metrics)

            # Ограничиваем историю (последние 100 оценок)
            if len(self.knowledge_base['performance_metrics']) > 100:
                self.knowledge_base['performance_metrics'] = self.knowledge_base['performance_metrics'][-100:]

            self._save_knowledge_base()

            return metrics

        except Exception as e:
            logger.error(f"❌ Ошибка оценки производительности AI: {e}")
            return {}

    def _get_performance_rating(self, score: int) -> str:
        """
        Получить текстовую оценку производительности AI

        Args:
            score: Числовой балл (0-3)

        Returns:
            Текстовая оценка
        """
        ratings = {
            0: "Критично низкая - требует улучшений",
            1: "Низкая - нуждается в доработке",
            2: "Средняя - работает, но можно лучше",
            3: "Высокая - отличная производительность"
        }
        return ratings.get(score, "Неизвестно")

    def get_performance_trends(self) -> Dict:
        """
        Анализирует тренды производительности AI со временем

        Returns:
            Словарь с трендами производительности
        """
        try:
            metrics_history = self.knowledge_base.get('performance_metrics', [])

            if len(metrics_history) < 2:
                return {'error': 'Недостаточно данных для анализа трендов'}

            # Анализируем последние 10 оценок
            recent_metrics = metrics_history[-10:]

            trends = {
                'period_analyzed': len(recent_metrics),
                'win_rate_trend': self._calculate_trend([m.get('ai_win_rate', 0) for m in recent_metrics]),
                'avg_pnl_trend': self._calculate_trend([m.get('ai_avg_pnl', 0) for m in recent_metrics]),
                'performance_score_trend': self._calculate_trend([m.get('ai_performance_score', 0) for m in recent_metrics]),
                'latest_performance': recent_metrics[-1] if recent_metrics else {}
            }

            # Определяем, улучшается ли AI
            improving = (
                trends['win_rate_trend'] > 0 and
                trends['avg_pnl_trend'] > 0 and
                trends['performance_score_trend'] >= 0
            )

            trends['ai_improving'] = improving
            trends['trend_summary'] = "AI улучшается" if improving else "AI стабильна или ухудшается"

            return trends

        except Exception as e:
            logger.error(f"❌ Ошибка анализа трендов производительности: {e}")
            return {'error': str(e)}

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Вычисляет тренд в значениях (линейная регрессия)

        Args:
            values: Список значений

        Returns:
            Коэффициент тренда (положительный = рост, отрицательный = падение)
        """
        try:
            if len(values) < 2:
                return 0

            x = np.arange(len(values))
            y = np.array(values)

            # Линейная регрессия
            slope = np.polyfit(x, y, 1)[0]

            return slope

        except Exception:
            return 0
