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
            logger.debug(f"⚠️ Ошибка загрузки базы знаний из БД: {e}")
        
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
            logger.debug("✅ База знаний сохранена в БД")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения базы знаний в БД: {e}")
    
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
        
        # Сохраняем историю улучшений
        self.knowledge_base['improvement_history'].extend(improvements)
        
        # Ограничиваем историю (последние 500 улучшений)
        if len(self.knowledge_base['improvement_history']) > 500:
            self.knowledge_base['improvement_history'] = self.knowledge_base['improvement_history'][-500:]
        
        self._save_knowledge_base()
        
        return improvements

