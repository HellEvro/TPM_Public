#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для хранения данных AI модуля

ВАЖНО: Все данные теперь хранятся в БД (ai_data.db)!
JSON файлы больше не используются.

Управляет данными для:
- Отслеживания решений AI (таблица ai_decisions)
- История обучения (таблица training_sessions)
- Метрики производительности (таблица performance_metrics)
- Версии моделей (таблица model_versions)
"""

import os
import json
import logging
import time
import uuid
import shutil
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import RLock

logger = logging.getLogger('AI.DataStorage')


class AIDataStorage:
    """Класс для управления данными AI модуля через БД"""
    
    def __init__(self, data_dir: str = 'data/ai'):
        self.data_dir = data_dir
        self.lock = RLock()
        
        # Создаем директорию если её нет
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            if self.ai_db:
                logger.info("✅ AI Database подключена для AIDataStorage")
            else:
                logger.warning("⚠️ AI Database не доступна")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
    
    # ==================== Управление решениями AI ====================
    
    def save_ai_decision(self, decision_id: str, decision_data: Dict):
        """Сохранить решение AI в БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, решение не сохранено")
            return
        
        try:
            # Преобразуем формат для БД
            # Важно: ai_signal может быть передан как 'ai_signal' или 'signal'
            signal = decision_data.get('signal') or decision_data.get('ai_signal')
            confidence = decision_data.get('confidence') or decision_data.get('ai_confidence')
            
            # Проверяем, что signal не None (обязательное поле в БД)
            if signal is None:
                logger.warning(f"⚠️ Signal не указан в решении AI для {decision_id}, используем 'WAIT'")
                signal = 'WAIT'
            
            # Проверяем, что confidence не None
            if confidence is None:
                confidence = 0.0
            
            decision = {
                'decision_id': decision_id,
                'symbol': decision_data.get('symbol'),
                'decision_type': decision_data.get('decision_type', 'SIGNAL'),
                'signal': signal,  # Преобразуем ai_signal -> signal
                'confidence': confidence,  # Преобразуем ai_confidence -> confidence
                'rsi': decision_data.get('rsi'),
                'trend': decision_data.get('trend'),
                'price': decision_data.get('price'),
                'market_data': decision_data.get('market_data'),
                'params': decision_data.get('params')
            }
            
            self.ai_db.save_ai_decision(decision)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения решения AI: {e}")
    
    def update_ai_decision(self, decision_id: str, updates: Dict):
        """Обновить решение AI в БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, решение не обновлено")
            return False
        
        try:
            # Если есть результат - обновляем через специальный метод
            if 'pnl' in updates or 'is_successful' in updates:
                pnl = updates.get('pnl', 0)
                is_successful = updates.get('is_successful', False)
                self.ai_db.update_ai_decision_result(decision_id, pnl, is_successful)
                return True
            else:
                # Для других обновлений нужно получить текущее решение и обновить
                decisions = self.ai_db.get_ai_decisions()
                for decision in decisions:
                    if decision.get('decision_id') == decision_id:
                        # Обновляем через сохранение с обновленными данными
                        decision.update(updates)
                        self.ai_db.save_ai_decision(decision)
                        return True
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка обновления решения AI: {e}")
            return False
    
    def get_ai_decisions(self, status: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Получить решения AI с фильтрацией из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return []
        
        try:
            decisions = self.ai_db.get_ai_decisions(status=status, symbol=symbol)
            
            # Преобразуем формат для совместимости
            result = []
            for decision in decisions:
                result.append({
                    'id': decision.get('decision_id'),
                    'symbol': decision.get('symbol'),
                    'decision_type': decision.get('decision_type'),
                    'signal': decision.get('signal'),
                    'confidence': decision.get('confidence'),
                    'rsi': decision.get('rsi'),
                    'trend': decision.get('trend'),
                    'price': decision.get('price'),
                    'market_data': decision.get('market_data'),
                    'params': decision.get('params'),
                    'status': decision.get('status', 'PENDING'),
                    'pnl': decision.get('result_pnl'),
                    'timestamp': decision.get('created_at')
                })
            
            return result
        except Exception as e:
            logger.error(f"❌ Ошибка получения решений AI: {e}")
            return []

    def save_ai_recommendation(self, symbol: str, direction: str, data: Dict) -> None:
        """Сохранить последнюю рекомендацию AI (вызывает только ai.py)."""
        if not self.ai_db:
            return
        try:
            self.ai_db.save_ai_recommendation(symbol, direction, data)
        except Exception as e:
            logger.warning(f"save_ai_recommendation: {e}")

    def get_latest_ai_recommendation(self, symbol: str, direction: str) -> Optional[Dict]:
        """Получить последнюю рекомендацию AI по символу и направлению (читает bots.py)."""
        if not self.ai_db:
            return None
        try:
            return self.ai_db.get_latest_ai_recommendation(symbol, direction)
        except Exception as e:
            logger.warning(f"get_latest_ai_recommendation: {e}")
            return None
    
    # ==================== История обучения ====================
    
    def add_training_record(self, training_data: Dict):
        """Добавить запись об обучении в БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, запись не добавлена")
            return
        
        try:
            # Добавляем timestamp если его нет
            if 'timestamp' not in training_data:
                training_data['timestamp'] = datetime.now().isoformat()
            
            self.ai_db.add_training_history_record(training_data)
            logger.info(f"🧠 Добавлена запись обучения AI в БД — event={training_data.get('event_type')}, status={training_data.get('status')}")
        except Exception as e:
            logger.error(f"❌ Ошибка добавления записи об обучении: {e}")
    
    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """Получить историю обучения из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return []
        
        try:
            return self.ai_db.get_training_history(limit=limit)
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории обучения: {e}")
            return []
    
    # ==================== Метрики производительности ====================
    
    def update_performance_metrics(self, metrics: Dict):
        """Обновить метрики производительности в БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, метрики не обновлены")
            return
        
        try:
            self.ai_db.save_performance_metrics(metrics)
        except Exception as e:
            logger.error(f"❌ Ошибка обновления метрик: {e}")
    
    def calculate_performance_metrics(self) -> Dict:
        """Вычислить метрики производительности на основе решений AI из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return {}
        
        try:
            decisions = self.ai_db.get_ai_decisions()
            
            if not decisions:
                return {}
            
            total_decisions = len(decisions)
            successful = sum(1 for d in decisions if d.get('status') == 'SUCCESS')
            failed = total_decisions - successful
            
            total_pnl = sum(d.get('result_pnl', 0) or 0 for d in decisions)
            avg_pnl = total_pnl / total_decisions if total_decisions > 0 else 0
            win_rate = successful / total_decisions if total_decisions > 0 else 0
            
            # Метрики по символам
            by_symbol = {}
            for decision in decisions:
                symbol = decision.get('symbol')
                if symbol:
                    if symbol not in by_symbol:
                        by_symbol[symbol] = {
                            'decisions': 0,
                            'successful': 0,
                            'failed': 0,
                            'total_pnl': 0
                        }
                    by_symbol[symbol]['decisions'] += 1
                    if decision.get('status') == 'SUCCESS':
                        by_symbol[symbol]['successful'] += 1
                    else:
                        by_symbol[symbol]['failed'] += 1
                    by_symbol[symbol]['total_pnl'] += decision.get('result_pnl', 0) or 0
            
            # Вычисляем win_rate и avg_pnl для каждого символа
            for symbol, metrics in by_symbol.items():
                metrics['win_rate'] = metrics['successful'] / metrics['decisions'] if metrics['decisions'] > 0 else 0
                metrics['avg_pnl'] = metrics['total_pnl'] / metrics['decisions'] if metrics['decisions'] > 0 else 0
            
            return {
                'overall': {
                    'total_ai_decisions': total_decisions,
                    'successful_decisions': successful,
                    'failed_decisions': failed,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': total_pnl,
                    'last_updated': datetime.now().isoformat()
                },
                'by_symbol': by_symbol
            }
        except Exception as e:
            logger.error(f"❌ Ошибка вычисления метрик: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict:
        """Получить метрики производительности из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return {'overall': {}, 'vs_script': {}, 'by_symbol': {}}
        
        try:
            return self.ai_db.get_performance_metrics()
        except Exception as e:
            logger.error(f"❌ Ошибка получения метрик: {e}")
            return {'overall': {}, 'vs_script': {}, 'by_symbol': {}}
    
    # ==================== Версии моделей ====================
    
    def save_model_version(self, version_data: Dict):
        """Сохранить информацию о версии модели в БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, версия модели не сохранена")
            return
        
        try:
            # Добавляем id если его нет
            if 'id' not in version_data:
                version_data['id'] = f"model_{int(datetime.now().timestamp())}"
            
            # Добавляем timestamp если его нет
            if 'timestamp' not in version_data:
                version_data['timestamp'] = datetime.now().isoformat()
            
            self.ai_db.save_model_version(version_data)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения версии модели: {e}")
    
    def get_model_versions(self, limit: int = 10) -> List[Dict]:
        """Получить версии моделей из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return []
        
        try:
            return self.ai_db.get_model_versions(limit=limit)
        except Exception as e:
            logger.error(f"❌ Ошибка получения версий моделей: {e}")
            return []
    
    def get_latest_model_version(self) -> Optional[Dict]:
        """Получить последнюю версию модели из БД"""
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return None
        
        try:
            return self.ai_db.get_latest_model_version()
        except Exception as e:
            logger.error(f"❌ Ошибка получения последней версии модели: {e}")
            return None
