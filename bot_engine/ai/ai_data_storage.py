#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для хранения данных AI модуля

Управляет JSON файлами для:
- Отслеживания решений AI
- История обучения
- Метрики производительности
- Версии моделей
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger('AI.DataStorage')


class AIDataStorage:
    """Класс для управления данными AI модуля"""
    
    def __init__(self, data_dir: str = 'data/ai'):
        self.data_dir = data_dir
        self.lock = Lock()
        
        # Создаем директорию если её нет
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Пути к файлам
        self.decisions_file = os.path.join(self.data_dir, 'ai_decisions_tracking.json')
        self.training_history_file = os.path.join(self.data_dir, 'ai_training_history.json')
        self.performance_metrics_file = os.path.join(self.data_dir, 'ai_performance_metrics.json')
        self.model_versions_file = os.path.join(self.data_dir, 'ai_model_versions.json')
        
        # Инициализируем файлы если их нет
        self._init_files()
    
    def _init_files(self):
        """Инициализирует файлы если их нет"""
        try:
            if not os.path.exists(self.decisions_file):
                self._save_data(self.decisions_file, {})
            
            if not os.path.exists(self.training_history_file):
                self._save_data(self.training_history_file, {'trainings': []})
            
            if not os.path.exists(self.performance_metrics_file):
                self._save_data(self.performance_metrics_file, {
                    'overall': {},
                    'vs_script': {},
                    'by_symbol': {}
                })
            
            if not os.path.exists(self.model_versions_file):
                self._save_data(self.model_versions_file, {'versions': []})
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации файлов: {e}")
    
    def _load_data(self, filepath: str) -> Dict:
        """Загрузить данные из файла"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError as json_error:
            logger.warning(f"⚠️ Файл {filepath} поврежден (JSON ошибка на позиции {json_error.pos})")
            logger.info("🗑️ Удаляем поврежденный файл")
            try:
                os.remove(filepath)
                logger.info("✅ Поврежденный файл удален")
            except Exception as del_error:
                logger.debug(f"⚠️ Не удалось удалить файл: {del_error}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных из {filepath}: {e}")
        return {}
    
    def _save_data(self, filepath: str, data: Dict):
        """Сохранить данные в файл (безопасно с retry логикой)"""
        max_retries = 5
        retry_delay = 0.5  # секунд
        
        for attempt in range(max_retries):
            try:
                with self.lock:
                    # Создаем уникальное имя временного файла
                    temp_file = f"{filepath}.tmp.{uuid.uuid4().hex[:8]}"
                    
                    # Сохраняем во временный файл сначала
                    try:
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as write_error:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                        raise write_error
                    
                    # Заменяем оригинальный файл атомарно
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except PermissionError:
                            if attempt < max_retries - 1:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except:
                                    pass
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                    
                    # Переименовываем временный файл
                    try:
                        os.rename(temp_file, filepath)
                    except PermissionError:
                        if attempt < max_retries - 1:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except:
                                pass
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise
                    
                    # Успешно сохранено
                    return
                    
            except (PermissionError, OSError) as file_error:
                if attempt < max_retries - 1:
                    logger.debug(f"⚠️ Файл {filepath} занят, повторная попытка {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"⚠️ Не удалось сохранить {filepath} после {max_retries} попыток (файл занят)")
                    logger.debug(f"   Ошибка: {file_error}")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения данных в {filepath}: {e}")
                return
    
    # ==================== Управление решениями AI ====================
    
    def save_ai_decision(self, decision_id: str, decision_data: Dict):
        """Сохранить решение AI"""
        try:
            with self.lock:
                decisions = self._load_data(self.decisions_file)
                decisions[decision_id] = decision_data
                self._save_data(self.decisions_file, decisions)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения решения AI: {e}")
    
    def update_ai_decision(self, decision_id: str, updates: Dict):
        """Обновить решение AI"""
        try:
            with self.lock:
                decisions = self._load_data(self.decisions_file)
                if decision_id in decisions:
                    decisions[decision_id].update(updates)
                    self._save_data(self.decisions_file, decisions)
                    return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления решения AI: {e}")
        return False
    
    def get_ai_decisions(self, status: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Получить решения AI с фильтрацией"""
        try:
            decisions = self._load_data(self.decisions_file)
            result = []
            
            for decision_id, decision in decisions.items():
                if status and decision.get('status') != status:
                    continue
                if symbol and decision.get('symbol') != symbol:
                    continue
                decision['id'] = decision_id
                result.append(decision)
            
            # Сортируем по времени (новые первыми)
            result.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return result
        except Exception as e:
            logger.error(f"❌ Ошибка получения решений AI: {e}")
            return []
    
    # ==================== История обучения ====================
    
    def add_training_record(self, training_data: Dict):
        """Добавить запись об обучении"""
        try:
            with self.lock:
                history = self._load_data(self.training_history_file)
                trainings = history.get('trainings', [])
                
                training_record = {
                    'id': f"training_{int(datetime.now().timestamp())}",
                    'timestamp': datetime.now().isoformat(),
                    **training_data
                }
                
                trainings.append(training_record)
                
                # Ограничиваем историю последними 100 записями
                if len(trainings) > 100:
                    trainings = trainings[-100:]
                
                history['trainings'] = trainings
                self._save_data(self.training_history_file, history)
        except Exception as e:
            logger.error(f"❌ Ошибка добавления записи об обучении: {e}")
    
    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """Получить историю обучения"""
        try:
            history = self._load_data(self.training_history_file)
            trainings = history.get('trainings', [])
            return trainings[-limit:] if limit else trainings
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории обучения: {e}")
            return []
    
    # ==================== Метрики производительности ====================
    
    def update_performance_metrics(self, metrics: Dict):
        """Обновить метрики производительности"""
        try:
            with self.lock:
                data = self._load_data(self.performance_metrics_file)
                
                # Обновляем общие метрики
                if 'overall' in metrics:
                    data['overall'].update(metrics['overall'])
                
                # Обновляем сравнение с скриптовыми правилами
                if 'vs_script' in metrics:
                    data['vs_script'].update(metrics['vs_script'])
                
                # Обновляем метрики по символам
                if 'by_symbol' in metrics:
                    if 'by_symbol' not in data:
                        data['by_symbol'] = {}
                    data['by_symbol'].update(metrics['by_symbol'])
                
                self._save_data(self.performance_metrics_file, data)
        except Exception as e:
            logger.error(f"❌ Ошибка обновления метрик: {e}")
    
    def calculate_performance_metrics(self) -> Dict:
        """Вычислить метрики производительности на основе решений AI"""
        try:
            decisions = self.get_ai_decisions(status='SUCCESS') + self.get_ai_decisions(status='FAILED')
            
            if not decisions:
                return {}
            
            total_decisions = len(decisions)
            successful = sum(1 for d in decisions if d.get('status') == 'SUCCESS')
            failed = total_decisions - successful
            
            total_pnl = sum(d.get('pnl', 0) for d in decisions)
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
                    by_symbol[symbol]['total_pnl'] += decision.get('pnl', 0)
            
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
        """Получить метрики производительности"""
        try:
            return self._load_data(self.performance_metrics_file)
        except Exception as e:
            logger.error(f"❌ Ошибка получения метрик: {e}")
            return {}
    
    # ==================== Версии моделей ====================
    
    def save_model_version(self, version_data: Dict):
        """Сохранить информацию о версии модели"""
        try:
            with self.lock:
                data = self._load_data(self.model_versions_file)
                versions = data.get('versions', [])
                
                version_record = {
                    'id': f"model_{int(datetime.now().timestamp())}",
                    'timestamp': datetime.now().isoformat(),
                    **version_data
                }
                
                versions.append(version_record)
                
                # Ограничиваем историю последними 50 версиями
                if len(versions) > 50:
                    versions = versions[-50:]
                
                data['versions'] = versions
                self._save_data(self.model_versions_file, data)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения версии модели: {e}")
    
    def get_model_versions(self, limit: int = 10) -> List[Dict]:
        """Получить версии моделей"""
        try:
            data = self._load_data(self.model_versions_file)
            versions = data.get('versions', [])
            return versions[-limit:] if limit else versions
        except Exception as e:
            logger.error(f"❌ Ошибка получения версий моделей: {e}")
            return []
    
    def get_latest_model_version(self) -> Optional[Dict]:
        """Получить последнюю версию модели"""
        versions = self.get_model_versions(limit=1)
        return versions[0] if versions else None

