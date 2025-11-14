#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сбора данных для AI системы

Собирает данные из:
- bots.py (свечи, RSI, стохастик, сигналы)
- bot_history.py (история трейдов)
- Рыночные данные
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

logger = logging.getLogger('AI.DataCollector')


class AIDataCollector:
    """
    Сборщик данных для обучения AI
    """
    
    def __init__(self, bots_service_url: str = 'http://127.0.0.1:5001',
                 app_service_url: str = 'http://127.0.0.1:5000'):
        """
        Инициализация сборщика данных
        
        Args:
            bots_service_url: URL сервиса bots.py
            app_service_url: URL сервиса app.py
        """
        self.bots_service_url = bots_service_url
        self.app_service_url = app_service_url
        self.data_dir = 'data/ai'
        self.lock = threading.Lock()
        
        # Создаем директорию для данных
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Файлы для хранения данных
        self.market_data_file = os.path.join(self.data_dir, 'market_data.json')
        self.bots_data_file = os.path.join(self.data_dir, 'bots_data.json')
        self.history_data_file = os.path.join(self.data_dir, 'history_data.json')
        
        logger.info("✅ AIDataCollector инициализирован")
    
    def _load_data(self, filepath: str) -> Dict:
        """Загрузить данные из файла"""
        try:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError as json_error:
                    # Файл поврежден - удаляем его
                    logger.warning(f"⚠️ Файл {filepath} поврежден (JSON ошибка на позиции {json_error.pos})")
                    logger.info("🗑️ Удаляем поврежденный файл")
                    try:
                        os.remove(filepath)
                        logger.info("✅ Поврежденный файл удален")
                    except Exception as del_error:
                        logger.debug(f"⚠️ Не удалось удалить файл: {del_error}")
                    return {}
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных из {filepath}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        return {}
    
    def _save_data(self, filepath: str, data: Dict):
        """
        Сохранить данные в файл (безопасно с retry логикой)
        
        Использует временный файл и атомарную замену для избежания конфликтов
        """
        max_retries = 5
        retry_delay = 0.5  # секунд
        
        for attempt in range(max_retries):
            try:
                with self.lock:
                    # Создаем уникальное имя временного файла
                    import uuid
                    temp_file = f"{filepath}.tmp.{uuid.uuid4().hex[:8]}"
                    
                    # Сохраняем во временный файл сначала
                    try:
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as write_error:
                        # Удаляем временный файл если ошибка записи
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
                        except PermissionError as perm_error:
                            # Файл занят - ждем и пробуем снова
                            if attempt < max_retries - 1:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except:
                                    pass
                                time.sleep(retry_delay * (attempt + 1))  # Увеличиваем задержку
                                continue
                            else:
                                raise perm_error
                    
                    # Переименовываем временный файл
                    try:
                        os.rename(temp_file, filepath)
                    except PermissionError as perm_error:
                        # Файл все еще занят
                        if attempt < max_retries - 1:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except:
                                pass
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise perm_error
                    
                    # Успешно сохранено
                    return
                    
            except PermissionError as perm_error:
                # Windows: файл занят другим процессом
                if attempt < max_retries - 1:
                    logger.debug(f"⚠️ Файл {filepath} занят, повторная попытка {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"⚠️ Не удалось сохранить {filepath} после {max_retries} попыток (файл занят другим процессом)")
                    logger.debug(f"   Ошибка: {perm_error}")
            except OSError as os_error:
                # Другие ошибки ОС (WinError 32 и т.д.)
                if attempt < max_retries - 1:
                    logger.debug(f"⚠️ Ошибка доступа к файлу {filepath}, повторная попытка {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"⚠️ Не удалось сохранить {filepath} после {max_retries} попыток")
                    logger.debug(f"   Ошибка: {os_error}")
            except Exception as e:
                # Другие ошибки
                logger.error(f"❌ Ошибка сохранения данных в {filepath}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return  # Не повторяем для других ошибок
    
    def _call_bots_api(self, endpoint: str, method: str = 'GET', data: Dict = None, silent: bool = False) -> Optional[Dict]:
        """
        Вызов API bots.py (неблокирующий)
        
        Args:
            endpoint: API endpoint
            method: HTTP метод
            data: Данные для POST запроса
            silent: Если True, не логирует предупреждения (для фоновых попыток)
        """
        try:
            url = f"{self.bots_service_url}{endpoint}"
            
            # Короткий таймаут для быстрого ответа
            timeout = 3 if silent else 5
            
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                if not silent:
                    logger.debug(f"⚠️ API {endpoint} вернул статус {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            # Не логируем предупреждения для фоновых попыток
            if not silent:
                logger.debug(f"⚠️ Сервис bots.py недоступен по адресу {self.bots_service_url} (продолжаем работу)")
            return None
        except requests.exceptions.Timeout:
            if not silent:
                logger.debug(f"⏳ Таймаут подключения к bots.py (продолжаем работу)")
            return None
        except Exception as e:
            if not silent:
                logger.debug(f"⚠️ Ошибка вызова API {endpoint}: {e}")
            return None
    
    def collect_bots_data(self) -> Dict:
        """
        Сбор данных из bots.py
        
        Собирает:
        - Список ботов и их статусы
        - RSI данные для всех монет
        - Свечи
        - Сигналы блокировок
        """
        logger.debug("📊 Сбор данных из bots.py...")
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'bots': [],
            'rsi_data': {},
            'signals': {}
        }
        
        try:
            # Получаем список ботов (неблокирующий вызов)
            bots_response = self._call_bots_api('/api/bots/list', silent=True)
            if bots_response and bots_response.get('success'):
                collected_data['bots'] = bots_response.get('bots', [])
            
            # Получаем RSI данные для монет (неблокирующий вызов)
            rsi_response = self._call_bots_api('/api/bots/coins-with-rsi', silent=True)
            if rsi_response and rsi_response.get('success'):
                collected_data['rsi_data'] = rsi_response.get('coins', {})
            
            # Получаем статус ботов (неблокирующий вызов)
            status_response = self._call_bots_api('/api/bots/status', silent=True)
            if status_response and status_response.get('success'):
                collected_data['bots_status'] = status_response.get('status', {})
            
            # Сохраняем данные
            existing_data = self._load_data(self.bots_data_file)
            if 'history' not in existing_data:
                existing_data['history'] = []
            
            existing_data['history'].append(collected_data)
            
            # Ограничиваем историю (последние 1000 записей)
            if len(existing_data['history']) > 1000:
                existing_data['history'] = existing_data['history'][-1000:]
            
            existing_data['last_update'] = datetime.now().isoformat()
            existing_data['latest'] = collected_data
            
            self._save_data(self.bots_data_file, existing_data)
            
            logger.debug(f"✅ Собрано данных: {len(collected_data.get('bots', []))} ботов, {len(collected_data.get('rsi_data', {}))} монет с RSI")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных из bots.py: {e}")
        
        return collected_data
    
    def collect_history_data(self) -> Dict:
        """
        Сбор данных из bot_history.py
        
        Собирает:
        - Историю трейдов
        - Статистику торговли
        - Закрытые позиции с PnL
        """
        logger.debug("📊 Сбор данных из bot_history...")
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'trades': [],
            'statistics': {}
        }
        
        # ВАЖНО: Загружаем напрямую из data/bot_history.json
        try:
            bot_history_file = os.path.join('data', 'bot_history.json')
            if os.path.exists(bot_history_file):
                with open(bot_history_file, 'r', encoding='utf-8') as f:
                    bot_history_data = json.load(f)
                
                # Извлекаем сделки из bot_history.json
                bot_trades = bot_history_data.get('trades', [])
                if bot_trades:
                    collected_data['trades'].extend(bot_trades)
                    logger.debug(f"📊 Загружено {len(bot_trades)} сделок напрямую из bot_history.json")
        except json.JSONDecodeError as json_error:
            logger.warning(f"⚠️ Файл bot_history.json поврежден (JSON ошибка на позиции {json_error.pos})")
            logger.info("🗑️ Удаляем поврежденный файл, bots.py пересоздаст его автоматически")
            try:
                os.remove(bot_history_file)
                logger.info("✅ Поврежденный файл удален")
            except Exception as del_error:
                logger.debug(f"⚠️ Не удалось удалить файл: {del_error}")
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки bot_history.json: {e}")
        
        try:
            # Получаем историю сделок через API (дополняем прямую загрузку) - неблокирующий вызов
            trades_response = self._call_bots_api('/api/bots/trades?limit=1000', silent=True)
            if trades_response and trades_response.get('success'):
                api_trades = trades_response.get('trades', [])
                # Объединяем с уже загруженными из bot_history.json (избегаем дубликатов)
                existing_ids = {t.get('id') for t in collected_data['trades'] if t.get('id')}
                for trade in api_trades:
                    trade_id = trade.get('id') or trade.get('timestamp')
                    if trade_id not in existing_ids:
                        collected_data['trades'].append(trade)
            
            # Получаем статистику - неблокирующий вызов
            stats_response = self._call_bots_api('/api/bots/statistics', silent=True)
            if stats_response and stats_response.get('success'):
                collected_data['statistics'] = stats_response.get('statistics', {})
            
            # Получаем историю действий - неблокирующий вызов
            history_response = self._call_bots_api('/api/bots/history?limit=500', silent=True)
            if history_response and history_response.get('success'):
                collected_data['actions'] = history_response.get('history', [])
            
            # Сохраняем данные
            existing_data = self._load_data(self.history_data_file)
            if 'history' not in existing_data:
                existing_data['history'] = []
            
            existing_data['history'].append(collected_data)
            
            # Ограничиваем историю
            if len(existing_data['history']) > 1000:
                existing_data['history'] = existing_data['history'][-1000:]
            
            existing_data['last_update'] = datetime.now().isoformat()
            existing_data['latest'] = collected_data
            
            self._save_data(self.history_data_file, existing_data)
            
            trades_count = len(collected_data.get('trades', []))
            logger.debug(f"✅ Собрано данных: {trades_count} сделок")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных из bot_history: {e}")
        
        return collected_data
    
    def load_full_candles_history(self, force_reload: bool = False) -> bool:
        """
        Загружает ВСЕ доступные свечи для всех монет
        
        Использует AICandlesLoader для загрузки максимального количества свечей
        (до 2000-5000 свечей на монету с пагинацией для качественного обучения)
        Минимум 2000 свечей на монету для качественного обучения (~1 год истории на 6H)
        
        Args:
            force_reload: Если True, загружает заново даже если файл существует
        
        Returns:
            True если успешно загружено или файл уже существует и актуален
        """
        try:
            from bot_engine.ai.ai_candles_loader import AICandlesLoader
            
            # Проверяем существование файла и время последнего обновления
            full_history_file = os.path.join('data', 'ai', 'candles_full_history.json')
            
            if not force_reload and os.path.exists(full_history_file):
                try:
                    # Проверяем время последнего обновления файла
                    file_mtime = os.path.getmtime(full_history_file)
                    file_age_seconds = time.time() - file_mtime
                    file_age_hours = file_age_seconds / 3600
                    
                    # Если файл обновлен менее часа назад - используем его без перезагрузки
                    if file_age_hours < 1.0:
                        logger.debug(f"✅ Файл свечей актуален ({file_age_hours:.1f}ч назад)")
                        return True
                    else:
                        logger.debug(f"🔄 Инкрементальное обновление ({file_age_hours:.1f}ч назад)")
                except Exception as check_error:
                    logger.debug(f"⚠️ Ошибка проверки файла: {check_error}")
                    # Продолжаем загрузку если не удалось проверить файл
            
            # Сокращенные логи
            logger.debug("📊 Загрузка свечей для AI...")
            
            # ВАЖНО: Инициализируем биржу напрямую, как в bots.py
            # Это позволяет ai.py работать независимо от bots.py
            exchange = None
            
            # Сначала пробуем получить из bots.py (если он запущен)
            try:
                from bots_modules.imports_and_globals import get_exchange
                exchange = get_exchange()
                if exchange:
                    logger.debug("✅ Биржа получена из bots.py")
            except Exception as e:
                logger.debug(f"⏳ Не удалось получить биржу из bots.py: {e}")
            
            # Если не получилось - инициализируем напрямую
            if not exchange:
                try:
                    logger.debug("💡 Инициализация биржи напрямую...")
                    from exchanges.exchange_factory import ExchangeFactory
                    from app.config import EXCHANGES
                    
                    exchange = ExchangeFactory.create_exchange(
                        'BYBIT',
                        EXCHANGES['BYBIT']['api_key'],
                        EXCHANGES['BYBIT']['api_secret']
                    )
                    
                    if exchange:
                        logger.debug("✅ Биржа инициализирована")
                    else:
                        logger.error("❌ ExchangeFactory вернул None")
                        return False
                except Exception as init_error:
                    logger.error(f"❌ Ошибка инициализации биржи: {init_error}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return False
            
            if not exchange:
                logger.error("❌ Не удалось получить объект биржи, проверьте API ключи")
                return False
            
            logger.debug("🚀 Начинаем загрузку свечей (может занять несколько минут)...")
            
            loader = AICandlesLoader(exchange_obj=exchange)
            success = loader.load_all_candles_full_history(max_workers=10)
            
            if success:
                logger.info("✅ История свечей загружена")
            else:
                logger.warning("⚠️ Загрузка свечей не завершена, проверьте логи")
            
            return success
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ ОШИБКА ЗАГРУЗКИ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ")
            logger.error("=" * 80)
            logger.error(f"   Ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            return False
    
    def collect_market_data(self) -> Dict:
        """
        Сбор рыночных данных ТОЛЬКО из полной истории свечей
        
        ВАЖНО: Использует ТОЛЬКО data/ai/candles_full_history.json
        Если файла нет - возвращает пустые данные (не использует candles_cache.json!)
        Файл должен быть загружен через load_full_candles_history() перед использованием
        """
        # Сокращенные логи
        logger.debug("📊 Сбор рыночных данных...")
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'candles': {},
            'indicators': {}
        }
        
        try:
            # ВАЖНО: Используем ТОЛЬКО полную историю свечей (data/ai/candles_full_history.json)
            # НЕ используем candles_cache.json - только полная история!
            full_history_file = os.path.join('data', 'ai', 'candles_full_history.json')
            candles_data = {}
            
            if not os.path.exists(full_history_file):
                logger.warning("⚠️ Файл candles_full_history.json не найден, ожидаем загрузки...")
                return collected_data
            
            # Читаем ТОЛЬКО из полной истории свечей
            try:
                logger.debug(f"📖 Чтение {full_history_file}...")
                
                with open(full_history_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                # Извлекаем свечи из структуры с метаданными
                if 'candles' in full_data:
                    candles_data = full_data['candles']
                elif isinstance(full_data, dict) and not full_data.get('metadata'):
                    candles_data = full_data
                else:
                    logger.warning("⚠️ Неожиданная структура файла candles_full_history.json")
                    candles_data = {}
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"❌ Файл candles_full_history.json поврежден (позиция {json_error.pos}), удаляем...")
                try:
                    os.remove(full_history_file)
                    logger.info("✅ Поврежденный файл удален")
                except Exception as del_error:
                    logger.debug(f"⚠️ Не удалось удалить файл: {del_error}")
                candles_data = {}
            except Exception as e:
                logger.error(f"❌ Ошибка чтения полной истории свечей: {e}")
                import traceback
                logger.error(traceback.format_exc())
                candles_data = {}
            
            # Обрабатываем свечи
            if candles_data:
                candles_count = 0
                total_candles = 0
                
                for symbol, candle_info in candles_data.items():
                    try:
                        candles = candle_info.get('candles', [])
                        if candles and len(candles) > 0:
                            # ВАЖНО: Используем ВСЕ свечи без ограничений!
                            # НЕ обрезаем до 1000 свечей - используем все что есть
                            candles_list = candles if isinstance(candles, list) else []
                            
                            collected_data['candles'][symbol] = {
                                'candles': candles_list,  # ВСЕ свечи без ограничений
                                'count': len(candles_list),
                                'timeframe': candle_info.get('timeframe', '6h'),
                                'last_update': candle_info.get('last_update') or candle_info.get('loaded_at'),
                                'source': full_history_file,  # ВСЕГДА из полной истории
                                'is_full_history': True  # ВСЕГДА полная история
                            }
                            candles_count += 1
                            total_candles += len(candles_list)
                            
                            # Логируем если свечей больше 1000 (полная история) или меньше (кэш)
                            if len(candles_list) > 1000:
                                logger.debug(f"📊 {symbol}: {len(candles_list)} свечей (полная история)")
                            elif len(candles_list) <= 1000:
                                logger.debug(f"📊 {symbol}: {len(candles_list)} свечей (возможно кэш, не полная история)")
                            
                            # Логируем каждые 100 монет
                            if candles_count % 100 == 0:
                                logger.debug(f"📊 Обработано свечей: {candles_count} монет...")
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка обработки свечей для {symbol}: {e}")
                        continue
                
                logger.info(f"✅ Обработано свечей: {candles_count} монет, {total_candles} свечей всего")
            else:
                logger.warning(f"⚠️ Файл {full_history_file} не найден или пуст")
            
            # 2. Получаем индикаторы через API (RSI, тренды, сигналы)
            rsi_response = self._call_bots_api('/api/bots/coins-with-rsi')
            if rsi_response and rsi_response.get('success'):
                coins_data = rsi_response.get('coins', {})
                
                logger.info(f"📊 Получено индикаторов для {len(coins_data)} монет")
                
                # Сохраняем индикаторы
                indicators_count = 0
                for symbol, coin_data in coins_data.items():
                    try:
                        collected_data['indicators'][symbol] = {
                            'rsi': coin_data.get('rsi6h'),
                            'trend': coin_data.get('trend6h'),
                            'signal': coin_data.get('signal'),
                            'price': coin_data.get('price'),
                            'volume': coin_data.get('volume'),
                            'stochastic': coin_data.get('stochastic'),
                            'stoch_rsi_k': coin_data.get('stoch_rsi_k'),
                            'stoch_rsi_d': coin_data.get('stoch_rsi_d'),
                            'enhanced_rsi': coin_data.get('enhanced_rsi'),
                            'trend_analysis': coin_data.get('trend_analysis'),
                            'time_filter_info': coin_data.get('time_filter_info'),
                            'exit_scam_info': coin_data.get('exit_scam_info'),
                            'source': 'coins_rsi_data'
                        }
                        indicators_count += 1
                        
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка обработки индикаторов для {symbol}: {e}")
                        continue
                
                logger.debug(f"✅ Индикаторы: {indicators_count} монет")
            
            # Итоговая статистика (кратко)
            logger.debug(f"📊 Данные собраны: {len(collected_data['candles'])} монет со свечами, {len(collected_data['indicators'])} с индикаторами")
            
            # ВАЖНО: НЕ сохраняем свечи в market_data.json - они уже в candles_full_history.json!
            # Сохраняем только индикаторы для быстрого доступа (опционально)
            # Но свечи всегда берутся напрямую из candles_full_history.json
            # Поэтому market_data.json больше не нужен - удаляем сохранение
            # (Индикаторы можно получать через API каждый раз, свечи - из candles_full_history.json)
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора рыночных данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return collected_data
    
    def get_training_data(self) -> Dict:
        """
        Получить данные для обучения
        
        ВАЖНО: Свечи берутся напрямую из candles_full_history.json
        market_data.json больше не используется для свечей!
        
        Returns:
            Словарь с данными для обучения
        """
        return {
            # market_data.json больше не используется - свечи из candles_full_history.json
            'bots_data': self._load_data(self.bots_data_file),
            'history_data': self._load_data(self.history_data_file)
        }
    
    def get_latest_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Получить последние рыночные данные для символа
        
        ВАЖНО: Свечи берутся напрямую из candles_full_history.json
        market_data.json больше не используется!
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Словарь с данными или None
        """
        # Свечи из полной истории
        full_history_file = os.path.join('data', 'ai', 'candles_full_history.json')
        candles = None
        
        if os.path.exists(full_history_file):
            try:
                with open(full_history_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                candles_data = {}
                if 'candles' in full_data:
                    candles_data = full_data['candles']
                elif isinstance(full_data, dict) and not full_data.get('metadata'):
                    candles_data = full_data
                
                if symbol in candles_data:
                    candle_info = candles_data[symbol]
                    candles = candle_info.get('candles', []) if isinstance(candle_info, dict) else []
            except Exception as e:
                logger.debug(f"⚠️ Ошибка чтения candles_full_history.json для {symbol}: {e}")
        
        # Индикаторы через API
        indicators = None
        rsi_response = self._call_bots_api('/api/bots/coins-with-rsi', silent=True)
        if rsi_response and rsi_response.get('success'):
            coins_data = rsi_response.get('coins', {})
            if symbol in coins_data:
                indicators = {
                    'rsi': coins_data[symbol].get('rsi6h'),
                    'trend': coins_data[symbol].get('trend6h'),
                    'signal': coins_data[symbol].get('signal'),
                    'price': coins_data[symbol].get('price'),
                    'volume': coins_data[symbol].get('volume')
                }
        
        if candles or indicators:
            return {
                'candles': candles,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            }
        
        return None

