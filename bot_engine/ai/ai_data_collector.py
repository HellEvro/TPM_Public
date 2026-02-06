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
from bot_engine.config_loader import get_current_timeframe

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
        
        # Файлы больше не используются - все данные в БД
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            pass
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
        
        logger.info("✅ AIDataCollector инициализирован")
    
    def _load_data(self, filepath: str) -> Dict:
        """Загрузить данные из файла"""
        try:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError as json_error:
                    # Больше не удаляем рабочие файлы — создаём резервную копию и пытаемся восстановить
                    logger.warning(f"⚠️ Файл {filepath} не прочитан (JSON ошибка на позиции {json_error.pos}).")
                    backup_file = f"{filepath}.backup"
                    corrupted_file = f"{filepath}.corrupted"

                    # Пытаемся прочитать резервную копию, если есть
                    if os.path.exists(backup_file):
                        try:
                            with open(backup_file, 'r', encoding='utf-8') as backup:
                                logger.info(f"   ✅ Используем резервную копию {backup_file}")
                                return json.load(backup)
                        except Exception as backup_error:
                            pass

                    # Сохраняем текущую версию как .corrupted для ручного анализа
                    try:
                        import shutil
                        shutil.copy2(filepath, corrupted_file)
                        logger.info(f"   📁 Сохранен проблемный файл: {corrupted_file}")
                    except Exception as copy_error:
                        pass

                    # Возвращаем пустой dict, но основной файл не трогаем
                    return {}
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных из {filepath}: {e}")
            import traceback
            pass
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
                    pass
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"⚠️ Не удалось сохранить {filepath} после {max_retries} попыток (файл занят другим процессом)")
                    pass
            except OSError as os_error:
                # Другие ошибки ОС (WinError 32 и т.д.)
                if attempt < max_retries - 1:
                    pass
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.warning(f"⚠️ Не удалось сохранить {filepath} после {max_retries} попыток")
                    pass
            except Exception as e:
                # Другие ошибки
                logger.error(f"❌ Ошибка сохранения данных в {filepath}: {e}")
                import traceback
                pass
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
                    pass
                return None
                
        except requests.exceptions.ConnectionError:
            # Не логируем предупреждения для фоновых попыток
            if not silent:
                pass
            return None
        except requests.exceptions.Timeout:
            if not silent:
                pass
            return None
        except Exception as e:
            if not silent:
                pass
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
        # Убрано: logger.debug("📊 Сбор данных из bots.py...") - слишком шумно
        
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
            
            # Сохраняем ТОЛЬКО в БД
            if not self.ai_db:
                logger.error("❌ AI Database не подключена!")
                return collected_data
            
            # ВАЖНО: Снапшоты больше не сохраняются!
            # Данные ботов уже есть в нормализованных таблицах:
            # - bots_data.db → bots (текущее состояние ботов)
            # - bots_data.db → rsi_cache_coins (RSI данные)
            # Снапшоты - это избыточное дублирование данных!
            try:
                # Не сохраняем снапшоты - данные уже в нормализованных таблицах
                pass
            except Exception as db_error:
                logger.error(f"❌ Ошибка сохранения в БД: {db_error}")
                import traceback
                logger.error(traceback.format_exc())
            
            logger.info(f"✅ Собрано данных: {len(collected_data.get('bots', []))} ботов, {len(collected_data.get('rsi_data', {}))} монет с RSI")
            
            # Обновляем статус data-service в БД
            self.update_data_service_status(
                last_collection=datetime.now().isoformat(),
                trades=len(collected_data.get('bots', [])),
                ready=True
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных из bots.py: {e}")
        
        return collected_data
    
    def collect_history_data(self) -> Dict:
        """
        Сбор данных из БД (приоритет) или bot_history.json (fallback)
        
        Собирает:
        - Историю трейдов
        - Статистику торговли
        - Закрытые позиции с PnL
        """
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'trades': [],
            'statistics': {}
        }
        
        # ПРИОРИТЕТ 1: Загружаем из БД (ai_database)
        try:
            from bot_engine.ai.ai_database import get_ai_database
            ai_db = get_ai_database()
            if ai_db:
                # Загружаем сделки ботов из БД
                # ВАЖНО: min_trades=0 чтобы получить ВСЕ сделки, не только для символов с >=10 сделками
                # ВАЖНО: Загружаем ВСЕ сделки - и реальные, и симуляции
                # Симуляции нужны для обучения ИИ на разных параметрах
                db_trades = ai_db.get_trades_for_training(
                    include_simulated=True,  # ВКЛЮЧАЕМ симуляции для обучения!
                    include_real=True,
                    include_exchange=True,  # ВАЖНО: Включаем сделки с биржи тоже!
                    min_trades=0,  # КРИТИЧНО: 0 чтобы получить все сделки, не фильтровать по символам
                    limit=None
                )
                
                if db_trades:
                    # Конвертируем формат БД в формат для коллектора
                    for trade in db_trades:
                        # get_trades_for_training возвращает данные с полями timestamp, close_timestamp
                        # но может быть и entry_time, exit_time в зависимости от источника
                        converted_trade = {
                            'id': trade.get('trade_id') or trade.get('id') or f"db_{trade.get('symbol')}_{trade.get('timestamp', '')}",
                            'timestamp': trade.get('timestamp') or trade.get('entry_time'),
                            'bot_id': trade.get('bot_id', trade.get('symbol')),
                            'symbol': trade.get('symbol'),
                            'direction': trade.get('direction'),
                            'entry_price': trade.get('entry_price'),
                            'exit_price': trade.get('exit_price'),
                            'pnl': trade.get('pnl'),
                            'roi': trade.get('roi'),
                            'status': trade.get('status', 'CLOSED'),
                            'close_timestamp': trade.get('close_timestamp') or trade.get('exit_time'),
                            'decision_source': trade.get('decision_source', 'SCRIPT'),
                            'is_simulated': trade.get('is_simulated', False) or (trade.get('source') == 'SIMULATED'),
                            'is_real': trade.get('is_real', True) and (trade.get('source') != 'SIMULATED')
                        }
                        collected_data['trades'].append(converted_trade)
                    
                    logger.info(f"✅ История трейдов: {len(db_trades)} сделок (загружено из БД)")
                else:
                    logger.warning(f"⚠️ История трейдов: 0 сделок в БД (проверьте наличие сделок в таблицах bot_trades, exchange_trades)")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки из БД: {e}, используем fallback")
            import traceback
            pass
        
        # FALLBACK: Загружаем из bot_history.json только если БД пуста
        if not collected_data['trades']:
            try:
                bot_history_file = os.path.join('data', 'bot_history.json')
                if os.path.exists(bot_history_file):
                    import shutil
                    snapshot_file = f"{bot_history_file}.snapshot"
                    try:
                        shutil.copy2(bot_history_file, snapshot_file)
                        with open(snapshot_file, 'r', encoding='utf-8') as f:
                            bot_history_data = json.load(f)
                    finally:
                        try:
                            if os.path.exists(snapshot_file):
                                os.remove(snapshot_file)
                        except Exception:
                            pass
                    
                    # Извлекаем сделки из bot_history.json
                    bot_trades = bot_history_data.get('trades', [])
                    if bot_trades:
                        collected_data['trades'].extend(bot_trades)
                        pass
            except json.JSONDecodeError as json_error:
                pass
            except Exception as e:
                pass
        
        try:
            # Получаем историю сделок через API (дополняем загрузку из БД) - неблокирующий вызов
            # ВАЖНО: Обернуто в try-except чтобы не блокировать выполнение
            try:
                trades_response = self._call_bots_api('/api/bots/trades?limit=1000', silent=True)
                if trades_response and trades_response.get('success'):
                    api_trades = trades_response.get('trades', [])
                    # Объединяем с уже загруженными из БД (избегаем дубликатов)
                    existing_ids = {t.get('id') for t in collected_data['trades'] if t.get('id')}
                    for trade in api_trades:
                        trade_id = trade.get('id') or trade.get('timestamp')
                        if trade_id not in existing_ids:
                            collected_data['trades'].append(trade)
            except Exception as api_error:
                pass
            
            # Получаем статистику - неблокирующий вызов
            try:
                stats_response = self._call_bots_api('/api/bots/statistics', silent=True)
                if stats_response and stats_response.get('success'):
                    collected_data['statistics'] = stats_response.get('statistics', {})
            except Exception as api_error:
                pass
            
            # Получаем историю действий - неблокирующий вызов
            try:
                history_response = self._call_bots_api('/api/bots/history?limit=500', silent=True)
                if history_response and history_response.get('success'):
                    collected_data['actions'] = history_response.get('history', [])
            except Exception as api_error:
                pass
            
            trades_count = len(collected_data.get('trades', []))
            
            # Обновляем статус data-service в БД (с обработкой ошибок)
            try:
                self.update_data_service_status(
                    trades=trades_count,
                    history_loaded=True
                )
            except Exception as status_error:
                pass
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных из bot_history: {e}")
            import traceback
            pass
        
        return collected_data
    
    def load_full_candles_history(self, force_reload: bool = False) -> bool:
        """
        Загружает ВСЕ доступные свечи для всех монет в БД
        
        Использует AICandlesLoader для загрузки максимального количества свечей
        (загружаются все доступные свечи, но для обучения используется максимум 1000 последних свечей)
        Для обучения ИИ достаточно 1000 свечей (~250 дней истории на 6H)
        
        ВАЖНО: Все свечи сохраняются в БД (таблица candles_history), файлы не используются!
        
        Args:
            force_reload: Если True, загружает заново даже если в БД уже есть данные
        
        Returns:
            True если успешно загружено или в БД уже есть актуальные данные
        """
        try:
            from bot_engine.ai.ai_candles_loader import AICandlesLoader
            
            # Проверяем БД вместо файла
            if not force_reload:
                try:
                    from bot_engine.ai.ai_database import get_ai_database
                    ai_db = get_ai_database()
                    if ai_db:
                        # Проверяем количество свечей в БД
                        candles_count = ai_db.count_candles()
                        if candles_count > 0:
                            # Проверяем время последнего обновления (берем максимальный created_at)
                            with ai_db._get_connection() as conn:
                                cursor = conn.cursor()
                                from bot_engine.config_loader import get_current_timeframe
                                cursor.execute("""
                                    SELECT MAX(created_at) as last_update
                                    FROM candles_history
                                    WHERE timeframe = ?
                                """, (get_current_timeframe(),))
                                row = cursor.fetchone()
                                if row and row['last_update']:
                                    from datetime import datetime
                                    last_update_str = row['last_update']
                                    try:
                                        last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                                        now = datetime.now(last_update.tzinfo) if last_update.tzinfo else datetime.now()
                                        age_seconds = (now - last_update.replace(tzinfo=None)).total_seconds() if not last_update.tzinfo else (now - last_update).total_seconds()
                                        age_hours = age_seconds / 3600
                                        
                                        # Если данные обновлены менее часа назад - используем БД без перезагрузки
                                        if age_hours < 1.0:
                                            pass
                                            return True
                                    except Exception:
                                        pass
                            
                            pass
                except Exception as check_error:
                    pass
                    # Продолжаем загрузку если не удалось проверить БД
            
            # Сокращенные логи
            logger.info("📊 Загрузка свечей для AI...")
            
            # ВАЖНО: Инициализируем биржу напрямую, как в bots.py
            # Это позволяет ai.py работать независимо от bots.py
            exchange = None
            
            # Сначала пробуем получить из bots.py (если он запущен)
            try:
                from bots_modules.imports_and_globals import get_exchange
                exchange = get_exchange()
                if exchange:
                    pass
            except Exception as e:
                pass
            
            # Если не получилось - инициализируем напрямую
            if not exchange:
                try:
                    logger.info("💡 Инициализация биржи напрямую...")
                    from exchanges.exchange_factory import ExchangeFactory
                    from app.config import EXCHANGES
                    
                    exchange = ExchangeFactory.create_exchange(
                        'BYBIT',
                        EXCHANGES['BYBIT']['api_key'],
                        EXCHANGES['BYBIT']['api_secret']
                    )
                    
                    if exchange:
                        logger.info("✅ Биржа инициализирована")
                    else:
                        logger.error("❌ ExchangeFactory вернул None")
                        return False
                except Exception as init_error:
                    logger.error(f"❌ Ошибка инициализации биржи: {init_error}")
                    import traceback
                    pass
                    return False
            
            if not exchange:
                logger.error("❌ Не удалось получить объект биржи, проверьте API ключи")
                return False
            
            logger.info("🚀 Начинаем загрузку свечей (может занять несколько минут)...")
            
            loader = AICandlesLoader(exchange_obj=exchange)
            success = loader.load_all_candles_full_history()  # max_workers из AILauncherConfig при ограничении ОЗУ
            
            if success:
                logger.info("✅ История свечей загружена")
                # Обновляем статус data-service в БД
                try:
                    from bot_engine.ai.ai_database import get_ai_database
                    ai_db = get_ai_database()
                    if ai_db:
                        candles_count = ai_db.count_candles()
                        self.update_data_service_status(
                            candles=candles_count,
                            history_loaded=True
                        )
                except Exception as status_error:
                    pass
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
        Сбор рыночных данных ТОЛЬКО из БД
        
        ВАЖНО: Использует ТОЛЬКО БД (таблица candles_history)
        Если БД пуста - возвращает пустые данные
        Свечи должны быть загружены через load_full_candles_history() перед использованием
        """
        # Сокращенные логи
        # Убрано: logger.debug("📊 Сбор рыночных данных...") - слишком шумно
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'candles': {},
            'indicators': {}
        }
        
        try:
            # Загружаем ТОЛЬКО из БД
            candles_data = {}
            try:
                from bot_engine.ai.ai_database import get_ai_database
                ai_db = get_ai_database()
                if not ai_db:
                    logger.warning("⚠️ AI Database не доступна")
                    return collected_data
                
                # Ограничиваем загрузку (при AI_MEMORY_LIMIT_MB лимиты из AILauncherConfig)
                try:
                    from bot_engine.ai.ai_launcher_config import AILauncherConfig
                    _max_sym = AILauncherConfig.MAX_SYMBOLS_FOR_CANDLES
                    _max_candles = AILauncherConfig.MAX_CANDLES_PER_SYMBOL
                except Exception:
                    _max_sym, _max_candles = 50, 1000
                candles_data = ai_db.get_all_candles_dict(
                    timeframe=get_current_timeframe(),
                    max_symbols=_max_sym,
                    max_candles_per_symbol=_max_candles
                )
                if candles_data and len(candles_data) > 0:
                    total_candles = sum(len(c) for c in candles_data.values())
                    logger.info(f"✅ Загружено {len(candles_data)} монет из БД ({total_candles:,} свечей, ограничено для экономии памяти)")
                else:
                    logger.warning("⚠️ БД пуста или get_all_candles_dict вернул пустой результат, ожидаем загрузки свечей...")
                    pass
            except Exception as db_error:
                logger.error(f"❌ Ошибка загрузки из БД: {db_error}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Обрабатываем свечи
            if candles_data:
                candles_count = 0
                total_candles = 0
                
                for symbol, candles_list in candles_data.items():
                    try:
                        # ВАЖНО: get_all_candles_dict() возвращает {symbol: [candles]}, а не {symbol: {'candles': [...]}}
                        if candles_list and len(candles_list) > 0:
                            # ВАЖНО: Используем ВСЕ свечи без ограничений!
                            # НЕ обрезаем до 1000 свечей - используем все что есть
                            if not isinstance(candles_list, list):
                                candles_list = []
                            
                            collected_data['candles'][symbol] = {
                                'candles': candles_list,  # ВСЕ свечи без ограничений
                                'count': len(candles_list),
                                'timeframe': get_current_timeframe(),  # Текущий таймфрейм
                                'last_update': None,  # БД не хранит last_update для каждой монеты
                                'source': 'ai_data.db',  # ВСЕГДА из БД
                                'is_full_history': True  # ВСЕГДА полная история
                            }
                            candles_count += 1
                            total_candles += len(candles_list)
                            
                            # Логируем если свечей больше 1000 (полная история) или меньше (кэш)
                            if len(candles_list) > 1000:
                                # Убрано: logger.debug(f"📊 {symbol}: {len(candles_list)} свечей (полная история)") - слишком шумно
                                pass
                            # Убрано: elif len(candles_list) <= 1000: logger.debug(...) - слишком шумно
                            
                            # Убрано: логирование каждые 100 монет - слишком шумно
                    except Exception as e:
                        pass
                        continue
                
                logger.info(f"✅ Обработано свечей: {candles_count} монет, {total_candles} свечей всего")
                # Обновляем статус data-service в БД (с обработкой ошибок)
                try:
                    self.update_data_service_status(
                        candles=total_candles,
                        history_loaded=True
                    )
                except Exception as status_error:
                    pass
            else:
                logger.warning("⚠️ БД пуста, ожидаем загрузки свечей...")
            
            # 2. Получаем индикаторы через API (RSI, тренды, сигналы) - неблокирующий вызов
            try:
                rsi_response = self._call_bots_api('/api/bots/coins-with-rsi', silent=True)
                if rsi_response and rsi_response.get('success'):
                    coins_data = rsi_response.get('coins', {})
                    
                    logger.info(f"📊 Получено индикаторов для {len(coins_data)} монет")
                    
                    # Получаем RSI и тренд с учетом текущего таймфрейма
                    from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
                    
                    # Сохраняем индикаторы
                    indicators_count = 0
                    for symbol, coin_data in coins_data.items():
                        try:
                            collected_data['indicators'][symbol] = {
                                'rsi': get_rsi_from_coin_data(coin_data),
                                'trend': get_trend_from_coin_data(coin_data),
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
                            pass
                            continue
                    
                    # Убрано: logger.debug(f"✅ Индикаторы: {indicators_count} монет") - слишком шумно
                else:
                    pass
            except Exception as api_error:
                pass
                # Продолжаем работу без индикаторов - это не критично
            
            # Итоговая статистика (кратко)
            # Убрано: logger.debug(f"📊 Данные собраны: {len(collected_data['candles'])} монет со свечами, {len(collected_data['indicators'])} с индикаторами") - слишком шумно
            
            # ВАЖНО: Свечи хранятся ТОЛЬКО в БД (таблица candles_history)!
            # Сохраняем только индикаторы для быстрого доступа (опционально)
            # Свечи всегда берутся из БД через ai_db.get_all_candles_dict()
            # Файлы больше не используются - все данные в БД
            # (Индикаторы можно получать через API каждый раз, свечи - из БД)
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора рыночных данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return collected_data
    
    def _get_bots_data(self) -> Dict:
        """
        Получает данные ботов из нормализованных таблиц
        
        ВАЖНО: Снапшоты больше не используются!
        Данные берутся напрямую из:
        - bots_data.db → bots (текущее состояние ботов)
        - bots_data.db → rsi_cache_coins (RSI данные)
        
        Returns:
            Словарь с данными ботов
        """
        result = {
            'history': [],
            'last_update': None,
            'latest': {}
        }
        
        # Загружаем напрямую из нормализованных таблиц
        try:
            from bot_engine.bots_database import get_bots_database
            bots_db = get_bots_database()
            
            # Загружаем текущее состояние ботов
            bots_state = bots_db.load_bots_state()
            bots_data = bots_state.get('bots', {})
            
            # Загружаем RSI данные
            rsi_cache = bots_db.load_rsi_cache(max_age_hours=6.0)
            rsi_data = rsi_cache.get('coins', {}) if rsi_cache else {}
            
            # Формируем результат
            result['latest'] = {
                'bots': bots_data,
                'rsi_data': rsi_data,
                'timestamp': datetime.now().isoformat()
            }
            result['last_update'] = result['latest']['timestamp']
            
        except Exception as db_error:
            logger.error(f"❌ Ошибка загрузки данных ботов из нормализованных таблиц: {db_error}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    def get_training_data(self) -> Dict:
        """
        Получить данные для обучения
        
        ВАЖНО: Свечи берутся ТОЛЬКО из БД (таблица candles_history)!
        Файлы больше не используются - все данные в БД!
        
        Returns:
            Словарь с данными для обучения
        """
        return {
            # Все данные из БД - файлы не используются
            'bots_data': self._get_bots_data(),
            'history_data': {}  # history_data.json больше не используется - все данные в БД
        }
    
    def get_latest_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Получить последние рыночные данные для символа
        
        ВАЖНО: Свечи берутся ТОЛЬКО из БД (таблица candles_history)!
        Файлы больше не используются!
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Словарь с данными или None
        """
        # Загружаем ТОЛЬКО из БД
        candles = None
        try:
            from bot_engine.ai.ai_database import get_ai_database
            ai_db = get_ai_database()
            if not ai_db:
                logger.warning(f"⚠️ AI Database не доступна для {symbol}")
                return None
            
            from bot_engine.config_loader import get_current_timeframe
            from bot_engine.config_loader import get_current_timeframe
            candles = ai_db.get_candles(symbol, timeframe=get_current_timeframe())
        except Exception as db_error:
            logger.error(f"❌ Ошибка загрузки свечей из БД для {symbol}: {db_error}")
        
        # Индикаторы через API
        indicators = None
        rsi_response = self._call_bots_api('/api/bots/coins-with-rsi', silent=True)
        if rsi_response and rsi_response.get('success'):
            coins_data = rsi_response.get('coins', {})
            if symbol in coins_data:
                # Получаем RSI и тренд с учетом текущего таймфрейма
                from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
                indicators = {
                    'rsi': get_rsi_from_coin_data(coins_data[symbol]),
                    'trend': get_trend_from_coin_data(coins_data[symbol]),
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
    
    def update_data_service_status(self, **kwargs):
        """
        Обновить статус data-service в БД
        
        ВАЖНО: Использует БД вместо файла data_service.json!
        
        Args:
            **kwargs: Поля статуса для обновления
        """
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна, статус не обновлен")
            return
        
        try:
            # Используем блокировку для предотвращения deadlock
            with self.lock:
                # Получаем текущий статус из БД
                current_status = self.ai_db.get_data_service_status('data_service')
                if current_status and current_status.get('status'):
                    status = current_status['status']
                else:
                    status = {}
                
                # Обновляем статус
                status.update(kwargs)
                status['timestamp'] = datetime.now().isoformat()
                
                # Сохраняем в БД
                self.ai_db.save_data_service_status('data_service', status)
                pass
        except Exception as e:
            pass
            # НЕ логируем как ERROR, чтобы не засорять логи - это не критично
    
    def get_data_service_status(self) -> Optional[Dict]:
        """
        Получить статус data-service из БД
        
        ВАЖНО: Использует БД вместо файла data_service.json!
        
        Returns:
            Словарь со статусом или None
        """
        if not self.ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return None
        
        try:
            result = self.ai_db.get_data_service_status('data_service')
            if result and result.get('status'):
                return result['status']
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса data-service: {e}")
            return None

