#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отдельная база данных для app.py (независимая от bots.py)

Хранит:
- История закрытых PnL (closed_pnl_history)

Это позволяет app.py работать независимо от bots.py
"""

import sqlite3
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from contextlib import contextmanager
import logging

logger = logging.getLogger('App.Database')


class AppDatabase:
    """
    Реляционная база данных для данных app.py
    """
    
    def __init__(self, db_path: str = None):
        """
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных (если None, используется data/app_data.db)
        """
        if db_path is None:
            base_dir = os.getcwd()
            db_path = os.path.join(base_dir, 'data', 'app_data.db')
            db_path = os.path.normpath(db_path)
        
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Создаем директорию если её нет
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        except OSError as e:
            logger.error(f"❌ Ошибка создания директории для БД: {e}")
            raise
        
        # Инициализируем базу данных
        self._init_database()
        
        logger.info(f"✅ App Database инициализирована: {db_path}")
    
    @contextmanager
    def _get_connection(self, retry_on_locked: bool = True, max_retries: int = 5):
        """
        Контекстный менеджер для работы с БД
        """
        last_error = None
        
        for attempt in range(max_retries if retry_on_locked else 1):
            try:
                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.row_factory = sqlite3.Row
                
                # Включаем WAL режим
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")  # 64MB кеш
                conn.execute("PRAGMA temp_store=MEMORY")
                
                try:
                    yield conn
                    conn.commit()
                    conn.close()
                    return
                except sqlite3.OperationalError as e:
                    error_str = str(e).lower()
                    # КРИТИЧНО: не делать continue — иначе "generator didn't stop after throw()"
                    if "database is locked" in error_str or "locked" in error_str:
                        conn.rollback()
                        conn.close()
                        last_error = e
                        raise
                    else:
                        conn.rollback()
                        conn.close()
                        raise
                except Exception as e:
                    try:
                        conn.rollback()
                    except:
                        pass
                    try:
                        conn.close()
                    except:
                        pass
                    raise e
                    
            except sqlite3.DatabaseError as e:
                error_str = str(e).lower()
                if "database disk image is malformed" in error_str or "malformed" in error_str:
                    logger.error(f"❌ КРИТИЧНО: БД повреждена: {e}")
                    raise
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 0.5
                    time.sleep(wait_time)
                    continue
                raise
        
        if last_error:
            raise last_error
    
    def _init_database(self):
        """Создает все таблицы и индексы"""
        db_exists = os.path.exists(self.db_path)
        
        if not db_exists:
            logger.info(f"📁 Создается новая база данных: {self.db_path}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # ==================== ТАБЛИЦА: ИСТОРИЯ ЗАКРЫТЫХ PNL ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS closed_pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    closed_pnl REAL NOT NULL,
                    close_time TEXT NOT NULL,
                    close_timestamp INTEGER NOT NULL,
                    exchange TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, close_timestamp, entry_price, exit_price)
                )
            """)
            
            # Индексы для closed_pnl_history
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_symbol ON closed_pnl_history(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_timestamp ON closed_pnl_history(close_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_exchange ON closed_pnl_history(exchange)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_time ON closed_pnl_history(close_time)")
            
            conn.commit()
            
            if not db_exists:
                logger.info("✅ Все таблицы и индексы созданы в новой базе данных")
            else:
                pass
    
    def save_closed_pnl_history(self, pnl_records: List[Dict]) -> int:
        """
        Сохраняет историю закрытых PnL в БД
        
        Args:
            pnl_records: Список записей PnL
        
        Returns:
            Количество сохраненных записей (новых, без дубликатов)
        """
        if not pnl_records:
            return 0
        
        saved_count = 0
        now = datetime.now().isoformat()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for record in pnl_records:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO closed_pnl_history 
                            (symbol, qty, entry_price, exit_price, closed_pnl, close_time, close_timestamp, exchange, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            record.get('symbol', ''),
                            record.get('qty', 0.0),
                            record.get('entry_price', 0.0),
                            record.get('exit_price', 0.0),
                            record.get('closed_pnl', 0.0),
                            record.get('close_time', ''),
                            record.get('close_timestamp', 0),
                            record.get('exchange', ''),
                            now
                        ))
                        
                        if cursor.rowcount > 0:
                            saved_count += 1
                    except Exception as e:
                        pass
                        continue
                
                conn.commit()
                
                if saved_count > 0:
                    pass
                
                return saved_count
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения истории PnL: {e}")
            import traceback
            pass
            return 0
    
    def load_closed_pnl_history(self, sort_by='time', period='all', start_date=None, end_date=None, exchange=None) -> List[Dict]:
        """
        Загружает историю закрытых PnL из БД
        
        Args:
            sort_by: Способ сортировки ('time' или 'pnl')
            period: Период фильтрации ('all', 'day', 'week', 'month', 'half_year', 'year', 'custom')
            start_date: Начальная дата для custom периода
            end_date: Конечная дата для custom периода
            exchange: Фильтр по бирже (если None, загружаются все)
        
        Returns:
            Список записей PnL в формате словарей
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Определяем диапазон дат
                end_time = int(time.time() * 1000)
                
                if period == 'custom' and start_date and end_date:
                    try:
                        if isinstance(start_date, str) and '-' in start_date:
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            start_time = int(start_dt.timestamp() * 1000)
                        else:
                            start_time = int(start_date)
                        
                        if isinstance(end_date, str) and '-' in end_date:
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            end_dt = end_dt.replace(hour=23, minute=59, second=59)
                            end_time = int(end_dt.timestamp() * 1000)
                        else:
                            end_time = int(end_date)
                    except Exception as e:
                        logger.error(f"Ошибка парсинга дат: {e}")
                        start_time = end_time - (30 * 24 * 60 * 60 * 1000)
                elif period == 'day':
                    start_time = end_time - (24 * 60 * 60 * 1000)
                elif period == 'week':
                    start_time = end_time - (7 * 24 * 60 * 60 * 1000)
                elif period == 'month':
                    start_time = end_time - (30 * 24 * 60 * 60 * 1000)
                elif period == 'half_year':
                    start_time = end_time - (180 * 24 * 60 * 60 * 1000)
                elif period == 'year':
                    start_time = end_time - (365 * 24 * 60 * 60 * 1000)
                else:  # period == 'all'
                    start_time = 0
                
                # Формируем SQL запрос
                query = """
                    SELECT symbol, qty, entry_price, exit_price, closed_pnl, close_time, close_timestamp, exchange
                    FROM closed_pnl_history
                    WHERE close_timestamp >= ? AND close_timestamp <= ?
                """
                params = [start_time, end_time]
                
                if exchange:
                    query += " AND exchange = ?"
                    params.append(exchange)
                
                # Сортировка
                if sort_by == 'pnl':
                    query += " ORDER BY ABS(closed_pnl) DESC"
                else:  # sort by time
                    query += " ORDER BY close_timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Преобразуем в список словарей
                result = []
                for row in rows:
                    result.append({
                        'symbol': row['symbol'],
                        'qty': row['qty'],
                        'entry_price': row['entry_price'],
                        'exit_price': row['exit_price'],
                        'closed_pnl': row['closed_pnl'],
                        'close_time': row['close_time'],
                        'close_timestamp': row['close_timestamp'],
                        'exchange': row['exchange']
                    })
                
                pass
                return result
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки истории PnL: {e}")
            import traceback
            pass
            return []
    
    def get_latest_pnl_timestamp(self, exchange=None) -> Optional[int]:
        """
        Получает timestamp последней записи PnL в БД
        
        Args:
            exchange: Фильтр по бирже (если None, проверяются все)
        
        Returns:
            Timestamp в миллисекундах или None если записей нет
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if exchange:
                    cursor.execute("""
                        SELECT MAX(close_timestamp) as max_ts
                        FROM closed_pnl_history
                        WHERE exchange = ?
                    """, (exchange,))
                else:
                    cursor.execute("""
                        SELECT MAX(close_timestamp) as max_ts
                        FROM closed_pnl_history
                    """)
                
                row = cursor.fetchone()
                if row and row['max_ts']:
                    return row['max_ts']
                return None
                
        except Exception as e:
            pass
            return None


# Глобальный экземпляр базы данных
_app_database_instance = None
_app_database_lock = threading.Lock()


def get_app_database(db_path: str = None) -> AppDatabase:
    """
    Получает глобальный экземпляр базы данных App
    
    Args:
        db_path: Путь к файлу базы данных (если None, используется data/app_data.db)
    
    Returns:
        Экземпляр AppDatabase
    """
    global _app_database_instance
    
    with _app_database_lock:
        if _app_database_instance is None:
            logger.info("🔧 Инициализация App Database...")
            _app_database_instance = AppDatabase(db_path)
        
        return _app_database_instance

