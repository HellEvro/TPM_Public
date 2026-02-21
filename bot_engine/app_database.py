#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реляционная база данных для хранения ВСЕХ данных app.py

📋 Обзор:
---------
Все JSON данные app.py теперь хранятся в SQLite БД вместо глобальных переменных.
Это обеспечивает масштабируемость, производительность и надежность.

Архитектура:
-----------
- Путь по умолчанию: data/app_data.db
- Поддержка UNC путей (сетевые диски)
- WAL режим для параллельных операций
- Автоматическое создание при первом использовании

Хранит:
------
- Позиции и статистика (positions_data)
- Закрытые PnL (closed_pnl)
- Максимальные значения прибыли/убытка (max_profit_values, max_loss_values)
- Другие данные app.py

Преимущества SQLite БД:
---------------------
✅ Хранит миллиарды записей
✅ Быстрый поиск по индексам
✅ WAL режим для параллельных чтений/записей
✅ Атомарные операции
✅ Поддержка UNC путей (сетевые диски)
✅ Автоматическая миграция схемы
"""

import sqlite3
import json
import os
import threading
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from contextlib import contextmanager
import logging

logger = logging.getLogger('App.Database')


def _get_project_root() -> Path:
    """
    Определяет корень проекта относительно текущего файла.
    Корень проекта - директория, где лежит app.py и bot_engine/
    """
    current = Path(__file__).resolve()
    # Поднимаемся от bot_engine/app_database.py до корня проекта
    # bot_engine/ -> корень
    for parent in [current.parent.parent] + list(current.parents):
        if parent and (parent / 'app.py').exists() and (parent / 'bot_engine').exists():
            return parent
    # Фолбек: поднимаемся на 1 уровень
    try:
        return current.parents[1]
    except IndexError:
        return current.parent


class AppDatabase:
    """
    Реляционная база данных для всех данных app.py
    """
    
    def __init__(self, db_path: str = None):
        """
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных (если None, используется data/app_data.db)
        """
        if db_path is None:
            # ✅ ПУТЬ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
            project_root = _get_project_root()
            db_path = project_root / 'data' / 'app_data.db'
            db_path = str(db_path.resolve())
        
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Создаем директорию если её нет (работает и с UNC путями)
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        except OSError as e:
            logger.error(f"❌ Ошибка создания директории для БД: {e}")
            raise
        
        # Инициализируем базу данных
        self._init_database()
        
        logger.info(f"✅ App Database инициализирована: {db_path}")
    
    def _check_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Проверяет целостность БД
        
        Returns:
            Tuple[bool, Optional[str]]: (is_ok, error_message)
        """
        if not os.path.exists(self.db_path):
            return True, None  # Нет БД - это нормально, будет создана
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]
            conn.close()
            
            if result == "ok":
                return True, None
            else:
                conn = sqlite3.connect(self.db_path, timeout=60.0)
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_results = cursor.fetchall()
                error_details = "; ".join([row[0] for row in integrity_results if row[0] != "ok"])
                conn.close()
                return False, error_details or result
        except Exception as e:
            return False, f"Ошибка проверки целостности: {e}"
    
    @contextmanager
    def _get_connection(self, retry_on_locked: bool = True, max_retries: int = 5):
        """
        Получает соединение с БД с retry логикой
        
        Args:
            retry_on_locked: Повторять ли попытки при блокировке
            max_retries: Максимальное количество попыток
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.row_factory = sqlite3.Row
                
                # Настройки производительности
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")  # 64MB
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA foreign_keys=ON")
                
                try:
                    yield conn
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
                
                return  # Успешно выполнили операцию
                
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                # КРИТИЧНО: не делать continue при "locked" — иначе "generator didn't stop after throw()"
                if "database is locked" in error_msg or "database table is locked" in error_msg:
                    last_error = e
                    logger.error(f"❌ БД заблокирована после {max_retries} попыток")
                    raise
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ Ошибка работы с БД: {e}")
                raise
        
        if last_error:
            raise last_error
    
    def _init_database(self):
        """Создает все таблицы и индексы"""
        if os.path.exists(self.db_path):
            logger.info("🔍 Проверка целостности БД...")
            is_ok, error_msg = self._check_integrity()
            
            if not is_ok:
                logger.error(f"❌ Обнаружены повреждения в БД: {error_msg}")
                logger.warning("🔧 Попытка автоматического исправления...")
        else:
            logger.info(f"📁 Создается новая база данных: {self.db_path}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # ==================== ТАБЛИЦА: ПОЗИЦИИ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: одна строка = одна позиция со всеми полями
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    max_profit REAL,
                    max_loss REAL,
                    roi REAL,
                    high_roi INTEGER DEFAULT 0,
                    high_loss INTEGER DEFAULT 0,
                    side TEXT,
                    size REAL,
                    realized_pnl REAL,
                    leverage REAL,
                    position_category TEXT NOT NULL,
                    last_update TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для positions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_category ON positions(position_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_pnl ON positions(pnl)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_last_update ON positions(last_update)")
            
            # ==================== ТАБЛИЦА: СТАТИСТИКА ПОЗИЦИЙ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_pnl REAL DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    total_loss REAL DEFAULT 0,
                    high_profitable_count INTEGER DEFAULT 0,
                    profitable_count INTEGER DEFAULT 0,
                    losing_count INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    last_update TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # ==================== ТАБЛИЦА: БЫСТРЫЙ РОСТ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rapid_growth_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    start_pnl REAL NOT NULL,
                    current_pnl REAL NOT NULL,
                    growth_ratio REAL NOT NULL,
                    last_update TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для rapid_growth_positions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rapid_growth_symbol ON rapid_growth_positions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rapid_growth_ratio ON rapid_growth_positions(growth_ratio)")
            
            # ==================== ТАБЛИЦА: ЗАКРЫТЫЕ PNL (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все поля в отдельных столбцах, data_json только для дополнительных данных
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS closed_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    closed_pnl REAL,
                    closed_pnl_percent REAL,
                    fee REAL,
                    close_timestamp INTEGER NOT NULL,
                    entry_timestamp INTEGER,
                    duration_seconds INTEGER,
                    exchange TEXT,
                    extra_data_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, side, close_timestamp)
                )
            """)
            
            # Индексы для closed_pnl
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_symbol ON closed_pnl(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_close_timestamp ON closed_pnl(close_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_closed_pnl ON closed_pnl(closed_pnl)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_closed_pnl_exchange ON closed_pnl(exchange)")
            
            # ==================== ТАБЛИЦА: ВИРТУАЛЬНЫЕ ЗАКРЫТЫЕ PNL (ПРИИ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS virtual_closed_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    closed_pnl REAL,
                    closed_pnl_percent REAL,
                    close_timestamp INTEGER NOT NULL,
                    entry_timestamp INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_virtual_closed_pnl_symbol ON virtual_closed_pnl(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_virtual_closed_pnl_close_timestamp ON virtual_closed_pnl(close_timestamp)")
            
            # ==================== ТАБЛИЦА: МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS max_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, value_type)
                )
            """)
            
            # Индексы для max_values
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_max_values_symbol ON max_values(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_max_values_type ON max_values(value_type)")
            
            # ==================== МИГРАЦИЯ: Нормализация positions_data из JSON в отдельные таблицы ====================
            try:
                # Проверяем, есть ли старая структура (positions_data с data_json)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions_data'")
                if cursor.fetchone():
                    # Проверяем, есть ли данные в старой таблице
                    cursor.execute("SELECT COUNT(*) FROM positions_data")
                    old_count = cursor.fetchone()[0]
                    
                    if old_count > 0:
                        # Проверяем, мигрированы ли уже данные
                        cursor.execute("SELECT COUNT(*) FROM positions")
                        new_count = cursor.fetchone()[0]
                        
                        if new_count == 0:
                            logger.info("📦 Обнаружены данные в positions_data, выполняю миграцию в нормализованные таблицы...")
                            
                            # Загружаем данные из старой таблицы
                            cursor.execute("SELECT data_type, data_json, last_update FROM positions_data")
                            old_rows = cursor.fetchall()
                            
                            now = datetime.now().isoformat()
                            
                            for row in old_rows:
                                data_type = row['data_type']
                                data_json = row['data_json']
                                last_update = row['last_update']
                                
                                try:
                                    data_value = json.loads(data_json)
                                    
                                    if data_type in ['high_profitable', 'profitable', 'losing']:
                                        # Мигрируем позиции
                                        positions = data_value if isinstance(data_value, list) else []
                                        for position in positions:
                                            cursor.execute("""
                                                INSERT INTO positions (
                                                    symbol, pnl, max_profit, max_loss, roi,
                                                    high_roi, high_loss, side, size, realized_pnl,
                                                    leverage, position_category, last_update, created_at, updated_at
                                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                            """, (
                                                position.get('symbol'),
                                                position.get('pnl', 0),
                                                position.get('max_profit'),
                                                position.get('max_loss'),
                                                position.get('roi'),
                                                1 if position.get('high_roi', False) else 0,
                                                1 if position.get('high_loss', False) else 0,
                                                position.get('side'),
                                                position.get('size'),
                                                position.get('realized_pnl'),
                                                position.get('leverage'),
                                                data_type,
                                                last_update,
                                                now,
                                                now
                                            ))
                                    elif data_type == 'stats':
                                        # Мигрируем статистику
                                        stats = data_value if isinstance(data_value, dict) else {}
                                        cursor.execute("""
                                            INSERT INTO positions_stats (
                                                total_pnl, total_profit, total_loss,
                                                high_profitable_count, profitable_count, losing_count,
                                                total_trades, last_update, created_at, updated_at
                                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """, (
                                            stats.get('total_pnl', 0),
                                            stats.get('total_profit', 0),
                                            stats.get('total_loss', 0),
                                            stats.get('high_profitable_count', 0),
                                            stats.get('profitable_count', 0),
                                            stats.get('losing_count', 0),
                                            stats.get('total_trades', 0),
                                            last_update,
                                            now,
                                            now
                                        ))
                                    elif data_type == 'rapid_growth':
                                        # Мигрируем rapid_growth
                                        rapid_growth = data_value if isinstance(data_value, list) else []
                                        for growth in rapid_growth:
                                            cursor.execute("""
                                                INSERT INTO rapid_growth_positions (
                                                    symbol, start_pnl, current_pnl, growth_ratio,
                                                    last_update, created_at, updated_at
                                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                            """, (
                                                growth.get('symbol'),
                                                growth.get('start_pnl', 0),
                                                growth.get('current_pnl', 0),
                                                growth.get('growth_ratio', 0),
                                                last_update,
                                                now,
                                                now
                                            ))
                                except Exception as e:
                                    logger.warning(f"⚠️ Ошибка миграции {data_type}: {e}")
                                    continue
                            
                            logger.info("✅ Миграция positions_data завершена: данные перенесены из JSON в нормализованные таблицы")
                        else:
                            pass
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Переименование data_json в extra_data_json для closed_pnl ====================
            try:
                # Проверяем, есть ли столбец data_json
                cursor.execute("PRAGMA table_info(closed_pnl)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'data_json' in columns and 'extra_data_json' not in columns:
                    logger.info("📦 Миграция: переименовываю data_json в extra_data_json для closed_pnl")
                    cursor.execute("ALTER TABLE closed_pnl RENAME COLUMN data_json TO extra_data_json")
                    logger.info("✅ Миграция closed_pnl завершена")
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Колонка is_virtual для виртуальных сделок ПРИИ ====================
            try:
                cursor.execute("PRAGMA table_info(closed_pnl)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'is_virtual' not in columns:
                    logger.info("📦 Миграция: добавляю колонку is_virtual в closed_pnl")
                    cursor.execute("ALTER TABLE closed_pnl ADD COLUMN is_virtual INTEGER NOT NULL DEFAULT 0")
                    logger.info("✅ Миграция closed_pnl.is_virtual завершена")
            except Exception as e:
                pass
            
            conn.commit()
            
            pass
    
    # ==================== МЕТОДЫ ДЛЯ POSITIONS_DATA ====================
    
    def save_positions_data(self, positions_data: Dict) -> bool:
        """
        Сохраняет positions_data в нормализованные таблицы БД
        
        Args:
            positions_data: Словарь с данными позиций
            
        Returns:
            bool: True если успешно сохранено
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                last_update = positions_data.get('last_update')
                
                # Удаляем старые данные позиций
                cursor.execute("DELETE FROM positions")
                
                # Сохраняем позиции в нормализованную таблицу
                for category in ['high_profitable', 'profitable', 'losing']:
                    positions = positions_data.get(category, [])
                    for position in positions:
                        cursor.execute("""
                            INSERT INTO positions (
                                symbol, pnl, max_profit, max_loss, roi,
                                high_roi, high_loss, side, size, realized_pnl,
                                leverage, position_category, last_update, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            position.get('symbol'),
                            position.get('pnl', 0),
                            position.get('max_profit'),
                            position.get('max_loss'),
                            position.get('roi'),
                            1 if position.get('high_roi', False) else 0,
                            1 if position.get('high_loss', False) else 0,
                            position.get('side'),
                            position.get('size'),
                            position.get('realized_pnl'),
                            position.get('leverage'),
                            category,
                            last_update,
                            now,
                            now
                        ))
                
                # Сохраняем статистику
                stats = positions_data.get('stats', {})
                cursor.execute("DELETE FROM positions_stats")
                cursor.execute("""
                    INSERT INTO positions_stats (
                        total_pnl, total_profit, total_loss,
                        high_profitable_count, profitable_count, losing_count,
                        total_trades, last_update, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats.get('total_pnl', 0),
                    stats.get('total_profit', 0),
                    stats.get('total_loss', 0),
                    stats.get('high_profitable_count', 0),
                    stats.get('profitable_count', 0),
                    stats.get('losing_count', 0),
                    stats.get('total_trades', 0),
                    last_update,
                    now,
                    now
                ))
                
                # Сохраняем rapid_growth
                rapid_growth = positions_data.get('rapid_growth', [])
                cursor.execute("DELETE FROM rapid_growth_positions")
                for growth in rapid_growth:
                    cursor.execute("""
                        INSERT INTO rapid_growth_positions (
                            symbol, start_pnl, current_pnl, growth_ratio,
                            last_update, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        growth.get('symbol'),
                        growth.get('start_pnl', 0),
                        growth.get('current_pnl', 0),
                        growth.get('growth_ratio', 0),
                        last_update,
                        now,
                        now
                    ))
                
                pass
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения positions_data: {e}")
            import traceback
            pass
            return False
    
    def load_positions_data(self) -> Dict:
        """
        Загружает positions_data из нормализованных таблиц БД
        
        Returns:
            Dict: Словарь с данными позиций
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                result = {
                    'high_profitable': [],
                    'profitable': [],
                    'losing': [],
                    'rapid_growth': [],
                    'stats': {},
                    'last_update': None,
                    'closed_pnl': [],
                    'total_trades': 0
                }
                
                # Загружаем позиции из нормализованной таблицы
                cursor.execute("""
                    SELECT symbol, pnl, max_profit, max_loss, roi,
                           high_roi, high_loss, side, size, realized_pnl,
                           leverage, position_category, last_update
                    FROM positions
                """)
                position_rows = cursor.fetchall()
                
                for row in position_rows:
                    position = {
                        'symbol': row['symbol'],
                        'pnl': row['pnl'],
                        'max_profit': row['max_profit'],
                        'max_loss': row['max_loss'],
                        'roi': row['roi'],
                        'high_roi': bool(row['high_roi']),
                        'high_loss': bool(row['high_loss']),
                        'side': row['side'],
                        'size': row['size'],
                        'realized_pnl': row['realized_pnl'],
                        'leverage': row['leverage']
                    }
                    
                    category = row['position_category']
                    if category in result:
                        result[category].append(position)
                    
                    if row['last_update']:
                        result['last_update'] = row['last_update']
                
                # Загружаем статистику
                cursor.execute("SELECT * FROM positions_stats ORDER BY id DESC LIMIT 1")
                stats_row = cursor.fetchone()
                if stats_row:
                    result['stats'] = {
                        'total_pnl': stats_row['total_pnl'],
                        'total_profit': stats_row['total_profit'],
                        'total_loss': stats_row['total_loss'],
                        'high_profitable_count': stats_row['high_profitable_count'],
                        'profitable_count': stats_row['profitable_count'],
                        'losing_count': stats_row['losing_count'],
                        'total_trades': stats_row['total_trades']
                    }
                    result['total_trades'] = stats_row['total_trades']
                    if stats_row['last_update']:
                        result['last_update'] = stats_row['last_update']
                
                # Загружаем rapid_growth
                cursor.execute("""
                    SELECT symbol, start_pnl, current_pnl, growth_ratio, last_update
                    FROM rapid_growth_positions
                """)
                growth_rows = cursor.fetchall()
                for row in growth_rows:
                    result['rapid_growth'].append({
                        'symbol': row['symbol'],
                        'start_pnl': row['start_pnl'],
                        'current_pnl': row['current_pnl'],
                        'growth_ratio': row['growth_ratio']
                    })
                    if row['last_update']:
                        result['last_update'] = row['last_update']
                
                pass
                return result
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки positions_data: {e}")
            import traceback
            pass
            # Пробуем загрузить из старой структуры для обратной совместимости
            try:
                cursor.execute("SELECT data_type, data_json, last_update FROM positions_data")
                rows = cursor.fetchall()
                result = {
                    'high_profitable': [],
                    'profitable': [],
                    'losing': [],
                    'rapid_growth': [],
                    'stats': {},
                    'last_update': None,
                    'closed_pnl': [],
                    'total_trades': 0
                }
                for row in rows:
                    data_type = row['data_type']
                    data_json = row['data_json']
                    last_update = row['last_update']
                    try:
                        data_value = json.loads(data_json)
                        result[data_type] = data_value
                        if last_update:
                            result['last_update'] = last_update
                    except json.JSONDecodeError:
                        pass
                if result['stats'] and isinstance(result['stats'], dict):
                    result['total_trades'] = result['stats'].get('total_trades', 0)
                return result
            except:
                return {
                    'high_profitable': [],
                    'profitable': [],
                    'losing': [],
                    'rapid_growth': [],
                    'stats': {},
                    'last_update': None,
                    'closed_pnl': [],
                    'total_trades': 0
                }
    
    # ==================== МЕТОДЫ ДЛЯ CLOSED_PNL ====================
    
    def save_closed_pnl(self, closed_pnl_list: List[Dict], exchange: str = None) -> bool:
        """
        Сохраняет закрытые PnL в БД
        
        Args:
            closed_pnl_list: Список словарей с данными закрытых PnL
            exchange: Название биржи
            
        Returns:
            bool: True если успешно сохранено
        """
        if not closed_pnl_list:
            return True
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                saved_count = 0
                for pnl_data in closed_pnl_list:
                    try:
                        # Извлекаем данные
                        symbol = pnl_data.get('symbol', '')
                        side = pnl_data.get('side', '')
                        entry_price = pnl_data.get('entry_price')
                        exit_price = pnl_data.get('exit_price')
                        size = pnl_data.get('size')
                        closed_pnl = pnl_data.get('closed_pnl', 0)
                        closed_pnl_percent = pnl_data.get('closed_pnl_percent', 0)
                        fee = pnl_data.get('fee', 0)
                        close_timestamp = pnl_data.get('close_timestamp', 0)
                        entry_timestamp = pnl_data.get('entry_timestamp')
                        duration_seconds = pnl_data.get('duration_seconds')
                        # Собираем дополнительные данные в extra_data_json
                        extra_data = {}
                        known_fields = {
                            'symbol', 'side', 'entry_price', 'exit_price', 'size',
                            'closed_pnl', 'closed_pnl_percent', 'fee',
                            'close_timestamp', 'entry_timestamp', 'duration_seconds', 'exchange'
                        }
                        for key, value in pnl_data.items():
                            if key not in known_fields:
                                extra_data[key] = value
                        extra_data_json = json.dumps(extra_data, ensure_ascii=False) if extra_data else None
                        
                        # Вставляем или обновляем запись (реальные сделки; виртуальные — в virtual_closed_pnl)
                        cursor.execute("""
                            INSERT OR REPLACE INTO closed_pnl (
                                symbol, side, entry_price, exit_price, size,
                                closed_pnl, closed_pnl_percent, fee,
                                close_timestamp, entry_timestamp, duration_seconds,
                                exchange, extra_data_json, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                COALESCE((SELECT created_at FROM closed_pnl 
                                    WHERE symbol = ? AND side = ? AND close_timestamp = ?), ?),
                                ?)
                        """, (
                            symbol, side, entry_price, exit_price, size,
                            closed_pnl, closed_pnl_percent, fee,
                            close_timestamp, entry_timestamp, duration_seconds,
                            exchange or '', extra_data_json,
                            symbol, side, close_timestamp, now, now
                        ))
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Ошибка сохранения записи closed_pnl: {e}")
                        continue
                
                logger.info(f"💾 Сохранено {saved_count} записей closed_pnl в БД")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения closed_pnl: {e}")
            import traceback
            pass
            return False
    
    def save_virtual_closed_pnl(self, symbol: str, side: str, entry_price: float, exit_price: float,
                                closed_pnl_percent: float, close_timestamp: int,
                                entry_timestamp: Optional[int] = None, size: float = 0) -> bool:
        """Сохраняет одну виртуальную закрытую сделку ПРИИ для отображения на странице Закрытые PnL."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO virtual_closed_pnl (
                        symbol, side, entry_price, exit_price, size,
                        closed_pnl, closed_pnl_percent, close_timestamp, entry_timestamp, created_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """, (symbol, side, entry_price, exit_price, size,
                      closed_pnl_percent, close_timestamp, entry_timestamp or close_timestamp, now))
                return True
        except Exception as e:
            logger.debug("Ошибка сохранения виртуальной сделки: %s", e)
            return False
    
    def get_closed_pnl(self, sort_by: str = 'time', period: str = 'all', 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       exchange: Optional[str] = None) -> List[Dict]:
        """
        Получает закрытые PnL из БД с фильтрацией
        
        Args:
            sort_by: Способ сортировки ('time' или 'pnl')
            period: Период фильтрации ('all', 'day', 'week', 'month', 'half_year', 'year', 'custom')
            start_date: Начальная дата для custom периода (timestamp в мс или строка 'YYYY-MM-DD')
            end_date: Конечная дата для custom периода (timestamp в мс или строка 'YYYY-MM-DD')
            exchange: Фильтр по бирже
            
        Returns:
            List[Dict]: Список закрытых PnL
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Определяем временной диапазон
                now = int(time.time() * 1000)  # миллисекунды
                now_dt = datetime.fromtimestamp(now / 1000)
                
                if period == 'all':
                    period_start = 0
                    period_end = now
                elif period == 'day':
                    # Начало текущего дня (00:00:00)
                    day_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    period_start = int(day_start.timestamp() * 1000)
                    period_end = now
                elif period == 'week':
                    # Начало текущей недели (понедельник 00:00:00)
                    days_since_monday = now_dt.weekday()  # 0 = понедельник, 6 = воскресенье
                    week_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    week_start = week_start - timedelta(days=days_since_monday)
                    period_start = int(week_start.timestamp() * 1000)
                    period_end = now
                elif period == 'month':
                    # Начало текущего месяца (1-е число 00:00:00)
                    month_start = now_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    period_start = int(month_start.timestamp() * 1000)
                    period_end = now
                elif period == 'half_year':
                    # Начало текущего полугодия (январь или июль, 1-е число 00:00:00)
                    if now_dt.month <= 6:
                        half_year_start = now_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                    else:
                        half_year_start = now_dt.replace(month=7, day=1, hour=0, minute=0, second=0, microsecond=0)
                    period_start = int(half_year_start.timestamp() * 1000)
                    period_end = now
                elif period == 'year':
                    # Начало текущего года (1 января 00:00:00)
                    year_start = now_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                    period_start = int(year_start.timestamp() * 1000)
                    period_end = now
                elif period == 'custom':
                    # Парсим даты
                    if start_date:
                        if isinstance(start_date, str) and '-' in start_date:
                            # Формат 'YYYY-MM-DD'
                            dt = datetime.strptime(start_date, '%Y-%m-%d')
                            period_start = int(dt.timestamp() * 1000)
                        else:
                            period_start = int(start_date)
                    else:
                        period_start = 0
                    
                    if end_date:
                        if isinstance(end_date, str) and '-' in end_date:
                            # Формат 'YYYY-MM-DD'
                            dt = datetime.strptime(end_date, '%Y-%m-%d')
                            period_end = int(dt.timestamp() * 1000)
                        else:
                            period_end = int(end_date)
                    else:
                        period_end = now
                else:
                    period_start = 0
                    period_end = now
                
                # Строим запрос
                query = """
                    SELECT symbol, side, entry_price, exit_price, size,
                           closed_pnl, closed_pnl_percent, fee,
                           close_timestamp, entry_timestamp, duration_seconds,
                           exchange, extra_data_json
                    FROM closed_pnl
                    WHERE close_timestamp >= ? AND close_timestamp <= ?
                """
                params = [period_start, period_end]
                
                if exchange:
                    query += " AND exchange = ?"
                    params.append(exchange)
                
                # Сортировка
                if sort_by == 'pnl':
                    query += " ORDER BY ABS(closed_pnl) DESC"
                else:  # По умолчанию по времени
                    query += " ORDER BY close_timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                result = []
                for row in rows:
                    ts = row['close_timestamp'] or 0
                    pnl_data = {
                        'symbol': row['symbol'],
                        'side': row['side'],
                        'entry_price': row['entry_price'],
                        'exit_price': row['exit_price'],
                        'size': row['size'],
                        'closed_pnl': row['closed_pnl'],
                        'closed_pnl_percent': row['closed_pnl_percent'],
                        'fee': row['fee'],
                        'close_timestamp': ts,
                        'close_time': datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S') if ts else '',
                        'entry_timestamp': row['entry_timestamp'],
                        'duration_seconds': row['duration_seconds'],
                        'exchange': row['exchange'],
                        'is_virtual': False,
                    }
                    
                    # Загружаем дополнительные данные из extra_data_json
                    if row['extra_data_json']:
                        try:
                            extra_data = json.loads(row['extra_data_json'])
                            pnl_data.update(extra_data)
                        except json.JSONDecodeError:
                            pass
                    
                    result.append(pnl_data)
                
                # Виртуальные сделки ПРИИ: подмешиваем из virtual_closed_pnl
                cursor.execute("""
                    SELECT symbol, side, entry_price, exit_price, size,
                           closed_pnl, closed_pnl_percent, close_timestamp, entry_timestamp
                    FROM virtual_closed_pnl
                    WHERE close_timestamp >= ? AND close_timestamp <= ?
                """, (period_start, period_end))
                vrows = cursor.fetchall()
                for row in vrows:
                    ts = row['close_timestamp'] or 0
                    result.append({
                        'symbol': row['symbol'],
                        'side': row['side'],
                        'entry_price': row['entry_price'],
                        'exit_price': row['exit_price'],
                        'size': row['size'] or 0,
                        'closed_pnl': row['closed_pnl'] or 0,
                        'closed_pnl_percent': row['closed_pnl_percent'],
                        'fee': 0,
                        'close_timestamp': ts,
                        'close_time': datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S') if ts else '',
                        'entry_timestamp': row['entry_timestamp'],
                        'duration_seconds': None,
                        'exchange': 'virtual',
                        'is_virtual': True,
                    })
                
                # Сортировка объединённого списка
                if sort_by == 'pnl':
                    result.sort(key=lambda x: abs(float(x.get('closed_pnl') or 0)), reverse=True)
                else:
                    result.sort(key=lambda x: int(x.get('close_timestamp') or 0), reverse=True)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки closed_pnl: {e}")
            import traceback
            pass
            return []
    
    def get_latest_closed_pnl_timestamp(self, exchange: Optional[str] = None) -> Optional[int]:
        """
        Получает timestamp последней закрытой позиции
        
        Args:
            exchange: Фильтр по бирже
            
        Returns:
            Optional[int]: Timestamp в миллисекундах или None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if exchange:
                    cursor.execute("""
                        SELECT MAX(close_timestamp) as max_ts 
                        FROM closed_pnl 
                        WHERE exchange = ?
                    """, (exchange,))
                else:
                    cursor.execute("SELECT MAX(close_timestamp) as max_ts FROM closed_pnl")
                
                row = cursor.fetchone()
                if row and row['max_ts']:
                    return int(row['max_ts'])
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения последнего timestamp: {e}")
            return None
    
    # ==================== МЕТОДЫ ДЛЯ MAX_VALUES ====================
    
    def save_max_values(self, max_profit_values: Dict, max_loss_values: Dict) -> bool:
        """
        Сохраняет максимальные значения прибыли/убытка
        
        Args:
            max_profit_values: Словарь {symbol: value}
            max_loss_values: Словарь {symbol: value}
            
        Returns:
            bool: True если успешно сохранено
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                timestamp = int(time.time() * 1000)
                
                # Сохраняем max_profit_values
                for symbol, value in max_profit_values.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO max_values 
                        (symbol, value_type, value, timestamp, created_at, updated_at)
                        VALUES (?, ?, ?, ?,
                            COALESCE((SELECT created_at FROM max_values WHERE symbol = ? AND value_type = ?), ?),
                            ?)
                    """, (symbol, 'profit', float(value), timestamp, symbol, 'profit', now, now))
                
                # Сохраняем max_loss_values
                for symbol, value in max_loss_values.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO max_values 
                        (symbol, value_type, value, timestamp, created_at, updated_at)
                        VALUES (?, ?, ?, ?,
                            COALESCE((SELECT created_at FROM max_values WHERE symbol = ? AND value_type = ?), ?),
                            ?)
                    """, (symbol, 'loss', float(value), timestamp, symbol, 'loss', now, now))
                
                pass
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения max_values: {e}")
            import traceback
            pass
            return False
    
    def load_max_values(self) -> Tuple[Dict, Dict]:
        """
        Загружает максимальные значения прибыли/убытка
        
        Returns:
            Tuple[Dict, Dict]: (max_profit_values, max_loss_values)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                max_profit_values = {}
                max_loss_values = {}
                
                cursor.execute("SELECT symbol, value_type, value FROM max_values")
                rows = cursor.fetchall()
                
                for row in rows:
                    symbol = row['symbol']
                    value_type = row['value_type']
                    value = row['value']
                    
                    if value_type == 'profit':
                        max_profit_values[symbol] = value
                    elif value_type == 'loss':
                        max_loss_values[symbol] = value
                
                pass
                return max_profit_values, max_loss_values
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки max_values: {e}")
            return {}, {}


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

