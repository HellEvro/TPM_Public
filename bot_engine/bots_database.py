#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реляционная база данных для хранения ВСЕХ данных bots.py

📋 Обзор:
---------
Все данные bots.py теперь хранятся в SQLite БД вместо JSON файлов.
Это обеспечивает масштабируемость, производительность и надежность.

Архитектура:
-----------
- Путь по умолчанию: data/bots_data.db
- Поддержка UNC путей (сетевые диски)
- WAL режим для параллельных операций
- Автоматическое создание при первом использовании
- Автоматическая миграция данных из JSON

Хранит:
-------
- Состояние ботов (bots_state)
- Реестр позиций (bot_positions_registry)
- RSI кэш (rsi_cache)
- Кэш свечей (candles_cache)
- Состояние процессов (process_state)
- Индивидуальные настройки монет (individual_coin_settings)
- Зрелые монеты (mature_coins)
- Кэш проверки зрелости (maturity_check_cache)
- Делистированные монеты (delisted)

Преимущества SQLite БД:
----------------------
✅ Хранит миллиарды записей
✅ Быстрый поиск по индексам
✅ WAL режим для параллельных чтений/записей
✅ Атомарные операции
✅ Поддержка UNC путей (сетевые диски)
✅ Автоматическая миграция схемы
✅ Автоматическая миграция данных из JSON

Использование:
-------------
```python
from bot_engine.bots_database import get_bots_database

# Получаем глобальный экземпляр (singleton)
db = get_bots_database()

# Сохраняем состояние ботов
db.save_bots_state(bots_data, auto_bot_config)

# Загружаем состояние ботов
state = db.load_bots_state()

# Получаем статистику
stats = db.get_database_stats()
```

Настройки производительности:
-----------------------------
- PRAGMA journal_mode=WAL - Write-Ahead Logging
- PRAGMA synchronous=NORMAL - баланс скорости/надежности
- PRAGMA cache_size=-64000 - 64MB кеш
- PRAGMA temp_store=MEMORY - временные таблицы в памяти

Документация:
------------
См. docs/AI_DATABASE_MIGRATION_GUIDE.md для подробного руководства
по архитектуре, миграции и best practices.
"""

import sqlite3
import json
import os
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from contextlib import contextmanager
import logging

# Импортируем утилиты для выполнения SQL-скриптов
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.database_utils import load_sql_file, execute_sql_string
except ImportError:
    # Если утилиты недоступны, используем встроенный код
    load_sql_file = None
    execute_sql_string = None

logger = logging.getLogger('Bots.Database')


def _get_project_root() -> Path:
    """
    Определяет корень проекта относительно текущего файла.
    Корень проекта - директория, где лежит bots.py и bot_engine/
    """
    current = Path(__file__).resolve()
    # Поднимаемся от bot_engine/bots_database.py до корня проекта
    # bot_engine/ -> корень
    for parent in [current.parent.parent] + list(current.parents):
        if parent and (parent / 'bots.py').exists() and (parent / 'bot_engine').exists():
            return parent
    # Фолбек: поднимаемся на 1 уровень
    try:
        return current.parents[1]
    except IndexError:
        return current.parent


class BotsDatabase:
    """
    Реляционная база данных для всех данных bots.py
    """
    
    def __init__(self, db_path: str = None):
        """
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных (если None, используется data/bots_data.db)
        """
        if db_path is None:
            # ✅ ПУТЬ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
            project_root = _get_project_root()
            db_path = project_root / 'data' / 'bots_data.db'
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
        
        logger.info(f"✅ Bots Database инициализирована: {db_path}")
    
    def _check_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Проверяет целостность БД (быстрая проверка без блокировок)
        
        Returns:
            Tuple[bool, Optional[str]]: (is_ok, error_message)
            is_ok = True если БД в порядке, False если есть проблемы
            error_message = описание проблемы или None
        """
        if not os.path.exists(self.db_path):
            return True, None  # Нет БД - это нормально, будет создана
        
        logger.debug("   [1/4] Проверка существования БД...")
        
        try:
            # Сначала проверяем, не заблокирована ли БД другим процессом
            # Пытаемся простое подключение с коротким таймаутом
            logger.debug("   [2/4] Проверка блокировки БД...")
            try:
                test_conn = sqlite3.connect(self.db_path, timeout=1.0)
                test_conn.close()
                logger.debug("   [2/4] ✅ БД не заблокирована")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # БД заблокирована - пропускаем проверку, чтобы не блокировать запуск
                    logger.debug("   [2/4] ⚠️ БД заблокирована другим процессом, пропускаем проверку целостности")
                    return True, None
                raise
            
            # Синхронизируем WAL файлы перед проверкой (это может решить проблему зависания)
            logger.debug("   [3/4] Подключение к БД и проверка режима журнала...")
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                cursor = conn.cursor()
                
                # Проверяем режим журнала
                logger.debug("   [3/4] Проверка режима журнала...")
                cursor.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                logger.debug(f"   [3/4] Режим журнала: {journal_mode}")
                
                # Если WAL режим - делаем checkpoint для синхронизации
                if journal_mode.upper() == 'WAL':
                    logger.debug("   [3/4] WAL режим обнаружен, выполнение checkpoint...")
                    try:
                        # Делаем пассивный checkpoint (не блокирует читателей)
                        cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        conn.commit()
                        logger.debug("   [3/4] ✅ Checkpoint выполнен")
                    except Exception as e:
                        logger.debug(f"   [3/4] ⚠️ Ошибка checkpoint (игнорируем): {e}")
                        pass  # Игнорируем ошибки checkpoint
                
                # Быстрая проверка целостности (быстрее чем integrity_check)
                logger.debug("   [4/4] Выполнение PRAGMA quick_check...")
                cursor.execute("PRAGMA busy_timeout = 2000")  # 2 секунды
                cursor.execute("PRAGMA quick_check")
                result = cursor.fetchone()[0]
                logger.debug(f"   [4/4] ✅ Результат проверки: {result}")
                conn.close()
                
                if result == "ok":
                    logger.debug("   ✅ Проверка целостности БД завершена успешно")
                    return True, None
                else:
                    # Есть проблемы - но не делаем полную проверку (она может быть очень долгой)
                    logger.warning(f"⚠️ Обнаружены проблемы в БД: {result}")
                    return False, result
                    
            except sqlite3.OperationalError as e:
                error_str = str(e).lower()
                if "locked" in error_str:
                    # БД заблокирована - пропускаем проверку
                    logger.debug("   [3/4] ⚠️ БД заблокирована, пропускаем проверку целостности")
                    return True, None
                # Другие ошибки - считаем БД валидной, чтобы не блокировать запуск
                logger.warning(f"⚠️ Ошибка проверки целостности БД: {e}, продолжаем работу...")
                return True, None
                
        except Exception as e:
            # В случае ошибки считаем БД валидной, чтобы не блокировать запуск
            logger.debug(f"ℹ️ Ошибка проверки целостности БД: {e}, продолжаем работу...")
            return True, None  # Возвращаем True, чтобы не блокировать запуск
    
    def _backup_database(self, max_retries: int = 3) -> Optional[str]:
        """
        Создает резервную копию БД перед удалением с retry логикой
        
        Args:
            max_retries: Максимальное количество попыток при блокировке файла
        
        Returns:
            Путь к резервной копии или None если не удалось создать
        """
        if not os.path.exists(self.db_path):
            return None
        
        # Создаем имя резервной копии с timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"
        
        # Пытаемся создать резервную копию с retry логикой
        for attempt in range(max_retries):
            try:
                # Пытаемся закрыть все соединения перед копированием
                if attempt > 0:
                    logger.debug(f"🔄 Попытка создания резервной копии {attempt + 1}/{max_retries}...")
                    time.sleep(1.0 * attempt)  # Увеличиваем задержку с каждой попыткой (1s, 2s, 3s...)
                
                # Копируем БД и связанные файлы
                shutil.copy2(self.db_path, backup_path)
                
                # Копируем WAL и SHM файлы если есть
                wal_file = self.db_path + '-wal'
                shm_file = self.db_path + '-shm'
                if os.path.exists(wal_file):
                    shutil.copy2(wal_file, f"{backup_path}-wal")
                if os.path.exists(shm_file):
                    shutil.copy2(shm_file, f"{backup_path}-shm")
                
                logger.warning(f"💾 Создана резервная копия БД: {backup_path}")
                return backup_path
            except PermissionError as e:
                # Файл заблокирован другим процессом
                if attempt < max_retries - 1:
                    logger.debug(f"⚠️ Файл БД заблокирован, повторяем попытку через {1.0 * (attempt + 1)}s...")
                    continue
                else:
                    logger.error(f"❌ Не удалось создать резервную копию БД после {max_retries} попыток: {e}")
                    return None
            except Exception as e:
                # Другие ошибки
                if attempt < max_retries - 1:
                    logger.debug(f"⚠️ Ошибка создания резервной копии (попытка {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1.0 * attempt)
                    continue
                else:
                    logger.error(f"❌ Ошибка создания резервной копии БД после {max_retries} попыток: {e}")
                    return None
        
        return None
    
    def _check_database_has_data(self) -> bool:
        """
        Проверяет, есть ли данные в БД (пытается прочитать хотя бы одну таблицу)
        
        Returns:
            True если в БД есть данные, False если БД пуста или повреждена
        """
        if not os.path.exists(self.db_path):
            return False
        
        try:
            # Пытаемся подключиться в режиме только чтения
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()
            
            # Проверяем наличие таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if not tables:
                conn.close()
                return False
            
            # Пытаемся посчитать записи в основных таблицах
            main_tables = ['bots_state', 'bot_positions_registry', 'individual_coin_settings', 'mature_coins']
            for table in main_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        conn.close()
                        return True
                except:
                    continue
            
            conn.close()
            return False
        except Exception as e:
            logger.debug(f"⚠️ Не удалось проверить данные в БД: {e}")
            return False
    
    def _recreate_database(self):
        """
        Удаляет поврежденную БД и создает новую (только при явной ошибке подключения)
        
        ВАЖНО: Перед удалением создает резервную копию и проверяет наличие данных
        """
        if not os.path.exists(self.db_path):
            return
        
        try:
            # Проверяем, есть ли данные в БД
            has_data = self._check_database_has_data()
            
            if has_data:
                # Если есть данные - ОБЯЗАТЕЛЬНО создаем резервную копию
                backup_path = self._backup_database()
                if not backup_path:
                    # Не удаляем БД если не удалось создать резервную копию!
                    logger.error(f"❌ КРИТИЧНО: Не удалось создать резервную копию БД с данными!")
                    logger.error(f"❌ БД НЕ БУДЕТ УДАЛЕНА для защиты данных!")
                    raise Exception("Не удалось создать резервную копию БД с данными - удаление отменено")
                logger.warning(f"⚠️ ВНИМАНИЕ: БД содержит данные, создана резервная копия: {backup_path}")
            else:
                # Если данных нет - все равно создаем резервную копию на всякий случай
                self._backup_database()
            
            # Удаляем поврежденный файл и связанные файлы WAL/SHM
            wal_file = self.db_path + '-wal'
            shm_file = self.db_path + '-shm'
            
            if os.path.exists(wal_file):
                os.remove(wal_file)
            if os.path.exists(shm_file):
                os.remove(shm_file)
            os.remove(self.db_path)
            
            logger.warning(f"🗑️ Удалена поврежденная БД: {self.db_path}")
            if has_data:
                logger.warning(f"💾 Данные сохранены в резервной копии - можно восстановить при необходимости")
        except Exception as e:
            logger.error(f"❌ Ошибка удаления поврежденной БД: {e}")
            raise
    
    def _repair_database(self) -> bool:
        """
        Пытается исправить поврежденную БД
        
        Улучшенная логика:
        - Retry для резервных копий (до 3 попыток)
        - Использование существующих копий, если не удалось создать новую
        - Пропуск VACUUM при критических повреждениях (malformed, disk i/o error)
        - Умное восстановление (выбирает более старую копию, если была создана новая поврежденная)
        
        Returns:
            True если удалось исправить, False в противном случае
        """
        try:
            logger.warning("🔧 Попытка исправления БД...")
            
            # Пытаемся создать резервную копию перед исправлением (с retry)
            backup_path = self._backup_database(max_retries=3)
            backup_created = backup_path is not None
            
            if not backup_created:
                logger.warning("⚠️ Не удалось создать резервную копию перед исправлением (файл может быть заблокирован)")
                logger.info("💡 Попробую использовать существующие резервные копии для восстановления...")
            
            # Пытаемся использовать VACUUM для исправления (только если БД не слишком повреждена)
            vacuum_tried = False
            try:
                conn = sqlite3.connect(self.db_path, timeout=300.0)  # 5 минут для VACUUM
                cursor = conn.cursor()
                logger.info("🔧 Выполняю VACUUM для исправления БД (это может занять время)...")
                cursor.execute("VACUUM")
                conn.commit()
                conn.close()
                logger.info("✅ VACUUM выполнен")
                vacuum_tried = True
            except Exception as vacuum_error:
                error_str = str(vacuum_error).lower()
                if "malformed" in error_str or "disk i/o error" in error_str:
                    logger.warning(f"⚠️ VACUUM невозможен из-за критического повреждения: {vacuum_error}")
                    logger.info("💡 Пропускаю VACUUM, пытаюсь восстановить из резервной копии...")
                else:
                    logger.warning(f"⚠️ VACUUM не помог: {vacuum_error}")
                try:
                    conn.close()
                except:
                    pass
            
            # Проверяем, исправилась ли БД (только если VACUUM был выполнен)
            if vacuum_tried:
                is_ok, error_msg = self._check_integrity()
                if is_ok:
                    logger.info("✅ БД успешно исправлена с помощью VACUUM")
                    return True
                else:
                    logger.warning(f"⚠️ БД все еще повреждена после VACUUM: {error_msg[:200]}...")
            
            # Пытаемся восстановить из резервной копии
            logger.info("🔄 Попытка восстановления из резервной копии...")
            backups = self.list_backups()
            
            if backups:
                # Если мы создали резервную копию только что, используем более старую
                if backup_created and len(backups) > 1:
                    # Используем предпоследнюю копию (последняя - это та, что мы только что создали)
                    older_backup = backups[1]['path']
                    logger.info(f"📦 Восстанавливаю из более старой резервной копии: {older_backup}")
                    if self.restore_from_backup(older_backup):
                        return True
                else:
                    # Используем последнюю доступную копию
                    latest_backup = backups[0]['path']
                    logger.info(f"📦 Восстанавливаю из резервной копии: {latest_backup}")
                    if self.restore_from_backup(latest_backup):
                        return True
            
            # Если не удалось восстановить
            if not backups:
                logger.error("❌ Нет доступных резервных копий для восстановления")
                if not backup_created:
                    logger.error("❌ КРИТИЧНО: Не удалось создать резервную копию и нет существующих копий!")
                    logger.error("⚠️ БД останется поврежденной. Рекомендуется:")
                    logger.error("   1. Закрыть все процессы, использующие БД")
                    logger.error("   2. Попробовать восстановить вручную: db.restore_from_backup()")
                    logger.error("   3. Или создать новую БД (данные будут потеряны)")
            
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка исправления БД: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    @contextmanager
    def _get_connection(self, retry_on_locked: bool = True, max_retries: int = 5):
        """
        Контекстный менеджер для работы с БД с поддержкой retry при блокировках и автоматическим исправлением ошибок
        
        Args:
            retry_on_locked: Повторять попытки при ошибке "database is locked"
            max_retries: Максимальное количество попыток при блокировке
        
        Автоматически настраивает БД для оптимальной производительности:
        - WAL режим для параллельных операций
        - Оптимизированные настройки кеша и синхронизации
        - Автоматический commit/rollback при ошибках
        - Retry логика при блокировках (до 5 попыток с экспоненциальной задержкой)
        - Автоматическое исправление критических ошибок:
          * `database disk image is malformed` - автоматическое исправление через VACUUM/restore
          * `disk I/O error` - автоматическое исправление и повтор операции
        
        Критические ошибки обрабатываются автоматически:
        1. При обнаружении ошибки автоматически запускается `_repair_database()`
        2. После исправления операция автоматически повторяется один раз
        3. Перед исправлением создается резервная копия
        
        Использование:
        ```python
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bots_state")
            # Автоматический commit при выходе
        ```
        """
        last_error = None
        
        for attempt in range(max_retries if retry_on_locked else 1):
            try:
                # Увеличиваем timeout для операций записи при параллельном доступе
                # 60 секунд должно быть достаточно для работы через сеть
                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.row_factory = sqlite3.Row
                
                # Включаем WAL режим для лучшей производительности (параллельные чтения)
                # WAL позволяет нескольким читателям работать одновременно с одним писателем
                conn.execute("PRAGMA journal_mode=WAL")
                # Оптимизируем для быстрых записей
                conn.execute("PRAGMA synchronous=NORMAL")  # Быстрее чем FULL, но безопаснее чем OFF
                conn.execute("PRAGMA cache_size=-64000")  # 64MB кеш
                conn.execute("PRAGMA temp_store=MEMORY")  # Временные таблицы в памяти
                
                # Успешное подключение
                try:
                    yield conn
                    conn.commit()
                    conn.close()
                    return  # Успешно выполнили операцию
                except sqlite3.OperationalError as e:
                    error_str = str(e).lower()
                    
                    # Обрабатываем ошибки блокировки
                    if "database is locked" in error_str or "locked" in error_str:
                        conn.rollback()
                        conn.close()
                        last_error = e
                        if retry_on_locked and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 0.5  # Экспоненциальная задержка: 0.5s, 1s, 1.5s...
                            logger.debug(f"⚠️ БД заблокирована (попытка {attempt + 1}/{max_retries}), ждем {wait_time:.1f}s...")
                            time.sleep(wait_time)
                            continue  # Повторяем попытку
                        else:
                            # Превышено количество попыток
                            logger.warning(f"⚠️ БД заблокирована после {max_retries} попыток")
                            raise
                    
                    # КРИТИЧНО: Обработка ошибок I/O
                    elif "disk i/o error" in error_str or "i/o error" in error_str:
                        conn.rollback()
                        conn.close()
                        logger.error(f"❌ КРИТИЧНО: Ошибка I/O при работе с БД: {e}")
                        logger.warning("🔧 Попытка автоматического исправления...")
                        if attempt == 0:
                            # Пытаемся исправить только один раз
                            if self._repair_database():
                                logger.info("✅ БД исправлена, повторяем операцию...")
                                time.sleep(1)  # Небольшая задержка перед повтором
                                continue
                            else:
                                logger.error("❌ Не удалось исправить БД после I/O ошибки")
                                raise
                        else:
                            raise
                    else:
                        # Другие OperationalError - не повторяем
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
                
                # КРИТИЧНО: Обработка ошибки "database disk image is malformed"
                if "database disk image is malformed" in error_str or "malformed" in error_str:
                    logger.error(f"❌ КРИТИЧНО: БД повреждена (malformed): {self.db_path}")
                    logger.error(f"❌ Ошибка: {e}")
                    logger.warning("🔧 Попытка автоматического исправления...")
                    if attempt == 0:
                        # Пытаемся исправить только один раз
                        if self._repair_database():
                            logger.info("✅ БД исправлена, повторяем подключение...")
                            time.sleep(1)  # Небольшая задержка перед повтором
                            continue
                        else:
                            logger.error("❌ Не удалось исправить поврежденную БД")
                            raise
                    else:
                        raise
                
                # КРИТИЧНО: Обработка ошибки I/O при подключении
                elif "disk i/o error" in error_str or "i/o error" in error_str:
                    logger.error(f"❌ КРИТИЧНО: Ошибка I/O при подключении к БД: {self.db_path}")
                    logger.error(f"❌ Ошибка: {e}")
                    logger.warning("🔧 Попытка автоматического исправления...")
                    if attempt == 0:
                        # Пытаемся исправить только один раз
                        if self._repair_database():
                            logger.info("✅ БД исправлена, повторяем подключение...")
                            time.sleep(1)  # Небольшая задержка перед повтором
                            continue
                        else:
                            logger.error("❌ Не удалось исправить БД после I/O ошибки")
                            raise
                    else:
                        raise
                
                # Обработка "file is not a database"
                elif "file is not a database" in error_str or ("not a database" in error_str and "unable to open" not in error_str):
                    logger.error(f"❌ Файл БД поврежден (явная ошибка SQLite): {self.db_path}")
                    logger.error(f"❌ Ошибка: {e}")
                    # Восстанавливаем БД только при явной ошибке
                    self._recreate_database()
                    # Пытаемся подключиться снова (только один раз)
                    if attempt == 0:
                        continue
                    else:
                        raise
                
                # Обработка блокировок при подключении
                elif "database is locked" in error_str or "locked" in error_str:
                    # Ошибка блокировки при подключении
                    last_error = e
                    if retry_on_locked and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 0.5
                        logger.debug(f"⚠️ БД заблокирована при подключении (попытка {attempt + 1}/{max_retries}), ждем {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"⚠️ БД заблокирована при подключении после {max_retries} попыток")
                        raise
                else:
                    # Другие ошибки - не повторяем
                    raise
        
        # Если дошли сюда, значит все попытки исчерпаны
        if last_error:
            raise last_error
    
    def _init_database(self):
        """Создает все таблицы и индексы"""
        # Проверяем целостность БД при каждом запуске
        db_exists = os.path.exists(self.db_path)
        
        if db_exists:
            logger.info("🔍 Проверка целостности БД...")
            is_ok, error_msg = self._check_integrity()
            
            if not is_ok:
                logger.error(f"❌ Обнаружены повреждения в БД: {error_msg}")
                logger.warning("🔧 Попытка автоматического исправления...")
                
                if self._repair_database():
                    logger.info("✅ БД успешно исправлена")
                    # Проверяем еще раз после исправления
                    is_ok, error_msg = self._check_integrity()
                    if not is_ok:
                        logger.error(f"❌ БД все еще повреждена после исправления: {error_msg}")
                        logger.error("⚠️ Рекомендуется восстановить из резервной копии вручную")
                else:
                    logger.error("❌ Не удалось автоматически исправить БД")
                    logger.error("⚠️ Попробуйте восстановить из резервной копии: db.restore_from_backup()")
            else:
                logger.debug("✅ БД проверена, целостность в порядке")
        else:
            logger.info(f"📁 Создается новая база данных: {self.db_path}")
        
        # SQLite автоматически создает файл БД при первом подключении
        # Не нужно создавать пустой файл через touch() - это создает невалидную БД
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Миграция: добавляем новые поля если их нет
            self._migrate_schema(cursor, conn)
            
            # ==================== ТАБЛИЦА: БОТЫ (НОРМАЛИЗОВАННАЯ СТРУКТУРА) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: одна строка = один бот со всеми полями
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    auto_managed INTEGER DEFAULT 0,
                    volume_mode TEXT,
                    volume_value REAL,
                    entry_price REAL,
                    entry_time TEXT,
                    entry_timestamp REAL,
                    position_side TEXT,
                    position_size REAL,
                    position_size_coins REAL,
                    position_start_time TEXT,
                    unrealized_pnl REAL DEFAULT 0.0,
                    unrealized_pnl_usdt REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    leverage REAL DEFAULT 1.0,
                    margin_usdt REAL,
                    max_profit_achieved REAL DEFAULT 0.0,
                    trailing_stop_price REAL,
                    trailing_activation_threshold REAL,
                    trailing_activation_profit REAL DEFAULT 0.0,
                    trailing_locked_profit REAL DEFAULT 0.0,
                    trailing_active INTEGER DEFAULT 0,
                    trailing_max_profit_usdt REAL DEFAULT 0.0,
                    trailing_step_usdt REAL,
                    trailing_step_price REAL,
                    trailing_steps INTEGER DEFAULT 0,
                    trailing_reference_price REAL,
                    trailing_last_update_ts REAL DEFAULT 0.0,
                    trailing_take_profit_price REAL,
                    break_even_activated INTEGER DEFAULT 0,
                    break_even_stop_price REAL,
                    order_id TEXT,
                    current_price REAL,
                    last_price REAL,
                    last_rsi REAL,
                    last_trend TEXT,
                    last_signal_time TEXT,
                    last_bar_timestamp REAL,
                    entry_trend TEXT,
                    opened_by_autobot INTEGER DEFAULT 0,
                    bot_id TEXT,
                    extra_data_json TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для bots
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bots_symbol ON bots(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bots_status ON bots(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bots_updated ON bots(updated_at)")
            
            # ==================== ТАБЛИЦА: КОНФИГУРАЦИЯ АВТОБОТА ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auto_bot_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для auto_bot_config
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_auto_bot_config_key ON auto_bot_config(key)")
            
            # ==================== ТАБЛИЦА: СОСТОЯНИЕ БОТОВ (СТАРАЯ, ДЛЯ МИГРАЦИИ) ====================
            # Оставляем для обратной совместимости и миграции
            # ==================== ТАБЛИЦА: СОСТОЯНИЕ БОТОВ ====================
            # ВАЖНО: Старая таблица bots_state с value_json БОЛЬШЕ НЕ СОЗДАЕТСЯ!
            # Все данные хранятся в нормализованных таблицах: bots и auto_bot_config
            # Старая таблица будет удалена после миграции данных (см. миграцию ниже)
            
            # ==================== ТАБЛИЦА: РЕЕСТР ПОЗИЦИЙ ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: одна строка = одна позиция
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_positions_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    managed_by_bot INTEGER DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для bot_positions_registry
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON bot_positions_registry(bot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON bot_positions_registry(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_side ON bot_positions_registry(side)")
            
            # ==================== ТАБЛИЦА: RSI КЭШ МЕТАДАННЫЕ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: метаданные кэша в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rsi_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_coins INTEGER DEFAULT 0,
                    successful_coins INTEGER DEFAULT 0,
                    failed_coins INTEGER DEFAULT 0,
                    extra_stats_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для rsi_cache
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_timestamp ON rsi_cache(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_created ON rsi_cache(created_at)")
            
            # ==================== ТАБЛИЦА: RSI КЭШ ДАННЫЕ МОНЕТ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: одна строка = одна монета со всеми полями
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rsi_cache_coins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    rsi6h REAL,
                    trend6h TEXT,
                    rsi_zone TEXT,
                    signal TEXT,
                    price REAL,
                    change24h REAL,
                    last_update TEXT,
                    blocked_by_scope INTEGER DEFAULT 0,
                    has_existing_position INTEGER DEFAULT 0,
                    is_mature INTEGER DEFAULT 1,
                    blocked_by_exit_scam INTEGER DEFAULT 0,
                    blocked_by_rsi_time INTEGER DEFAULT 0,
                    trading_status TEXT,
                    is_delisting INTEGER DEFAULT 0,
                    trend_analysis_json TEXT,
                    enhanced_rsi_json TEXT,
                    time_filter_info_json TEXT,
                    exit_scam_info_json TEXT,
                    extra_coin_data_json TEXT,
                    FOREIGN KEY (cache_id) REFERENCES rsi_cache(id) ON DELETE CASCADE
                )
            """)
            
            # Индексы для rsi_cache_coins
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_cache_id ON rsi_cache_coins(cache_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_symbol ON rsi_cache_coins(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_rsi6h ON rsi_cache_coins(rsi6h)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_signal ON rsi_cache_coins(signal)")
            
            # ==================== ТАБЛИЦА: СОСТОЯНИЕ ПРОЦЕССОВ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: одна строка = один процесс со всеми полями
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS process_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    process_name TEXT UNIQUE NOT NULL,
                    active INTEGER DEFAULT 0,
                    initialized INTEGER DEFAULT 0,
                    last_update TEXT,
                    last_check TEXT,
                    last_save TEXT,
                    last_sync TEXT,
                    update_count INTEGER DEFAULT 0,
                    check_count INTEGER DEFAULT 0,
                    save_count INTEGER DEFAULT 0,
                    connection_count INTEGER DEFAULT 0,
                    signals_processed INTEGER DEFAULT 0,
                    bots_created INTEGER DEFAULT 0,
                    last_error TEXT,
                    extra_process_data_json TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для process_state
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_name ON process_state(process_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_active ON process_state(active)")
            
            # ==================== ТАБЛИЦА: ИНДИВИДУАЛЬНЫЕ НАСТРОЙКИ МОНЕТ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все настройки в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS individual_coin_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    -- RSI пороги входа
                    rsi_long_threshold INTEGER,
                    rsi_short_threshold INTEGER,
                    -- RSI пороги выхода
                    rsi_exit_long_with_trend INTEGER,
                    rsi_exit_long_against_trend INTEGER,
                    rsi_exit_short_with_trend INTEGER,
                    rsi_exit_short_against_trend INTEGER,
                    -- Риск-менеджмент
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    -- Trailing stop
                    trailing_stop_activation REAL,
                    trailing_stop_distance REAL,
                    trailing_take_distance REAL,
                    trailing_update_interval REAL,
                    -- Break even
                    break_even_trigger REAL,
                    break_even_protection REAL,
                    -- Ограничения
                    max_position_hours REAL,
                    -- RSI временной фильтр
                    rsi_time_filter_enabled INTEGER DEFAULT 0,
                    rsi_time_filter_candles INTEGER,
                    rsi_time_filter_upper INTEGER,
                    rsi_time_filter_lower INTEGER,
                    -- Фильтры тренда
                    avoid_down_trend INTEGER DEFAULT 0,
                    -- Дополнительные настройки в JSON (для будущих расширений)
                    extra_settings_json TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для individual_coin_settings
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin_settings_symbol ON individual_coin_settings(symbol)")
            
            # ==================== ТАБЛИЦА: ЗРЕЛЫЕ МОНЕТЫ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все поля в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mature_coins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    is_mature INTEGER DEFAULT 0,
                    candles_count INTEGER,
                    min_required INTEGER,
                    config_min_rsi_low INTEGER,
                    config_max_rsi_high INTEGER,
                    extra_maturity_data_json TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для mature_coins
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_symbol ON mature_coins(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_timestamp ON mature_coins(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_is_mature ON mature_coins(is_mature)")
            
            # ==================== ТАБЛИЦА: КЭШ ПРОВЕРКИ ЗРЕЛОСТИ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все параметры конфигурации в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maturity_check_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coins_count INTEGER NOT NULL,
                    min_candles INTEGER,
                    min_rsi_low INTEGER,
                    max_rsi_high INTEGER,
                    extra_config_json TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # ==================== ТАБЛИЦА: ДЕЛИСТИРОВАННЫЕ МОНЕТЫ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS delisted (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    delisted_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для delisted
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_delisted_symbol ON delisted(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_delisted_date ON delisted(delisted_at)")
            
            # ==================== ТАБЛИЦА: КЭШ СВЕЧЕЙ (НОРМАЛИЗОВАННАЯ) ====================
            # Метаданные кэша свечей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '6h',
                    candles_count INTEGER DEFAULT 0,
                    first_candle_time INTEGER,
                    last_candle_time INTEGER,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для candles_cache
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_symbol ON candles_cache(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_updated ON candles_cache(updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_timeframe ON candles_cache(timeframe)")
            
            # ==================== ТАБЛИЦА: ДАННЫЕ СВЕЧЕЙ КЭША (НОРМАЛИЗОВАННАЯ) ====================
            # Отдельная таблица для хранения свечей (вместо JSON)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_cache_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_id INTEGER NOT NULL,
                    time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    FOREIGN KEY (cache_id) REFERENCES candles_cache(id) ON DELETE CASCADE
                )
            """)
            
            # Индексы для candles_cache_data
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_id ON candles_cache_data(cache_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_time ON candles_cache_data(time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_time ON candles_cache_data(cache_id, time)")
            
            # ==================== ТАБЛИЦА: ИСТОРИЯ ТОРГОВЛИ БОТОВ ====================
            # Нормализованная структура для хранения истории всех сделок ботов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_trades_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_timestamp REAL,
                    exit_timestamp REAL,
                    position_size_usdt REAL,
                    position_size_coins REAL,
                    pnl REAL,
                    roi REAL,
                    status TEXT NOT NULL DEFAULT 'CLOSED',
                    close_reason TEXT,
                    decision_source TEXT DEFAULT 'SCRIPT',
                    ai_decision_id TEXT,
                    ai_confidence REAL,
                    entry_rsi REAL,
                    exit_rsi REAL,
                    entry_trend TEXT,
                    exit_trend TEXT,
                    entry_volatility REAL,
                    entry_volume_ratio REAL,
                    is_successful INTEGER DEFAULT 0,
                    is_simulated INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'bot',
                    order_id TEXT,
                    extra_data_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для bot_trades_history
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_bot_id ON bot_trades_history(bot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_symbol ON bot_trades_history(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_status ON bot_trades_history(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_entry_time ON bot_trades_history(entry_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_exit_time ON bot_trades_history(exit_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_decision_source ON bot_trades_history(decision_source)")
            
            # ==================== ТАБЛИЦА: МЕТАДАННЫЕ БД ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для db_metadata
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_db_metadata_key ON db_metadata(key)")
            
            # Если БД новая - устанавливаем флаг что миграция не выполнена
            if not db_exists:
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT OR IGNORE INTO db_metadata (key, value, updated_at, created_at)
                    VALUES ('json_migration_completed', '0', ?, ?)
                """, (now, now))
                logger.info("✅ Все таблицы и индексы созданы в новой базе данных")
            else:
                logger.debug("✅ Все таблицы и индексы проверены")
            
            conn.commit()
    
    def _migrate_schema(self, cursor, conn):
        """
        Миграция схемы БД: добавляет новые поля если их нет
        
        Это безопасная операция - она только добавляет новые поля,
        не удаляет существующие данные или таблицы.
        
        Пример использования:
        ```python
        # Проверяем наличие поля
        try:
            cursor.execute("SELECT new_field FROM bots_state LIMIT 1")
        except sqlite3.OperationalError:
            # Поля нет - добавляем
            logger.info("📦 Миграция: добавляем new_field в bots_state")
            cursor.execute("ALTER TABLE bots_state ADD COLUMN new_field TEXT")
        ```
        """
        try:
            # ==================== МИГРАЦИЯ: bot_positions_registry из EAV в нормализованный формат ====================
            # Проверяем, есть ли старая структура (EAV формат с position_data_json)
            try:
                cursor.execute("SELECT position_data_json FROM bot_positions_registry LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая EAV структура bot_positions_registry, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT bot_id, symbol, position_data_json, updated_at, created_at FROM bot_positions_registry")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Группируем данные по bot_id (в EAV формате один bot_id имеет несколько строк)
                    positions_dict = {}
                    for row in old_rows:
                        bot_id = row[0]
                        attr_name = row[1]  # Это название атрибута (entry_price, quantity и т.д.)
                        attr_value = row[2]  # Это значение атрибута в JSON
                        updated_at = row[3]
                        created_at = row[4]
                        
                        if bot_id not in positions_dict:
                            positions_dict[bot_id] = {
                                'updated_at': updated_at,
                                'created_at': created_at
                            }
                        
                        # Парсим значение атрибута
                        try:
                            value = json.loads(attr_value)
                            positions_dict[bot_id][attr_name] = value
                        except:
                            positions_dict[bot_id][attr_name] = attr_value
                    
                    # Удаляем старую таблицу
                    cursor.execute("DROP TABLE IF EXISTS bot_positions_registry")
                    
                    # Создаем новую таблицу с нормализованной структурой
                    cursor.execute("""
                        CREATE TABLE bot_positions_registry (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            bot_id TEXT NOT NULL UNIQUE,
                            symbol TEXT NOT NULL,
                            side TEXT NOT NULL,
                            entry_price REAL NOT NULL,
                            quantity REAL NOT NULL,
                            opened_at TEXT NOT NULL,
                            managed_by_bot INTEGER DEFAULT 1,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    # Создаем индексы
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON bot_positions_registry(bot_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON bot_positions_registry(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_side ON bot_positions_registry(side)")
                    
                    # Вставляем данные в новую структуру
                    migrated_count = 0
                    for bot_id, pos_data in positions_dict.items():
                        try:
                            # Извлекаем значения полей
                            symbol = pos_data.get('symbol', '')
                            side = pos_data.get('side', 'LONG')
                            entry_price = pos_data.get('entry_price', 0.0)
                            quantity = pos_data.get('quantity', 0.0)
                            opened_at = pos_data.get('opened_at', datetime.now().isoformat())
                            managed_by_bot = 1 if pos_data.get('managed_by_bot', True) else 0
                            updated_at = pos_data.get('updated_at', datetime.now().isoformat())
                            created_at = pos_data.get('created_at', datetime.now().isoformat())
                            
                            # Вставляем в новую таблицу
                            cursor.execute("""
                                INSERT INTO bot_positions_registry 
                                (bot_id, symbol, side, entry_price, quantity, opened_at, managed_by_bot, updated_at, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (bot_id, symbol, side, entry_price, quantity, opened_at, managed_by_bot, updated_at, created_at))
                            migrated_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции позиции {bot_id}: {e}")
                            continue
                    
                    logger.info(f"✅ Миграция bot_positions_registry завершена: {migrated_count} позиций мигрировано из EAV в нормализованный формат")
                else:
                    # Таблица пуста, просто пересоздаем с новой структурой
                    cursor.execute("DROP TABLE IF EXISTS bot_positions_registry")
                    cursor.execute("""
                        CREATE TABLE bot_positions_registry (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            bot_id TEXT NOT NULL UNIQUE,
                            symbol TEXT NOT NULL,
                            side TEXT NOT NULL,
                            entry_price REAL NOT NULL,
                            quantity REAL NOT NULL,
                            opened_at TEXT NOT NULL,
                            managed_by_bot INTEGER DEFAULT 1,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON bot_positions_registry(bot_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON bot_positions_registry(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_side ON bot_positions_registry(side)")
                    logger.info("✅ Таблица bot_positions_registry пересоздана с нормализованной структурой")
                    
            except sqlite3.OperationalError:
                # Таблица не существует или уже новая структура - ничего не делаем
                pass
            
            # ==================== МИГРАЦИЯ: bots_state из JSON в нормализованные таблицы ====================
            # Проверяем, есть ли данные в старой таблице bots_state
            try:
                cursor.execute("SELECT value_json FROM bots_state WHERE key = 'main'")
                row = cursor.fetchone()
                
                if row:
                    # Проверяем, мигрированы ли уже данные
                    cursor.execute("SELECT COUNT(*) FROM bots")
                    bots_count = cursor.fetchone()[0]
                    
                    if bots_count == 0:
                        # Данные еще не мигрированы
                        logger.info("📦 Обнаружены данные в bots_state, выполняю миграцию в нормализованные таблицы...")
                        
                        state_data = json.loads(row[0])
                        bots_data = state_data.get('bots', {})
                        auto_bot_config = state_data.get('auto_bot_config', {})
                        
                        # Мигрируем ботов
                        now = datetime.now().isoformat()
                        migrated_bots = 0
                        
                        for symbol, bot_data in bots_data.items():
                            try:
                                # Извлекаем все поля бота
                                extra_data = {}
                                
                                # Вспомогательная функция для безопасной конвертации в float
                                def safe_float(value, default=None):
                                    if value is None:
                                        return default
                                    if isinstance(value, (int, float)):
                                        return float(value)
                                    if isinstance(value, str):
                                        value = value.strip()
                                        if value == '' or value.lower() == 'none':
                                            return default
                                        try:
                                            return float(value)
                                        except (ValueError, TypeError):
                                            return default
                                    return default
                                
                                # Вспомогательная функция для безопасной конвертации в int
                                def safe_int(value, default=0):
                                    if value is None:
                                        return default
                                    if isinstance(value, (int, float)):
                                        return int(value)
                                    if isinstance(value, str):
                                        value = value.strip()
                                        if value == '' or value.lower() == 'none':
                                            return default
                                        try:
                                            return int(float(value))
                                        except (ValueError, TypeError):
                                            return default
                                    return default
                                
                                # Основные поля
                                status = bot_data.get('status', 'idle')
                                auto_managed = 1 if bot_data.get('auto_managed', False) else 0
                                volume_mode = bot_data.get('volume_mode', 'usdt')
                                volume_value = safe_float(bot_data.get('volume_value'))
                                
                                # Позиция
                                entry_price = safe_float(bot_data.get('entry_price'))
                                entry_time = bot_data.get('entry_time') or bot_data.get('position_start_time')
                                entry_timestamp = safe_float(bot_data.get('entry_timestamp'))
                                position_side = bot_data.get('position_side')
                                position_size = safe_float(bot_data.get('position_size'))
                                position_size_coins = safe_float(bot_data.get('position_size_coins'))
                                position_start_time = bot_data.get('position_start_time')
                                
                                # PnL
                                unrealized_pnl = safe_float(bot_data.get('unrealized_pnl'), 0.0)
                                unrealized_pnl_usdt = safe_float(bot_data.get('unrealized_pnl_usdt'), 0.0)
                                realized_pnl = safe_float(bot_data.get('realized_pnl'), 0.0)
                                
                                # Другие поля
                                leverage = safe_float(bot_data.get('leverage'), 1.0)
                                margin_usdt = safe_float(bot_data.get('margin_usdt'))
                                max_profit_achieved = safe_float(bot_data.get('max_profit_achieved'), 0.0)
                                
                                # Trailing stop
                                trailing_stop_price = safe_float(bot_data.get('trailing_stop_price'))
                                trailing_activation_threshold = safe_float(bot_data.get('trailing_activation_threshold'))
                                trailing_activation_profit = safe_float(bot_data.get('trailing_activation_profit'), 0.0)
                                trailing_locked_profit = safe_float(bot_data.get('trailing_locked_profit'), 0.0)
                                trailing_active = 1 if bot_data.get('trailing_active', False) else 0
                                trailing_max_profit_usdt = safe_float(bot_data.get('trailing_max_profit_usdt'), 0.0)
                                trailing_step_usdt = safe_float(bot_data.get('trailing_step_usdt'))
                                trailing_step_price = safe_float(bot_data.get('trailing_step_price'))
                                trailing_steps = safe_int(bot_data.get('trailing_steps'), 0)
                                trailing_reference_price = safe_float(bot_data.get('trailing_reference_price'))
                                trailing_last_update_ts = safe_float(bot_data.get('trailing_last_update_ts'), 0.0)
                                trailing_take_profit_price = safe_float(bot_data.get('trailing_take_profit_price'))
                                
                                # Break even
                                break_even_activated = 1 if bot_data.get('break_even_activated', False) else 0
                                break_even_stop_price = safe_float(bot_data.get('break_even_stop_price'))
                                
                                # Другие
                                order_id = bot_data.get('order_id')
                                current_price = safe_float(bot_data.get('current_price'))
                                last_price = safe_float(bot_data.get('last_price'))
                                last_rsi = safe_float(bot_data.get('last_rsi'))
                                last_trend = bot_data.get('last_trend')
                                last_signal_time = bot_data.get('last_signal_time')
                                last_bar_timestamp = safe_float(bot_data.get('last_bar_timestamp'))
                                entry_trend = bot_data.get('entry_trend')
                                opened_by_autobot = 1 if bot_data.get('opened_by_autobot', False) else 0
                                bot_id = bot_data.get('id')
                                
                                # Собираем все остальные поля в extra_data_json
                                known_fields = {
                                    'symbol', 'status', 'auto_managed', 'volume_mode', 'volume_value',
                                    'entry_price', 'entry_time', 'entry_timestamp', 'position_side',
                                    'position_size', 'position_size_coins', 'position_start_time',
                                    'unrealized_pnl', 'unrealized_pnl_usdt', 'realized_pnl', 'leverage',
                                    'margin_usdt', 'max_profit_achieved', 'trailing_stop_price',
                                    'trailing_activation_threshold', 'trailing_activation_profit',
                                    'trailing_locked_profit', 'trailing_active', 'trailing_max_profit_usdt',
                                    'trailing_step_usdt', 'trailing_step_price', 'trailing_steps',
                                    'trailing_reference_price', 'trailing_last_update_ts', 'trailing_take_profit_price',
                                    'break_even_activated', 'break_even_stop_price', 'order_id',
                                    'current_price', 'last_price', 'last_rsi', 'last_trend',
                                    'last_signal_time', 'last_bar_timestamp', 'entry_trend',
                                    'opened_by_autobot', 'id', 'position', 'rsi_data', 'scaling_enabled',
                                    'scaling_levels', 'scaling_current_level', 'scaling_group_id', 'created_at'
                                }
                                
                                for key, value in bot_data.items():
                                    if key not in known_fields:
                                        extra_data[key] = value
                                
                                # Сохраняем сложные структуры в extra_data
                                if bot_data.get('position'):
                                    extra_data['position'] = bot_data['position']
                                if bot_data.get('rsi_data'):
                                    extra_data['rsi_data'] = bot_data['rsi_data']
                                
                                extra_data_json = json.dumps(extra_data) if extra_data else None
                                created_at = bot_data.get('created_at', now)
                                
                                # Вставляем в новую таблицу (45 столбцов: symbol до created_at)
                                cursor.execute("""
                                    INSERT INTO bots (
                                        symbol, status, auto_managed, volume_mode, volume_value,
                                        entry_price, entry_time, entry_timestamp, position_side,
                                        position_size, position_size_coins, position_start_time,
                                        unrealized_pnl, unrealized_pnl_usdt, realized_pnl, leverage,
                                        margin_usdt, max_profit_achieved, trailing_stop_price,
                                        trailing_activation_threshold, trailing_activation_profit,
                                        trailing_locked_profit, trailing_active, trailing_max_profit_usdt,
                                        trailing_step_usdt, trailing_step_price, trailing_steps,
                                        trailing_reference_price, trailing_last_update_ts, trailing_take_profit_price,
                                        break_even_activated, break_even_stop_price, order_id,
                                        current_price, last_price, last_rsi, last_trend,
                                        last_signal_time, last_bar_timestamp, entry_trend,
                                        opened_by_autobot, bot_id, extra_data_json,
                                        updated_at, created_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    symbol, status, auto_managed, volume_mode, volume_value,
                                    entry_price, entry_time, entry_timestamp, position_side,
                                    position_size, position_size_coins, position_start_time,
                                    unrealized_pnl, unrealized_pnl_usdt, realized_pnl, leverage,
                                    margin_usdt, max_profit_achieved, trailing_stop_price,
                                    trailing_activation_threshold, trailing_activation_profit,
                                    trailing_locked_profit, trailing_active, trailing_max_profit_usdt,
                                    trailing_step_usdt, trailing_step_price, trailing_steps,
                                    trailing_reference_price, trailing_last_update_ts, trailing_take_profit_price,
                                    break_even_activated, break_even_stop_price, order_id,
                                    current_price, last_price, last_rsi, last_trend,
                                    last_signal_time, last_bar_timestamp, entry_trend,
                                    opened_by_autobot, bot_id, extra_data_json,
                                    now, created_at or now
                                ))
                                migrated_bots += 1
                            except Exception as e:
                                logger.warning(f"⚠️ Ошибка миграции бота {symbol}: {e}")
                                continue
                        
                        # Мигрируем auto_bot_config
                        if auto_bot_config:
                            for key, value in auto_bot_config.items():
                                try:
                                    cursor.execute("""
                                        INSERT OR REPLACE INTO auto_bot_config (key, value, updated_at, created_at)
                                        VALUES (?, ?, ?, ?)
                                    """, (key, json.dumps(value) if not isinstance(value, (str, int, float, bool)) else str(value), now, now))
                                except Exception as e:
                                    logger.warning(f"⚠️ Ошибка миграции auto_bot_config.{key}: {e}")
                        
                        logger.info(f"✅ Миграция bots_state завершена: {migrated_bots} ботов мигрировано из JSON в нормализованные таблицы")
                        
                        # Удаляем старую таблицу bots_state после успешной миграции
                        try:
                            cursor.execute("DROP TABLE IF EXISTS bots_state")
                            logger.info("🗑️ Старая таблица bots_state удалена (данные мигрированы в нормализованные таблицы)")
                        except Exception as drop_error:
                            logger.warning(f"⚠️ Не удалось удалить старую таблицу bots_state: {drop_error}")
                        
                        # ВСЕГДА пытаемся удалить старую таблицу bots_state после миграции
                        try:
                            cursor.execute("DROP TABLE IF EXISTS bots_state")
                            logger.info("🗑️ Старая таблица bots_state удалена после миграции")
                        except Exception:
                            pass  # Игнорируем ошибки - возможно таблица уже удалена
                    else:
                        logger.debug("ℹ️ Данные bots уже мигрированы")
                        
                        # ВСЕГДА удаляем старую таблицу bots_state - данные уже в нормализованных таблицах
                        try:
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bots_state'")
                            if cursor.fetchone():
                                cursor.execute("DROP TABLE IF EXISTS bots_state")
                                logger.info("🗑️ Старая таблица bots_state удалена (данные мигрированы в таблицы bots и auto_bot_config)")
                        except Exception as cleanup_error:
                            logger.debug(f"⚠️ Ошибка очистки старой таблицы bots_state: {cleanup_error}")
                        
            except sqlite3.OperationalError:
                # Таблица bots_state не существует - это нормально, значит уже удалена или не создавалась
                logger.debug("ℹ️ Таблица bots_state не существует (уже удалена или не создавалась)")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции bots_state: {e}")
            
            # ==================== МИГРАЦИЯ: candles_cache из JSON в нормализованные таблицы ====================
            # Проверяем, есть ли старая структура (с candles_json)
            try:
                cursor.execute("SELECT candles_json FROM candles_cache LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая JSON структура candles_cache, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT id, symbol, candles_json, timeframe, updated_at, created_at FROM candles_cache")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Создаем новую таблицу candles_cache_data если её еще нет
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS candles_cache_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cache_id INTEGER NOT NULL,
                            time INTEGER NOT NULL,
                            open REAL NOT NULL,
                            high REAL NOT NULL,
                            low REAL NOT NULL,
                            close REAL NOT NULL,
                            volume REAL NOT NULL,
                            FOREIGN KEY (cache_id) REFERENCES candles_cache(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Создаем индексы для candles_cache_data
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_id ON candles_cache_data(cache_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_time ON candles_cache_data(time)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_time ON candles_cache_data(cache_id, time)")
                    
                    # Добавляем новые колонки в candles_cache если их нет
                    try:
                        cursor.execute("SELECT candles_count FROM candles_cache LIMIT 1")
                    except sqlite3.OperationalError:
                        cursor.execute("ALTER TABLE candles_cache ADD COLUMN candles_count INTEGER DEFAULT 0")
                        cursor.execute("ALTER TABLE candles_cache ADD COLUMN first_candle_time INTEGER")
                        cursor.execute("ALTER TABLE candles_cache ADD COLUMN last_candle_time INTEGER")
                    
                    migrated_count = 0
                    for old_row in old_rows:
                        cache_id = old_row['id']
                        symbol = old_row['symbol']
                        candles_json = old_row['candles_json']
                        timeframe = old_row['timeframe']
                        updated_at = old_row['updated_at']
                        created_at = old_row['created_at']
                        
                        try:
                            candles = json.loads(candles_json) if candles_json else []
                            
                            if candles:
                                # Определяем временные границы
                                times = [c.get('time') for c in candles if c.get('time')]
                                first_time = min(times) if times else None
                                last_time = max(times) if times else None
                                
                                # Обновляем метаданные в candles_cache
                                cursor.execute("""
                                    UPDATE candles_cache 
                                    SET candles_count = ?, first_candle_time = ?, last_candle_time = ?
                                    WHERE id = ?
                                """, (len(candles), first_time, last_time, cache_id))
                                
                                # Удаляем старые свечи для этого cache_id
                                cursor.execute("DELETE FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
                                
                                # Вставляем свечи в нормализованную таблицу
                                for candle in candles:
                                    cursor.execute("""
                                        INSERT INTO candles_cache_data 
                                        (cache_id, time, open, high, low, close, volume)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        cache_id,
                                        candle.get('time'),
                                        candle.get('open'),
                                        candle.get('high'),
                                        candle.get('low'),
                                        candle.get('close'),
                                        candle.get('volume', 0)
                                    ))
                                
                                migrated_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции свечей для {symbol}: {e}")
                            continue
                    
                    # Удаляем колонку candles_json после миграции
                    # SQLite не поддерживает DROP COLUMN напрямую, нужно пересоздать таблицу
                    try:
                        # Сохраняем данные из старой таблицы
                        cursor.execute("""
                            SELECT id, symbol, timeframe, candles_count, first_candle_time, last_candle_time, updated_at, created_at
                            FROM candles_cache
                        """)
                        old_data = cursor.fetchall()
                        
                        # Удаляем старую таблицу
                        cursor.execute("DROP TABLE IF EXISTS candles_cache")
                        
                        # Создаем новую таблицу без candles_json
                        cursor.execute("""
                            CREATE TABLE candles_cache (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                symbol TEXT UNIQUE NOT NULL,
                                timeframe TEXT NOT NULL DEFAULT '6h',
                                candles_count INTEGER DEFAULT 0,
                                first_candle_time INTEGER,
                                last_candle_time INTEGER,
                                updated_at TEXT NOT NULL,
                                created_at TEXT NOT NULL
                            )
                        """)
                        
                        # Восстанавливаем индексы
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_symbol ON candles_cache(symbol)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_updated ON candles_cache(updated_at)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_timeframe ON candles_cache(timeframe)")
                        
                        # Восстанавливаем данные
                        for row in old_data:
                            cursor.execute("""
                                INSERT INTO candles_cache 
                                (id, symbol, timeframe, candles_count, first_candle_time, last_candle_time, updated_at, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, row)
                        
                        logger.info(f"✅ Миграция candles_cache завершена: {migrated_count} символов мигрировано из JSON в нормализованные таблицы, колонка candles_json удалена")
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка удаления колонки candles_json: {e}")
                        # Продолжаем работу, даже если не удалось удалить колонку
                        
            except sqlite3.OperationalError:
                # Колонка candles_json не существует - значит уже мигрировано или новая структура
                logger.debug("ℹ️ candles_cache уже в нормализованном формате")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции candles_cache: {e}")
            
            # ==================== МИГРАЦИЯ: individual_coin_settings из JSON в нормализованные столбцы ====================
            # Проверяем, есть ли старая структура (с settings_json)
            try:
                cursor.execute("SELECT settings_json FROM individual_coin_settings LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая JSON структура individual_coin_settings, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT symbol, settings_json, updated_at, created_at FROM individual_coin_settings")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Удаляем старую таблицу
                    cursor.execute("DROP TABLE IF EXISTS individual_coin_settings")
                    
                    # Создаем новую таблицу с нормализованной структурой
                    cursor.execute("""
                        CREATE TABLE individual_coin_settings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT UNIQUE NOT NULL,
                            rsi_long_threshold INTEGER,
                            rsi_short_threshold INTEGER,
                            rsi_exit_long_with_trend INTEGER,
                            rsi_exit_long_against_trend INTEGER,
                            rsi_exit_short_with_trend INTEGER,
                            rsi_exit_short_against_trend INTEGER,
                            max_loss_percent REAL,
                            take_profit_percent REAL,
                            trailing_stop_activation REAL,
                            trailing_stop_distance REAL,
                            trailing_take_distance REAL,
                            trailing_update_interval REAL,
                            break_even_trigger REAL,
                            break_even_protection REAL,
                            max_position_hours REAL,
                            rsi_time_filter_enabled INTEGER DEFAULT 0,
                            rsi_time_filter_candles INTEGER,
                            rsi_time_filter_upper INTEGER,
                            rsi_time_filter_lower INTEGER,
                            avoid_down_trend INTEGER DEFAULT 0,
                            extra_settings_json TEXT,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    # Создаем индекс
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin_settings_symbol ON individual_coin_settings(symbol)")
                    
                    # Мигрируем данные
                    migrated_count = 0
                    for row in old_rows:
                        try:
                            symbol = row[0]
                            settings_json = row[1]
                            updated_at = row[2]
                            created_at = row[3]
                            
                            # Парсим JSON
                            settings = json.loads(settings_json)
                            
                            # Извлекаем основные поля
                            extra_settings = {}
                            known_fields = {
                                'rsi_long_threshold', 'rsi_short_threshold',
                                'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                                'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                                'max_loss_percent', 'take_profit_percent',
                                'trailing_stop_activation', 'trailing_stop_distance',
                                'trailing_take_distance', 'trailing_update_interval',
                                'break_even_trigger', 'break_even_protection',
                                'max_position_hours', 'rsi_time_filter_enabled',
                                'rsi_time_filter_candles', 'rsi_time_filter_upper',
                                'rsi_time_filter_lower', 'avoid_down_trend'
                            }
                            
                            for key, value in settings.items():
                                if key not in known_fields:
                                    extra_settings[key] = value
                            
                            extra_settings_json = json.dumps(extra_settings) if extra_settings else None
                            
                            # Вставляем в новую таблицу
                            cursor.execute("""
                                INSERT INTO individual_coin_settings (
                                    symbol, rsi_long_threshold, rsi_short_threshold,
                                    rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                                    rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                                    max_loss_percent, take_profit_percent,
                                    trailing_stop_activation, trailing_stop_distance,
                                    trailing_take_distance, trailing_update_interval,
                                    break_even_trigger, break_even_protection,
                                    max_position_hours, rsi_time_filter_enabled,
                                    rsi_time_filter_candles, rsi_time_filter_upper,
                                    rsi_time_filter_lower, avoid_down_trend,
                                    extra_settings_json, updated_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol,
                                settings.get('rsi_long_threshold'),
                                settings.get('rsi_short_threshold'),
                                settings.get('rsi_exit_long_with_trend'),
                                settings.get('rsi_exit_long_against_trend'),
                                settings.get('rsi_exit_short_with_trend'),
                                settings.get('rsi_exit_short_against_trend'),
                                settings.get('max_loss_percent'),
                                settings.get('take_profit_percent'),
                                settings.get('trailing_stop_activation'),
                                settings.get('trailing_stop_distance'),
                                settings.get('trailing_take_distance'),
                                settings.get('trailing_update_interval'),
                                settings.get('break_even_trigger'),
                                settings.get('break_even_protection'),
                                settings.get('max_position_hours'),
                                1 if settings.get('rsi_time_filter_enabled') else 0,
                                settings.get('rsi_time_filter_candles'),
                                settings.get('rsi_time_filter_upper'),
                                settings.get('rsi_time_filter_lower'),
                                1 if settings.get('avoid_down_trend') else 0,
                                extra_settings_json,
                                updated_at,
                                created_at
                            ))
                            migrated_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции настроек для {symbol}: {e}")
                            continue
                    
                    logger.info(f"✅ Миграция individual_coin_settings завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
                else:
                    # Таблица пуста, просто пересоздаем с новой структурой
                    cursor.execute("DROP TABLE IF EXISTS individual_coin_settings")
                    cursor.execute("""
                        CREATE TABLE individual_coin_settings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT UNIQUE NOT NULL,
                            rsi_long_threshold INTEGER,
                            rsi_short_threshold INTEGER,
                            rsi_exit_long_with_trend INTEGER,
                            rsi_exit_long_against_trend INTEGER,
                            rsi_exit_short_with_trend INTEGER,
                            rsi_exit_short_against_trend INTEGER,
                            max_loss_percent REAL,
                            take_profit_percent REAL,
                            trailing_stop_activation REAL,
                            trailing_stop_distance REAL,
                            trailing_take_distance REAL,
                            trailing_update_interval REAL,
                            break_even_trigger REAL,
                            break_even_protection REAL,
                            max_position_hours REAL,
                            rsi_time_filter_enabled INTEGER DEFAULT 0,
                            rsi_time_filter_candles INTEGER,
                            rsi_time_filter_upper INTEGER,
                            rsi_time_filter_lower INTEGER,
                            avoid_down_trend INTEGER DEFAULT 0,
                            extra_settings_json TEXT,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin_settings_symbol ON individual_coin_settings(symbol)")
                    logger.info("✅ Таблица individual_coin_settings пересоздана с нормализованной структурой")
                    
            except sqlite3.OperationalError:
                # Таблица не существует или уже новая структура - ничего не делаем
                pass
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции individual_coin_settings: {e}")
            
            # ==================== МИГРАЦИЯ: mature_coins из JSON в нормализованные столбцы ====================
            # Проверяем, есть ли старая структура (с maturity_data_json)
            try:
                cursor.execute("SELECT maturity_data_json FROM mature_coins LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая JSON структура mature_coins, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT symbol, timestamp, maturity_data_json, updated_at, created_at FROM mature_coins")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Удаляем старую таблицу
                    cursor.execute("DROP TABLE IF EXISTS mature_coins")
                    
                    # Создаем новую таблицу с нормализованной структурой
                    cursor.execute("""
                        CREATE TABLE mature_coins (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT UNIQUE NOT NULL,
                            timestamp REAL NOT NULL,
                            is_mature INTEGER DEFAULT 0,
                            candles_count INTEGER,
                            min_required INTEGER,
                            config_min_rsi_low INTEGER,
                            config_max_rsi_high INTEGER,
                            extra_maturity_data_json TEXT,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    # Создаем индексы
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_symbol ON mature_coins(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_timestamp ON mature_coins(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_is_mature ON mature_coins(is_mature)")
                    
                    # Мигрируем данные
                    migrated_count = 0
                    for row in old_rows:
                        try:
                            symbol = row[0]
                            timestamp = row[1]
                            maturity_data_json = row[2]
                            updated_at = row[3]
                            created_at = row[4]
                            
                            # Парсим JSON
                            maturity_data = json.loads(maturity_data_json)
                            
                            # Извлекаем основные поля
                            is_mature = 1 if maturity_data.get('is_mature', False) else 0
                            details = maturity_data.get('details', {})
                            candles_count = details.get('candles_count')
                            min_required = details.get('min_required')
                            config_min_rsi_low = details.get('config_min_rsi_low')
                            config_max_rsi_high = details.get('config_max_rsi_high')
                            
                            # Собираем остальные поля в extra_maturity_data_json
                            extra_data = {}
                            known_fields = {'is_mature', 'details'}
                            for key, value in maturity_data.items():
                                if key not in known_fields:
                                    extra_data[key] = value
                            
                            # Также сохраняем неизвестные поля из details
                            known_details_fields = {'candles_count', 'min_required', 'config_min_rsi_low', 'config_max_rsi_high'}
                            for key, value in details.items():
                                if key not in known_details_fields:
                                    if 'extra_details' not in extra_data:
                                        extra_data['extra_details'] = {}
                                    extra_data['extra_details'][key] = value
                            
                            extra_maturity_data_json = json.dumps(extra_data) if extra_data else None
                            
                            # Вставляем в новую таблицу
                            cursor.execute("""
                                INSERT INTO mature_coins (
                                    symbol, timestamp, is_mature, candles_count,
                                    min_required, config_min_rsi_low, config_max_rsi_high,
                                    extra_maturity_data_json, updated_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol,
                                timestamp,
                                is_mature,
                                candles_count,
                                min_required,
                                config_min_rsi_low,
                                config_max_rsi_high,
                                extra_maturity_data_json,
                                updated_at,
                                created_at
                            ))
                            migrated_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции зрелой монеты {symbol}: {e}")
                            continue
                    
                    logger.info(f"✅ Миграция mature_coins завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
                else:
                    # Таблица пуста, просто пересоздаем с новой структурой
                    cursor.execute("DROP TABLE IF EXISTS mature_coins")
                    cursor.execute("""
                        CREATE TABLE mature_coins (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT UNIQUE NOT NULL,
                            timestamp REAL NOT NULL,
                            is_mature INTEGER DEFAULT 0,
                            candles_count INTEGER,
                            min_required INTEGER,
                            config_min_rsi_low INTEGER,
                            config_max_rsi_high INTEGER,
                            extra_maturity_data_json TEXT,
                            updated_at TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_symbol ON mature_coins(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_timestamp ON mature_coins(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mature_coins_is_mature ON mature_coins(is_mature)")
                    logger.info("✅ Таблица mature_coins пересоздана с нормализованной структурой")
                    
            except sqlite3.OperationalError:
                # Таблица не существует или уже новая структура - ничего не делаем
                pass
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции mature_coins: {e}")
            
            # ==================== МИГРАЦИЯ: maturity_check_cache из JSON в нормализованные столбцы ====================
            # Сначала проверяем и добавляем столбцы, если их нет
            new_fields_maturity = [
                ('min_candles', 'INTEGER'),
                ('min_rsi_low', 'INTEGER'),
                ('max_rsi_high', 'INTEGER'),
                ('extra_config_json', 'TEXT')
            ]
            for field_name, field_type in new_fields_maturity:
                try:
                    cursor.execute(f"SELECT {field_name} FROM maturity_check_cache LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в maturity_check_cache")
                    cursor.execute(f"ALTER TABLE maturity_check_cache ADD COLUMN {field_name} {field_type}")
            
            # Проверяем, есть ли старая структура (с config_hash как JSON)
            try:
                cursor.execute("SELECT config_hash FROM maturity_check_cache LIMIT 1")
                # Если запрос выполнился - проверяем структуру
                row = cursor.fetchone()
                if row and row[0]:
                    # Пытаемся распарсить config_hash как JSON
                    try:
                        config_data = json.loads(row[0])
                        # Если это словарь с min_candles/min_rsi_low/max_rsi_high - значит старая структура
                        if isinstance(config_data, dict) and ('min_candles' in config_data or 'min_rsi_low' in config_data):
                            logger.info("📦 Обнаружена старая JSON структура maturity_check_cache, выполняю миграцию...")
                            
                            # Загружаем все данные из старой структуры
                            cursor.execute("SELECT coins_count, config_hash, updated_at, created_at FROM maturity_check_cache")
                            old_rows = cursor.fetchall()
                            
                            if old_rows:
                                # Удаляем старую таблицу
                                cursor.execute("DROP TABLE IF EXISTS maturity_check_cache")
                                
                                # Создаем новую таблицу с нормализованной структурой
                                cursor.execute("""
                                    CREATE TABLE maturity_check_cache (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        coins_count INTEGER NOT NULL,
                                        min_candles INTEGER,
                                        min_rsi_low INTEGER,
                                        max_rsi_high INTEGER,
                                        extra_config_json TEXT,
                                        updated_at TEXT NOT NULL,
                                        created_at TEXT NOT NULL
                                    )
                                """)
                                
                                # Мигрируем данные
                                migrated_count = 0
                                for old_row in old_rows:
                                    try:
                                        coins_count = old_row[0]
                                        config_hash = old_row[1]
                                        updated_at = old_row[2]
                                        created_at = old_row[3]
                                        
                                        # Парсим JSON из config_hash
                                        config_data = json.loads(config_hash) if config_hash else {}
                                        
                                        # Извлекаем основные поля
                                        min_candles = config_data.get('min_candles')
                                        min_rsi_low = config_data.get('min_rsi_low')
                                        max_rsi_high = config_data.get('max_rsi_high')
                                        
                                        # Собираем остальные поля в extra_config_json
                                        extra_data = {}
                                        known_fields = {'min_candles', 'min_rsi_low', 'max_rsi_high'}
                                        for key, value in config_data.items():
                                            if key not in known_fields:
                                                extra_data[key] = value
                                        
                                        extra_config_json = json.dumps(extra_data) if extra_data else None
                                        
                                        # Вставляем в новую таблицу
                                        cursor.execute("""
                                            INSERT INTO maturity_check_cache (
                                                coins_count, min_candles, min_rsi_low, max_rsi_high,
                                                extra_config_json, updated_at, created_at
                                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                        """, (
                                            coins_count,
                                            min_candles,
                                            min_rsi_low,
                                            max_rsi_high,
                                            extra_config_json,
                                            updated_at,
                                            created_at
                                        ))
                                        migrated_count += 1
                                    except Exception as e:
                                        logger.warning(f"⚠️ Ошибка миграции кэша проверки зрелости: {e}")
                                        continue
                                
                                logger.info(f"✅ Миграция maturity_check_cache завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
                    except (json.JSONDecodeError, TypeError):
                        # config_hash не JSON или уже строка - пропускаем миграцию
                        pass
            except sqlite3.OperationalError:
                # Таблица не существует или уже новая структура - ничего не делаем
                pass
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции maturity_check_cache: {e}")
            
            # ==================== МИГРАЦИЯ: process_state из JSON в нормализованные столбцы ====================
            # Проверяем, есть ли столбец process_name (новая структура)
            try:
                cursor.execute("SELECT process_name FROM process_state LIMIT 1")
                # Столбец process_name существует - таблица уже в новой структуре
                logger.debug("ℹ️ Таблица process_state уже в нормализованной структуре")
            except sqlite3.OperationalError:
                # Столбца process_name нет - нужно проверить старую структуру или создать новую
                try:
                    # Проверяем, есть ли старая структура (с value_json и key)
                    cursor.execute("SELECT value_json FROM process_state WHERE key = 'main' LIMIT 1")
                    row = cursor.fetchone()
                    
                    if row:
                        # Есть старая структура - мигрируем
                        logger.info("📦 Обнаружена старая структура process_state, выполняю миграцию в нормализованные столбцы...")
                        
                        state_data = json.loads(row[0])
                        process_state_dict = state_data.get('process_state', {})
                        
                        # Удаляем старую таблицу
                        cursor.execute("DROP TABLE IF EXISTS process_state")
                        
                        # Создаем новую таблицу с нормализованной структурой
                        cursor.execute("""
                            CREATE TABLE process_state (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                process_name TEXT UNIQUE NOT NULL,
                                active INTEGER DEFAULT 0,
                                initialized INTEGER DEFAULT 0,
                                last_update TEXT,
                                last_check TEXT,
                                last_save TEXT,
                                last_sync TEXT,
                                update_count INTEGER DEFAULT 0,
                                check_count INTEGER DEFAULT 0,
                                save_count INTEGER DEFAULT 0,
                                connection_count INTEGER DEFAULT 0,
                                signals_processed INTEGER DEFAULT 0,
                                bots_created INTEGER DEFAULT 0,
                                last_error TEXT,
                                extra_process_data_json TEXT,
                                updated_at TEXT NOT NULL,
                                created_at TEXT NOT NULL
                            )
                        """)
                        
                        # Создаем индексы
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_name ON process_state(process_name)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_active ON process_state(active)")
                        
                        # Мигрируем данные
                        now = datetime.now().isoformat()
                        migrated_count = 0
                        
                        for process_name, process_data in process_state_dict.items():
                            try:
                                # Извлекаем поля процесса
                                active = 1 if process_data.get('active', False) else 0
                                initialized = 1 if process_data.get('initialized', False) else 0
                                last_update = process_data.get('last_update')
                                last_check = process_data.get('last_check')
                                last_save = process_data.get('last_save')
                                last_sync = process_data.get('last_sync')
                                update_count = process_data.get('update_count', 0)
                                check_count = process_data.get('check_count', 0)
                                save_count = process_data.get('save_count', 0)
                                connection_count = process_data.get('connection_count', 0)
                                signals_processed = process_data.get('signals_processed', 0)
                                bots_created = process_data.get('bots_created', 0)
                                last_error = process_data.get('last_error')
                                
                                # Собираем остальные поля в extra_process_data_json
                                extra_data = {}
                                known_fields = {
                                    'active', 'initialized', 'last_update', 'last_check',
                                    'last_save', 'last_sync', 'update_count', 'check_count',
                                    'save_count', 'connection_count', 'signals_processed',
                                    'bots_created', 'last_error'
                                }
                                
                                for key, value in process_data.items():
                                    if key not in known_fields:
                                        extra_data[key] = value
                                
                                extra_process_data_json = json.dumps(extra_data) if extra_data else None
                                
                                # Вставляем в новую таблицу
                                cursor.execute("""
                                    INSERT INTO process_state (
                                        process_name, active, initialized, last_update,
                                        last_check, last_save, last_sync, update_count,
                                        check_count, save_count, connection_count,
                                        signals_processed, bots_created, last_error,
                                        extra_process_data_json, updated_at, created_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    process_name, active, initialized, last_update,
                                    last_check, last_save, last_sync, update_count,
                                    check_count, save_count, connection_count,
                                    signals_processed, bots_created, last_error,
                                    extra_process_data_json, now, now
                                ))
                                migrated_count += 1
                            except Exception as e:
                                logger.warning(f"⚠️ Ошибка миграции процесса {process_name}: {e}")
                                continue
                        
                        logger.info(f"✅ Миграция process_state завершена: {migrated_count} процессов мигрировано из JSON в нормализованные столбцы")
                    else:
                        # Старой структуры нет, но и новой тоже нет - таблица пустая или не существует
                        # Проверяем, существует ли таблица вообще
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='process_state'")
                        if cursor.fetchone():
                            # Таблица существует, но без process_name - пересоздаем
                            logger.info("📦 Таблица process_state существует без process_name, пересоздаю...")
                            cursor.execute("DROP TABLE IF EXISTS process_state")
                            cursor.execute("""
                                CREATE TABLE process_state (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    process_name TEXT UNIQUE NOT NULL,
                                    active INTEGER DEFAULT 0,
                                    initialized INTEGER DEFAULT 0,
                                    last_update TEXT,
                                    last_check TEXT,
                                    last_save TEXT,
                                    last_sync TEXT,
                                    update_count INTEGER DEFAULT 0,
                                    check_count INTEGER DEFAULT 0,
                                    save_count INTEGER DEFAULT 0,
                                    connection_count INTEGER DEFAULT 0,
                                    signals_processed INTEGER DEFAULT 0,
                                    bots_created INTEGER DEFAULT 0,
                                    last_error TEXT,
                                    extra_process_data_json TEXT,
                                    updated_at TEXT NOT NULL,
                                    created_at TEXT NOT NULL
                                )
                            """)
                            cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_name ON process_state(process_name)")
                            cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_state_active ON process_state(active)")
                            logger.info("✅ Таблица process_state пересоздана с нормализованной структурой")
                except sqlite3.OperationalError:
                    # Таблица process_state не существует - ничего не делаем, она создастся при CREATE TABLE IF NOT EXISTS
                    pass
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка миграции process_state: {e}")
            
            # ==================== МИГРАЦИЯ: rsi_cache из JSON в нормализованные таблицы ====================
            # Проверяем, есть ли старая структура (с coins_data_json)
            try:
                cursor.execute("SELECT coins_data_json FROM rsi_cache LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая JSON структура rsi_cache, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT id, timestamp, coins_data_json, stats_json, created_at FROM rsi_cache")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Удаляем старые таблицы
                    cursor.execute("DROP TABLE IF EXISTS rsi_cache_coins")
                    cursor.execute("DROP TABLE IF EXISTS rsi_cache")
                    
                    # Создаем новые таблицы с нормализованной структурой
                    cursor.execute("""
                        CREATE TABLE rsi_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            total_coins INTEGER DEFAULT 0,
                            successful_coins INTEGER DEFAULT 0,
                            failed_coins INTEGER DEFAULT 0,
                            extra_stats_json TEXT,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE TABLE rsi_cache_coins (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cache_id INTEGER NOT NULL,
                            symbol TEXT NOT NULL,
                            rsi6h REAL,
                            trend6h TEXT,
                            rsi_zone TEXT,
                            signal TEXT,
                            price REAL,
                            change24h REAL,
                            last_update TEXT,
                            blocked_by_scope INTEGER DEFAULT 0,
                            has_existing_position INTEGER DEFAULT 0,
                            is_mature INTEGER DEFAULT 1,
                            blocked_by_exit_scam INTEGER DEFAULT 0,
                            blocked_by_rsi_time INTEGER DEFAULT 0,
                            trading_status TEXT,
                            is_delisting INTEGER DEFAULT 0,
                            trend_analysis_json TEXT,
                            enhanced_rsi_json TEXT,
                            time_filter_info_json TEXT,
                            exit_scam_info_json TEXT,
                            extra_coin_data_json TEXT,
                            FOREIGN KEY (cache_id) REFERENCES rsi_cache(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Создаем индексы
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_timestamp ON rsi_cache(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_created ON rsi_cache(created_at)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_cache_id ON rsi_cache_coins(cache_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_symbol ON rsi_cache_coins(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_rsi6h ON rsi_cache_coins(rsi6h)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_signal ON rsi_cache_coins(signal)")
                    
                    # Мигрируем данные
                    migrated_caches = 0
                    migrated_coins = 0
                    
                    for old_row in old_rows:
                        try:
                            old_id = old_row[0]
                            timestamp = old_row[1]
                            coins_data_json = old_row[2]
                            stats_json = old_row[3]
                            created_at = old_row[4]
                            
                            # Парсим JSON
                            coins_data = json.loads(coins_data_json) if coins_data_json else {}
                            stats = json.loads(stats_json) if stats_json else {}
                            
                            # Извлекаем статистику
                            total_coins = stats.get('total_coins', len(coins_data))
                            successful_coins = stats.get('successful_coins', 0)
                            failed_coins = stats.get('failed_coins', 0)
                            
                            # Собираем остальные поля stats в extra_stats_json
                            extra_stats = {}
                            known_stats_fields = {'total_coins', 'successful_coins', 'failed_coins'}
                            for key, value in stats.items():
                                if key not in known_stats_fields:
                                    extra_stats[key] = value
                            
                            extra_stats_json = json.dumps(extra_stats) if extra_stats else None
                            
                            # Вставляем метаданные кэша
                            cursor.execute("""
                                INSERT INTO rsi_cache (
                                    timestamp, total_coins, successful_coins, failed_coins,
                                    extra_stats_json, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                            """, (timestamp, total_coins, successful_coins, failed_coins, extra_stats_json, created_at))
                            
                            cache_id = cursor.lastrowid
                            
                            # Мигрируем данные монет
                            for symbol, coin_data in coins_data.items():
                                try:
                                    # Извлекаем основные поля
                                    rsi6h = coin_data.get('rsi6h')
                                    trend6h = coin_data.get('trend6h')
                                    rsi_zone = coin_data.get('rsi_zone')
                                    signal = coin_data.get('signal')
                                    price = coin_data.get('price')
                                    change24h = coin_data.get('change24h') or coin_data.get('change_24h')
                                    last_update = coin_data.get('last_update')
                                    blocked_by_scope = 1 if coin_data.get('blocked_by_scope', False) else 0
                                    has_existing_position = 1 if coin_data.get('has_existing_position', False) else 0
                                    is_mature = 1 if coin_data.get('is_mature', True) else 0
                                    blocked_by_exit_scam = 1 if coin_data.get('blocked_by_exit_scam', False) else 0
                                    blocked_by_rsi_time = 1 if coin_data.get('blocked_by_rsi_time', False) else 0
                                    trading_status = coin_data.get('trading_status')
                                    is_delisting = 1 if coin_data.get('is_delisting', False) else 0
                                    
                                    # Сохраняем сложные структуры в JSON
                                    trend_analysis_json = json.dumps(coin_data.get('trend_analysis')) if coin_data.get('trend_analysis') else None
                                    enhanced_rsi_json = json.dumps(coin_data.get('enhanced_rsi')) if coin_data.get('enhanced_rsi') else None
                                    time_filter_info_json = json.dumps(coin_data.get('time_filter_info')) if coin_data.get('time_filter_info') else None
                                    exit_scam_info_json = json.dumps(coin_data.get('exit_scam_info')) if coin_data.get('exit_scam_info') else None
                                    
                                    # Собираем остальные поля в extra_coin_data_json
                                    extra_coin_data = {}
                                    known_coin_fields = {
                                        'symbol', 'rsi6h', 'trend6h', 'rsi_zone', 'signal', 'price',
                                        'change24h', 'change_24h', 'last_update', 'blocked_by_scope',
                                        'has_existing_position', 'is_mature', 'blocked_by_exit_scam',
                                        'blocked_by_rsi_time', 'trading_status', 'is_delisting',
                                        'trend_analysis', 'enhanced_rsi', 'time_filter_info', 'exit_scam_info'
                                    }
                                    
                                    for key, value in coin_data.items():
                                        if key not in known_coin_fields:
                                            extra_coin_data[key] = value
                                    
                                    extra_coin_data_json = json.dumps(extra_coin_data) if extra_coin_data else None
                                    
                                    # Вставляем монету
                                    cursor.execute("""
                                        INSERT INTO rsi_cache_coins (
                                            cache_id, symbol, rsi6h, trend6h, rsi_zone, signal,
                                            price, change24h, last_update, blocked_by_scope,
                                            has_existing_position, is_mature, blocked_by_exit_scam,
                                            blocked_by_rsi_time, trading_status, is_delisting,
                                            trend_analysis_json, enhanced_rsi_json, time_filter_info_json,
                                            exit_scam_info_json, extra_coin_data_json
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        cache_id, symbol, rsi6h, trend6h, rsi_zone, signal,
                                        price, change24h, last_update, blocked_by_scope,
                                        has_existing_position, is_mature, blocked_by_exit_scam,
                                        blocked_by_rsi_time, trading_status, is_delisting,
                                        trend_analysis_json, enhanced_rsi_json, time_filter_info_json,
                                        exit_scam_info_json, extra_coin_data_json
                                    ))
                                    migrated_coins += 1
                                except Exception as e:
                                    logger.warning(f"⚠️ Ошибка миграции монеты {symbol}: {e}")
                                    continue
                            
                            migrated_caches += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции кэша: {e}")
                            continue
                    
                    logger.info(f"✅ Миграция rsi_cache завершена: {migrated_caches} кэшей, {migrated_coins} монет мигрировано из JSON в нормализованные таблицы")
                else:
                    # Таблица пуста, просто пересоздаем с новой структурой
                    cursor.execute("DROP TABLE IF EXISTS rsi_cache_coins")
                    cursor.execute("DROP TABLE IF EXISTS rsi_cache")
                    cursor.execute("""
                        CREATE TABLE rsi_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            total_coins INTEGER DEFAULT 0,
                            successful_coins INTEGER DEFAULT 0,
                            failed_coins INTEGER DEFAULT 0,
                            extra_stats_json TEXT,
                            created_at TEXT NOT NULL
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE rsi_cache_coins (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cache_id INTEGER NOT NULL,
                            symbol TEXT NOT NULL,
                            rsi6h REAL,
                            trend6h TEXT,
                            rsi_zone TEXT,
                            signal TEXT,
                            price REAL,
                            change24h REAL,
                            last_update TEXT,
                            blocked_by_scope INTEGER DEFAULT 0,
                            has_existing_position INTEGER DEFAULT 0,
                            is_mature INTEGER DEFAULT 1,
                            blocked_by_exit_scam INTEGER DEFAULT 0,
                            blocked_by_rsi_time INTEGER DEFAULT 0,
                            trading_status TEXT,
                            is_delisting INTEGER DEFAULT 0,
                            trend_analysis_json TEXT,
                            enhanced_rsi_json TEXT,
                            time_filter_info_json TEXT,
                            exit_scam_info_json TEXT,
                            extra_coin_data_json TEXT,
                            FOREIGN KEY (cache_id) REFERENCES rsi_cache(id) ON DELETE CASCADE
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_timestamp ON rsi_cache(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_created ON rsi_cache(created_at)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_cache_id ON rsi_cache_coins(cache_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_symbol ON rsi_cache_coins(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_rsi6h ON rsi_cache_coins(rsi6h)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rsi_cache_coins_signal ON rsi_cache_coins(signal)")
                    logger.info("✅ Таблицы rsi_cache пересозданы с нормализованной структурой")
                    
            except sqlite3.OperationalError:
                # Таблица не существует или уже новая структура - ничего не делаем
                pass
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции rsi_cache: {e}")
            
            # ==================== МИГРАЦИЯ ДАННЫХ: Перенос сделок из других БД ====================
            try:
                # Проверяем, выполнена ли миграция данных из других БД
                cursor.execute("""
                    SELECT value FROM db_metadata 
                    WHERE key = 'trades_migration_from_other_dbs'
                """)
                migration_result = cursor.fetchone()
                
                if not migration_result or migration_result[0] != '1':
                    logger.info("📦 Начинаю миграцию сделок из других баз данных...")
                    
                    project_root = _get_project_root()
                    ai_db_path = project_root / 'data' / 'ai_data.db'
                    app_db_path = project_root / 'data' / 'app_data.db'
                    
                    # Пытаемся использовать SQL-скрипт миграции
                    migration_sql_path = project_root / 'migrations' / '001_migrate_trades_from_other_dbs.sql'
                    
                    if load_sql_file and migration_sql_path.exists():
                        try:
                            # Загружаем SQL-скрипт
                            sql_script = load_sql_file(str(migration_sql_path))
                            
                            # Заменяем плейсхолдеры на реальные пути (экранируем для SQL)
                            ai_db_path_str = str(ai_db_path).replace("'", "''").replace("\\", "/")
                            app_db_path_str = str(app_db_path).replace("'", "''").replace("\\", "/")
                            
                            sql_script = sql_script.replace('{AI_DB_PATH}', ai_db_path_str)
                            sql_script = sql_script.replace('{APP_DB_PATH}', app_db_path_str)
                            
                            # Проверяем наличие таблиц перед выполнением
                            should_migrate_ai = ai_db_path.exists()
                            should_migrate_app = app_db_path.exists()
                            
                            if should_migrate_ai:
                                # Проверяем наличие таблиц в ai_data.db
                                try:
                                    ai_conn = sqlite3.connect(str(ai_db_path))
                                    ai_cursor = ai_conn.cursor()
                                    ai_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('bot_trades', 'exchange_trades')")
                                    ai_tables = [row[0] for row in ai_cursor.fetchall()]
                                    ai_conn.close()
                                    
                                    if not ai_tables:
                                        should_migrate_ai = False
                                        logger.info("   ℹ️ Таблицы bot_trades и exchange_trades не найдены в ai_data.db")
                                except Exception as e:
                                    logger.warning(f"⚠️ Не удалось проверить ai_data.db: {e}")
                                    should_migrate_ai = False
                            
                            if should_migrate_app:
                                # Проверяем наличие таблицы closed_pnl в app_data.db
                                try:
                                    app_conn = sqlite3.connect(str(app_db_path))
                                    app_cursor = app_conn.cursor()
                                    app_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='closed_pnl'")
                                    if not app_cursor.fetchone():
                                        should_migrate_app = False
                                        logger.info("   ℹ️ Таблица closed_pnl не найдена в app_data.db")
                                    
                                    # Проверяем наличие необходимых полей
                                    if should_migrate_app:
                                        app_cursor.execute("PRAGMA table_info(closed_pnl)")
                                        columns = [row[1] for row in app_cursor.fetchall()]
                                        if 'symbol' not in columns or 'pnl' not in columns:
                                            should_migrate_app = False
                                            logger.info("   ℹ️ Таблица closed_pnl не содержит необходимых полей")
                                    
                                    app_conn.close()
                                except Exception as e:
                                    logger.warning(f"⚠️ Не удалось проверить app_data.db: {e}")
                                    should_migrate_app = False
                            
                            # Удаляем части скрипта для несуществующих БД
                            if not should_migrate_ai:
                                # Удаляем секцию миграции из ai_data.db
                                import re
                                sql_script = re.sub(r'-- =+.*?ai_data\.db.*?DETACH DATABASE ai_db;', '', sql_script, flags=re.DOTALL)
                            
                            if not should_migrate_app:
                                # Удаляем секцию миграции из app_data.db
                                import re
                                sql_script = re.sub(r'-- =+.*?app_data\.db.*?DETACH DATABASE app_db;', '', sql_script, flags=re.DOTALL)
                            
                            # Выполняем SQL-скрипт через database_utils
                            if should_migrate_ai or should_migrate_app:
                                db_path_str = str(self.db_path)
                                success, error, count = execute_sql_string(db_path_str, sql_script, stop_on_error=False)
                                
                                if success:
                                    # Подсчитываем количество мигрированных записей
                                    cursor.execute("SELECT COUNT(*) FROM bot_trades_history")
                                    total_count = cursor.fetchone()[0]
                                    
                                    # Устанавливаем флаг миграции
                                    now = datetime.now().isoformat()
                                    cursor.execute("""
                                        INSERT OR REPLACE INTO db_metadata (key, value, updated_at, created_at)
                                        VALUES ('trades_migration_from_other_dbs', '1', ?, 
                                            COALESCE((SELECT created_at FROM db_metadata WHERE key = 'trades_migration_from_other_dbs'), ?))
                                    """, (now, now))
                                    
                                    logger.info(f"✅ Миграция завершена через SQL-скрипт: выполнено {count} запросов, всего записей в bot_trades_history: {total_count}")
                                    conn.commit()
                                    return
                                else:
                                    logger.warning(f"⚠️ Ошибка выполнения SQL-скрипта миграции: {error}")
                                    # Продолжаем с встроенным кодом как fallback
                                    raise Exception(f"SQL-скрипт не выполнен: {error}")
                            else:
                                # Нет данных для миграции, просто устанавливаем флаг
                                now = datetime.now().isoformat()
                                cursor.execute("""
                                    INSERT OR REPLACE INTO db_metadata (key, value, updated_at, created_at)
                                    VALUES ('trades_migration_from_other_dbs', '1', ?, ?)
                                """, (now, now))
                                logger.info("✅ Миграция проверена: данных для миграции не найдено")
                                conn.commit()
                                return
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Не удалось использовать SQL-скрипт миграции, используем встроенный код: {e}")
                            # Продолжаем с встроенным кодом как fallback
                            # Не выбрасываем исключение, чтобы выполнить встроенный код
                    else:
                        # SQL-скрипт не найден или утилиты недоступны, используем встроенный код
                        if not migration_sql_path.exists():
                            logger.debug("ℹ️ SQL-скрипт миграции не найден, используем встроенный код")
                        else:
                            logger.debug("ℹ️ Утилиты database_utils недоступны, используем встроенный код")
                        raise Exception("Используем встроенный код миграции")
                    
                    # Если дошли сюда - миграция выполнена через SQL-скрипт
                    conn.commit()
                    return
                    
                    # ========== FALLBACK: Встроенный код миграции (если SQL-скрипт недоступен) ==========
                    migrated_count = 0
                    
                    # Миграция из ai_data.db -> bot_trades
                    if ai_db_path.exists():
                        try:
                            # Подключаем ai_data.db (экранируем путь для SQL)
                            ai_db_path_str = str(ai_db_path).replace("'", "''").replace("\\", "/")
                            cursor.execute(f"ATTACH DATABASE '{ai_db_path_str}' AS ai_db")
                            
                            # Проверяем наличие таблицы bot_trades
                            cursor.execute("""
                                SELECT name FROM ai_db.sqlite_master 
                                WHERE type='table' AND name='bot_trades'
                            """)
                            if cursor.fetchone():
                                # Мигрируем bot_trades
                                cursor.execute("""
                                    INSERT OR IGNORE INTO bot_trades_history (
                                        bot_id, symbol, direction, entry_price, exit_price,
                                        entry_time, exit_time, entry_timestamp, exit_timestamp,
                                        position_size_usdt, position_size_coins, pnl, roi,
                                        status, close_reason, decision_source, ai_decision_id,
                                        ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
                                        entry_volatility, entry_volume_ratio, is_successful,
                                        is_simulated, source, order_id, extra_data_json,
                                        created_at, updated_at
                                    )
                                    SELECT 
                                        COALESCE(bot_id, 'unknown') as bot_id,
                                        symbol,
                                        COALESCE(direction, 'LONG') as direction,
                                        entry_price,
                                        exit_price,
                                        entry_time,
                                        exit_time,
                                        CASE 
                                            WHEN entry_time IS NOT NULL 
                                            THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
                                            ELSE NULL
                                        END as entry_timestamp,
                                        CASE 
                                            WHEN exit_time IS NOT NULL 
                                            THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
                                            ELSE NULL
                                        END as exit_timestamp,
                                        position_size as position_size_usdt,
                                        position_size_coins,
                                        pnl,
                                        roi,
                                        COALESCE(status, 'CLOSED') as status,
                                        close_reason,
                                        'AI_BOT_TRADE' as decision_source,
                                        ai_decision_id,
                                        ai_confidence,
                                        entry_rsi,
                                        exit_rsi,
                                        entry_trend,
                                        exit_trend,
                                        entry_volatility,
                                        entry_volume_ratio,
                                        CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
                                        0 as is_simulated,
                                        'ai_bot' as source,
                                        order_id,
                                        json_object(
                                            'rsi_params', rsi_params,
                                            'risk_params', risk_params,
                                            'config_params', config_params,
                                            'filters_params', filters_params,
                                            'entry_conditions', entry_conditions,
                                            'exit_conditions', exit_conditions,
                                            'restrictions', restrictions,
                                            'extra_config', extra_config_json
                                        ) as extra_data_json,
                                        COALESCE(created_at, datetime('now')) as created_at,
                                        COALESCE(updated_at, datetime('now')) as updated_at
                                    FROM ai_db.bot_trades
                                    WHERE status = 'CLOSED' AND pnl IS NOT NULL
                                """)
                                count1 = cursor.rowcount
                                migrated_count += count1
                                logger.info(f"   ✅ Мигрировано {count1} сделок из ai_data.db -> bot_trades")
                            
                            # Мигрируем exchange_trades
                            cursor.execute("""
                                SELECT name FROM ai_db.sqlite_master 
                                WHERE type='table' AND name='exchange_trades'
                            """)
                            if cursor.fetchone():
                                cursor.execute("""
                                    INSERT OR IGNORE INTO bot_trades_history (
                                        bot_id, symbol, direction, entry_price, exit_price,
                                        entry_time, exit_time, entry_timestamp, exit_timestamp,
                                        position_size_usdt, position_size_coins, pnl, roi,
                                        status, close_reason, decision_source, ai_decision_id,
                                        ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
                                        entry_volatility, entry_volume_ratio, is_successful,
                                        is_simulated, source, order_id, extra_data_json,
                                        created_at, updated_at
                                    )
                                    SELECT 
                                        COALESCE(bot_id, 'exchange') as bot_id,
                                        symbol,
                                        COALESCE(direction, 'LONG') as direction,
                                        entry_price,
                                        exit_price,
                                        entry_time,
                                        exit_time,
                                        CASE 
                                            WHEN entry_time IS NOT NULL 
                                            THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
                                            ELSE NULL
                                        END as entry_timestamp,
                                        CASE 
                                            WHEN exit_time IS NOT NULL 
                                            THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
                                            ELSE NULL
                                        END as exit_timestamp,
                                        position_size as position_size_usdt,
                                        position_size_coins,
                                        pnl,
                                        roi,
                                        COALESCE(status, 'CLOSED') as status,
                                        close_reason,
                                        'EXCHANGE' as decision_source,
                                        ai_decision_id,
                                        ai_confidence,
                                        entry_rsi,
                                        exit_rsi,
                                        entry_trend,
                                        exit_trend,
                                        entry_volatility,
                                        entry_volume_ratio,
                                        CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
                                        CASE WHEN is_real = 0 OR is_real IS NULL THEN 1 ELSE 0 END as is_simulated,
                                        'exchange' as source,
                                        order_id,
                                        json_object(
                                            'is_real', is_real,
                                            'exchange', exchange,
                                            'extra_data', extra_data_json
                                        ) as extra_data_json,
                                        COALESCE(created_at, datetime('now')) as created_at,
                                        COALESCE(updated_at, datetime('now')) as updated_at
                                    FROM ai_db.exchange_trades
                                    WHERE status = 'CLOSED' AND pnl IS NOT NULL
                                      AND (is_real = 1 OR is_real IS NULL)
                                """)
                                count2 = cursor.rowcount
                                migrated_count += count2
                                logger.info(f"   ✅ Мигрировано {count2} сделок из ai_data.db -> exchange_trades")
                            
                            # Отключаем ai_db
                            cursor.execute("DETACH DATABASE ai_db")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции из ai_data.db: {e}")
                            try:
                                cursor.execute("DETACH DATABASE ai_db")
                            except:
                                pass
                    
                    # Миграция из app_data.db -> closed_pnl
                    if app_db_path.exists():
                        try:
                            # Подключаем app_data.db (экранируем путь для SQL)
                            app_db_path_str = str(app_db_path).replace("'", "''").replace("\\", "/")
                            cursor.execute(f"ATTACH DATABASE '{app_db_path_str}' AS app_db")
                            
                            # Проверяем наличие таблицы closed_pnl
                            cursor.execute("""
                                SELECT name FROM app_db.sqlite_master 
                                WHERE type='table' AND name='closed_pnl'
                            """)
                            if cursor.fetchone():
                                # Проверяем структуру таблицы closed_pnl
                                cursor.execute("PRAGMA table_info(app_db.closed_pnl)")
                                columns = [row[1] for row in cursor.fetchall()]
                                
                                # Мигрируем closed_pnl (если есть необходимые поля)
                                if 'symbol' in columns and 'pnl' in columns:
                                    cursor.execute("""
                                        INSERT OR IGNORE INTO bot_trades_history (
                                            bot_id, symbol, direction, entry_price, exit_price,
                                            entry_time, exit_time, entry_timestamp, exit_timestamp,
                                            position_size_usdt, position_size_coins, pnl, roi,
                                            status, close_reason, decision_source, ai_decision_id,
                                            ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
                                            entry_volatility, entry_volume_ratio, is_successful,
                                            is_simulated, source, order_id, extra_data_json,
                                            created_at, updated_at
                                        )
                                        SELECT 
                                            COALESCE(bot_id, 'app') as bot_id,
                                            symbol,
                                            COALESCE(direction, 'LONG') as direction,
                                            entry_price,
                                            exit_price,
                                            entry_time,
                                            exit_time,
                                            CASE 
                                                WHEN entry_time IS NOT NULL 
                                                THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
                                                ELSE NULL
                                            END as entry_timestamp,
                                            CASE 
                                                WHEN exit_time IS NOT NULL 
                                                THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
                                                ELSE NULL
                                            END as exit_timestamp,
                                            position_size as position_size_usdt,
                                            position_size_coins,
                                            pnl,
                                            roi,
                                            'CLOSED' as status,
                                            close_reason,
                                            'APP_CLOSED_PNL' as decision_source,
                                            NULL as ai_decision_id,
                                            NULL as ai_confidence,
                                            NULL as entry_rsi,
                                            NULL as exit_rsi,
                                            NULL as entry_trend,
                                            NULL as exit_trend,
                                            NULL as entry_volatility,
                                            NULL as entry_volume_ratio,
                                            CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
                                            0 as is_simulated,
                                            'app_closed_pnl' as source,
                                            order_id,
                                            COALESCE(extra_data_json, '{}') as extra_data_json,
                                            COALESCE(created_at, datetime('now')) as created_at,
                                            COALESCE(updated_at, datetime('now')) as updated_at
                                        FROM app_db.closed_pnl
                                        WHERE pnl IS NOT NULL
                                    """)
                                    count3 = cursor.rowcount
                                    migrated_count += count3
                                    logger.info(f"   ✅ Мигрировано {count3} сделок из app_data.db -> closed_pnl")
                            
                            # Отключаем app_db
                            cursor.execute("DETACH DATABASE app_db")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции из app_data.db: {e}")
                            try:
                                cursor.execute("DETACH DATABASE app_db")
                            except:
                                pass
                    
                    if migrated_count > 0:
                        # Устанавливаем флаг, что миграция выполнена
                        now = datetime.now().isoformat()
                        cursor.execute("""
                            INSERT OR REPLACE INTO db_metadata (key, value, updated_at, created_at)
                            VALUES ('trades_migration_from_other_dbs', '1', ?, 
                                COALESCE((SELECT created_at FROM db_metadata WHERE key = 'trades_migration_from_other_dbs'), ?))
                        """, (now, now))
                        logger.info(f"✅ Миграция завершена: всего мигрировано {migrated_count} сделок")
                    else:
                        # Устанавливаем флаг, что миграция проверена (но данных не было)
                        now = datetime.now().isoformat()
                        cursor.execute("""
                            INSERT OR REPLACE INTO db_metadata (key, value, updated_at, created_at)
                            VALUES ('trades_migration_from_other_dbs', '1', ?, ?)
                        """, (now, now))
                        logger.info("✅ Миграция проверена: данных для миграции не найдено")
                        
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции данных из других БД: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            conn.commit()
        except Exception as e:
            logger.debug(f"⚠️ Ошибка миграции схемы: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Не прерываем выполнение - миграция схемы не критична
    
    # ==================== МЕТОДЫ ДЛЯ СОСТОЯНИЯ БОТОВ ====================
    
    def save_bots_state(self, bots_data: Dict, auto_bot_config: Dict) -> bool:
        """
        Сохраняет состояние ботов в нормализованные таблицы
        
        Args:
            bots_data: Словарь с данными ботов {symbol: bot_dict}
            auto_bot_config: Конфигурация автобота
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Сохраняем каждого бота в нормализованную таблицу
                    for symbol, bot_data in bots_data.items():
                        try:
                            # Извлекаем все поля бота
                            extra_data = {}
                            
                            # Основные поля
                            status = bot_data.get('status', 'idle')
                            auto_managed = 1 if bot_data.get('auto_managed', False) else 0
                            volume_mode = bot_data.get('volume_mode', 'usdt')
                            volume_value = float(bot_data.get('volume_value', 0.0)) if bot_data.get('volume_value') not in (None, '') else None
                            
                            # Позиция
                            entry_price = float(bot_data.get('entry_price', 0.0)) if bot_data.get('entry_price') not in (None, '') else None
                            entry_time = bot_data.get('entry_time') or bot_data.get('position_start_time')
                            entry_timestamp = bot_data.get('entry_timestamp')
                            position_side = bot_data.get('position_side')
                            position_size = float(bot_data.get('position_size', 0.0)) if bot_data.get('position_size') not in (None, '') else None
                            position_size_coins = float(bot_data.get('position_size_coins', 0.0)) if bot_data.get('position_size_coins') not in (None, '') else None
                            position_start_time = bot_data.get('position_start_time')
                            
                            # PnL
                            unrealized_pnl = float(bot_data.get('unrealized_pnl', 0.0)) if bot_data.get('unrealized_pnl') not in (None, '') else 0.0
                            unrealized_pnl_usdt = float(bot_data.get('unrealized_pnl_usdt', 0.0)) if bot_data.get('unrealized_pnl_usdt') not in (None, '') else 0.0
                            realized_pnl = float(bot_data.get('realized_pnl', 0.0)) if bot_data.get('realized_pnl') not in (None, '') else 0.0
                            
                            # Другие поля
                            leverage = float(bot_data.get('leverage', 1.0)) if bot_data.get('leverage') not in (None, '') else 1.0
                            margin_usdt = float(bot_data.get('margin_usdt', 0.0)) if bot_data.get('margin_usdt') not in (None, '') else None
                            max_profit_achieved = float(bot_data.get('max_profit_achieved', 0.0)) if bot_data.get('max_profit_achieved') not in (None, '') else 0.0
                            
                            # Trailing stop
                            trailing_stop_price = float(bot_data.get('trailing_stop_price', 0.0)) if bot_data.get('trailing_stop_price') not in (None, '') else None
                            trailing_activation_threshold = float(bot_data.get('trailing_activation_threshold', 0.0)) if bot_data.get('trailing_activation_threshold') not in (None, '') else None
                            trailing_activation_profit = float(bot_data.get('trailing_activation_profit', 0.0)) if bot_data.get('trailing_activation_profit') not in (None, '') else 0.0
                            trailing_locked_profit = float(bot_data.get('trailing_locked_profit', 0.0)) if bot_data.get('trailing_locked_profit') not in (None, '') else 0.0
                            trailing_active = 1 if bot_data.get('trailing_active', False) else 0
                            trailing_max_profit_usdt = float(bot_data.get('trailing_max_profit_usdt', 0.0)) if bot_data.get('trailing_max_profit_usdt') not in (None, '') else 0.0
                            trailing_step_usdt = float(bot_data.get('trailing_step_usdt', 0.0)) if bot_data.get('trailing_step_usdt') not in (None, '') else None
                            trailing_step_price = float(bot_data.get('trailing_step_price', 0.0)) if bot_data.get('trailing_step_price') not in (None, '') else None
                            trailing_steps = int(bot_data.get('trailing_steps', 0)) if bot_data.get('trailing_steps') not in (None, '') else 0
                            trailing_reference_price = float(bot_data.get('trailing_reference_price', 0.0)) if bot_data.get('trailing_reference_price') not in (None, '') else None
                            trailing_last_update_ts = float(bot_data.get('trailing_last_update_ts', 0.0)) if bot_data.get('trailing_last_update_ts') not in (None, '') else 0.0
                            trailing_take_profit_price = float(bot_data.get('trailing_take_profit_price', 0.0)) if bot_data.get('trailing_take_profit_price') not in (None, '') else None
                            
                            # Break even
                            break_even_activated = 1 if bot_data.get('break_even_activated', False) else 0
                            break_even_stop_price = float(bot_data.get('break_even_stop_price', 0.0)) if bot_data.get('break_even_stop_price') not in (None, '') else None
                            
                            # Другие
                            order_id = bot_data.get('order_id')
                            current_price = float(bot_data.get('current_price', 0.0)) if bot_data.get('current_price') not in (None, '') else None
                            last_price = float(bot_data.get('last_price', 0.0)) if bot_data.get('last_price') not in (None, '') else None
                            last_rsi = float(bot_data.get('last_rsi', 0.0)) if bot_data.get('last_rsi') not in (None, '') else None
                            last_trend = bot_data.get('last_trend')
                            last_signal_time = bot_data.get('last_signal_time')
                            last_bar_timestamp = float(bot_data.get('last_bar_timestamp', 0.0)) if bot_data.get('last_bar_timestamp') not in (None, '') else None
                            entry_trend = bot_data.get('entry_trend')
                            opened_by_autobot = 1 if bot_data.get('opened_by_autobot', False) else 0
                            bot_id = bot_data.get('id')
                            
                            # Собираем все остальные поля в extra_data_json
                            known_fields = {
                                'symbol', 'status', 'auto_managed', 'volume_mode', 'volume_value',
                                'entry_price', 'entry_time', 'entry_timestamp', 'position_side',
                                'position_size', 'position_size_coins', 'position_start_time',
                                'unrealized_pnl', 'unrealized_pnl_usdt', 'realized_pnl', 'leverage',
                                'margin_usdt', 'max_profit_achieved', 'trailing_stop_price',
                                'trailing_activation_threshold', 'trailing_activation_profit',
                                'trailing_locked_profit', 'trailing_active', 'trailing_max_profit_usdt',
                                'trailing_step_usdt', 'trailing_step_price', 'trailing_steps',
                                'trailing_reference_price', 'trailing_last_update_ts', 'trailing_take_profit_price',
                                'break_even_activated', 'break_even_stop_price', 'order_id',
                                'current_price', 'last_price', 'last_rsi', 'last_trend',
                                'last_signal_time', 'last_bar_timestamp', 'entry_trend',
                                'opened_by_autobot', 'id', 'position', 'rsi_data', 'scaling_enabled',
                                'scaling_levels', 'scaling_current_level', 'scaling_group_id', 'created_at'
                            }
                            
                            for key, value in bot_data.items():
                                if key not in known_fields:
                                    extra_data[key] = value
                            
                            # Сохраняем сложные структуры в extra_data
                            if bot_data.get('position'):
                                extra_data['position'] = bot_data['position']
                            if bot_data.get('rsi_data'):
                                extra_data['rsi_data'] = bot_data['rsi_data']
                            
                            extra_data_json = json.dumps(extra_data) if extra_data else None
                            
                            # Получаем created_at из существующей записи или используем текущее время
                            cursor.execute("SELECT created_at FROM bots WHERE symbol = ?", (symbol,))
                            existing = cursor.fetchone()
                            final_created_at = existing[0] if existing else (bot_data.get('created_at') or now)
                            
                            # Вставляем или обновляем бота
                            cursor.execute("""
                                INSERT OR REPLACE INTO bots (
                                    symbol, status, auto_managed, volume_mode, volume_value,
                                    entry_price, entry_time, entry_timestamp, position_side,
                                    position_size, position_size_coins, position_start_time,
                                    unrealized_pnl, unrealized_pnl_usdt, realized_pnl, leverage,
                                    margin_usdt, max_profit_achieved, trailing_stop_price,
                                    trailing_activation_threshold, trailing_activation_profit,
                                    trailing_locked_profit, trailing_active, trailing_max_profit_usdt,
                                    trailing_step_usdt, trailing_step_price, trailing_steps,
                                    trailing_reference_price, trailing_last_update_ts, trailing_take_profit_price,
                                    break_even_activated, break_even_stop_price, order_id,
                                    current_price, last_price, last_rsi, last_trend,
                                    last_signal_time, last_bar_timestamp, entry_trend,
                                    opened_by_autobot, bot_id, extra_data_json,
                                    updated_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol, status, auto_managed, volume_mode, volume_value,
                                entry_price, entry_time, entry_timestamp, position_side,
                                position_size, position_size_coins, position_start_time,
                                unrealized_pnl, unrealized_pnl_usdt, realized_pnl, leverage,
                                margin_usdt, max_profit_achieved, trailing_stop_price,
                                trailing_activation_threshold, trailing_activation_profit,
                                trailing_locked_profit, trailing_active, trailing_max_profit_usdt,
                                trailing_step_usdt, trailing_step_price, trailing_steps,
                                trailing_reference_price, trailing_last_update_ts, trailing_take_profit_price,
                                break_even_activated, break_even_stop_price, order_id,
                                current_price, last_price, last_rsi, last_trend,
                                last_signal_time, last_bar_timestamp, entry_trend,
                                opened_by_autobot, bot_id, extra_data_json,
                                now, final_created_at
                            ))
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка сохранения бота {symbol}: {e}")
                            continue
                    
                    # Сохраняем auto_bot_config
                    for key, value in auto_bot_config.items():
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO auto_bot_config (key, value, updated_at, created_at)
                                VALUES (?, ?, ?, COALESCE((SELECT created_at FROM auto_bot_config WHERE key = ?), ?))
                            """, (key, json.dumps(value) if not isinstance(value, (str, int, float, bool)) else str(value), now, key, now))
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка сохранения auto_bot_config.{key}: {e}")
                    
                    conn.commit()
            
            logger.debug("💾 Состояние ботов сохранено в нормализованные таблицы БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения состояния ботов: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_bots_state(self) -> Dict:
        """
        Загружает состояние ботов из нормализованных таблиц
        
        Returns:
            Словарь с состоянием {bots: {symbol: bot_dict}, auto_bot_config: {...}}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Загружаем ботов из нормализованной таблицы
                cursor.execute("""
                    SELECT symbol, status, auto_managed, volume_mode, volume_value,
                           entry_price, entry_time, entry_timestamp, position_side,
                           position_size, position_size_coins, position_start_time,
                           unrealized_pnl, unrealized_pnl_usdt, realized_pnl, leverage,
                           margin_usdt, max_profit_achieved, trailing_stop_price,
                           trailing_activation_threshold, trailing_activation_profit,
                           trailing_locked_profit, trailing_active, trailing_max_profit_usdt,
                           trailing_step_usdt, trailing_step_price, trailing_steps,
                           trailing_reference_price, trailing_last_update_ts, trailing_take_profit_price,
                           break_even_activated, break_even_stop_price, order_id,
                           current_price, last_price, last_rsi, last_trend,
                           last_signal_time, last_bar_timestamp, entry_trend,
                           opened_by_autobot, bot_id, extra_data_json,
                           updated_at, created_at
                    FROM bots
                """)
                rows = cursor.fetchall()
                
                bots_data = {}
                for row in rows:
                    symbol = row[0]
                    bot_dict = {
                        'symbol': symbol,
                        'status': row[1],
                        'auto_managed': bool(row[2]),
                        'volume_mode': row[3],
                        'volume_value': row[4],
                        'entry_price': row[5],
                        'entry_time': row[6],
                        'entry_timestamp': row[7],
                        'position_side': row[8],
                        'position_size': row[9],
                        'position_size_coins': row[10],
                        'position_start_time': row[11],
                        'unrealized_pnl': row[12],
                        'unrealized_pnl_usdt': row[13],
                        'realized_pnl': row[14],
                        'leverage': row[15],
                        'margin_usdt': row[16],
                        'max_profit_achieved': row[17],
                        'trailing_stop_price': row[18],
                        'trailing_activation_threshold': row[19],
                        'trailing_activation_profit': row[20],
                        'trailing_locked_profit': row[21],
                        'trailing_active': bool(row[22]),
                        'trailing_max_profit_usdt': row[23],
                        'trailing_step_usdt': row[24],
                        'trailing_step_price': row[25],
                        'trailing_steps': row[26],
                        'trailing_reference_price': row[27],
                        'trailing_last_update_ts': row[28],
                        'trailing_take_profit_price': row[29],
                        'break_even_activated': bool(row[30]),
                        'break_even_stop_price': row[31],
                        'order_id': row[32],
                        'current_price': row[33],
                        'last_price': row[34],
                        'last_rsi': row[35],
                        'last_trend': row[36],
                        'last_signal_time': row[37],
                        'last_bar_timestamp': row[38],
                        'entry_trend': row[39],
                        'opened_by_autobot': bool(row[40]),
                        'id': row[41],
                        'created_at': row[43]
                    }
                    
                    # Загружаем extra_data_json если есть
                    if row[42]:
                        try:
                            extra_data = json.loads(row[42])
                            bot_dict.update(extra_data)
                        except:
                            pass
                    
                    bots_data[symbol] = bot_dict
                
                # Загружаем auto_bot_config
                cursor.execute("SELECT key, value FROM auto_bot_config")
                config_rows = cursor.fetchall()
                auto_bot_config = {}
                for config_row in config_rows:
                    key = config_row[0]
                    value = config_row[1]
                    try:
                        # Пытаемся распарсить как JSON, если не получается - оставляем как строку
                        auto_bot_config[key] = json.loads(value)
                    except:
                        auto_bot_config[key] = value
                
                return {
                    'bots': bots_data,
                    'auto_bot_config': auto_bot_config,
                    'version': '2.0'  # Новая версия с нормализованной структурой
                }
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки состояния ботов: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ РЕЕСТРА ПОЗИЦИЙ ====================
    
    def save_bot_positions_registry(self, registry: Dict) -> bool:
        """
        Сохраняет реестр позиций ботов
        
        Args:
            registry: Словарь {bot_id: {symbol: str, side: str, entry_price: float, quantity: float, opened_at: str, managed_by_bot: bool}}
                      ИЛИ {bot_id: position_dict} где position_dict содержит все поля позиции
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Удаляем старые записи
                    cursor.execute("DELETE FROM bot_positions_registry")
                    
                    # Вставляем новые записи в нормализованном формате
                    for bot_id, position_data in registry.items():
                        # Поддерживаем оба формата: прямой словарь или вложенный
                        if isinstance(position_data, dict):
                            # Извлекаем поля позиции
                            symbol = position_data.get('symbol', '')
                            side = position_data.get('side', 'LONG')
                            entry_price = float(position_data.get('entry_price', 0.0))
                            quantity = float(position_data.get('quantity', 0.0))
                            opened_at = position_data.get('opened_at', now)
                            managed_by_bot = 1 if position_data.get('managed_by_bot', True) else 0
                            created_at = position_data.get('created_at', now)
                            
                            cursor.execute("""
                                INSERT INTO bot_positions_registry 
                                (bot_id, symbol, side, entry_price, quantity, opened_at, managed_by_bot, updated_at, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                bot_id,
                                symbol,
                                side,
                                entry_price,
                                quantity,
                                opened_at,
                                managed_by_bot,
                                now,
                                created_at
                            ))
                    
                    conn.commit()
            
            logger.debug(f"💾 Реестр позиций сохранен в БД ({len(registry)} записей)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения реестра позиций: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_bot_positions_registry(self) -> Dict:
        """
        Загружает реестр позиций ботов
        
        Returns:
            Словарь {bot_id: {symbol: str, side: str, entry_price: float, quantity: float, opened_at: str, managed_by_bot: bool}}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT bot_id, symbol, side, entry_price, quantity, opened_at, managed_by_bot, updated_at, created_at
                    FROM bot_positions_registry
                """)
                rows = cursor.fetchall()
                
                registry = {}
                for row in rows:
                    bot_id = row['bot_id']
                    position_data = {
                        'symbol': row['symbol'],
                        'side': row['side'],
                        'entry_price': row['entry_price'],
                        'quantity': row['quantity'],
                        'opened_at': row['opened_at'],
                        'managed_by_bot': bool(row['managed_by_bot']),
                        'updated_at': row['updated_at'],
                        'created_at': row['created_at']
                    }
                    registry[bot_id] = position_data
                
                return registry
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки реестра позиций: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ RSI КЭША ====================
    
    def save_rsi_cache(self, coins_data: Dict, stats: Dict = None) -> bool:
        """
        Сохраняет RSI кэш в нормализованные таблицы
        
        Args:
            coins_data: Словарь {symbol: {rsi6h, trend6h, signal, price, ...}}
            stats: Статистика {total_coins, successful_coins, failed_coins, ...}
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            timestamp = now
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Извлекаем статистику
                    total_coins = stats.get('total_coins', len(coins_data)) if stats else len(coins_data)
                    successful_coins = stats.get('successful_coins', 0) if stats else 0
                    failed_coins = stats.get('failed_coins', 0) if stats else 0
                    
                    # Собираем остальные поля stats в extra_stats_json
                    extra_stats = {}
                    if stats:
                        known_stats_fields = {'total_coins', 'successful_coins', 'failed_coins'}
                        for key, value in stats.items():
                            if key not in known_stats_fields:
                                extra_stats[key] = value
                    
                    extra_stats_json = json.dumps(extra_stats) if extra_stats else None
                    
                    # Удаляем старые записи (оставляем только последний кэш)
                    cursor.execute("DELETE FROM rsi_cache_coins")
                    cursor.execute("DELETE FROM rsi_cache")
                    
                    # Вставляем метаданные кэша
                    cursor.execute("""
                        INSERT INTO rsi_cache (
                            timestamp, total_coins, successful_coins, failed_coins,
                            extra_stats_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (timestamp, total_coins, successful_coins, failed_coins, extra_stats_json, now))
                    
                    cache_id = cursor.lastrowid
                    
                    # Вставляем данные монет
                    for symbol, coin_data in coins_data.items():
                        try:
                            # Извлекаем основные поля
                            rsi6h = coin_data.get('rsi6h')
                            trend6h = coin_data.get('trend6h')
                            rsi_zone = coin_data.get('rsi_zone')
                            signal = coin_data.get('signal')
                            price = coin_data.get('price')
                            change24h = coin_data.get('change24h') or coin_data.get('change_24h')
                            last_update = coin_data.get('last_update')
                            blocked_by_scope = 1 if coin_data.get('blocked_by_scope', False) else 0
                            has_existing_position = 1 if coin_data.get('has_existing_position', False) else 0
                            is_mature = 1 if coin_data.get('is_mature', True) else 0
                            blocked_by_exit_scam = 1 if coin_data.get('blocked_by_exit_scam', False) else 0
                            blocked_by_rsi_time = 1 if coin_data.get('blocked_by_rsi_time', False) else 0
                            trading_status = coin_data.get('trading_status')
                            is_delisting = 1 if coin_data.get('is_delisting', False) else 0
                            
                            # Сохраняем сложные структуры в JSON
                            trend_analysis_json = json.dumps(coin_data.get('trend_analysis')) if coin_data.get('trend_analysis') else None
                            enhanced_rsi_json = json.dumps(coin_data.get('enhanced_rsi')) if coin_data.get('enhanced_rsi') else None
                            time_filter_info_json = json.dumps(coin_data.get('time_filter_info')) if coin_data.get('time_filter_info') else None
                            exit_scam_info_json = json.dumps(coin_data.get('exit_scam_info')) if coin_data.get('exit_scam_info') else None
                            
                            # Собираем остальные поля в extra_coin_data_json
                            extra_coin_data = {}
                            known_coin_fields = {
                                'symbol', 'rsi6h', 'trend6h', 'rsi_zone', 'signal', 'price',
                                'change24h', 'change_24h', 'last_update', 'blocked_by_scope',
                                'has_existing_position', 'is_mature', 'blocked_by_exit_scam',
                                'blocked_by_rsi_time', 'trading_status', 'is_delisting',
                                'trend_analysis', 'enhanced_rsi', 'time_filter_info', 'exit_scam_info'
                            }
                            
                            for key, value in coin_data.items():
                                if key not in known_coin_fields:
                                    extra_coin_data[key] = value
                            
                            extra_coin_data_json = json.dumps(extra_coin_data) if extra_coin_data else None
                            
                            # Вставляем монету
                            cursor.execute("""
                                INSERT INTO rsi_cache_coins (
                                    cache_id, symbol, rsi6h, trend6h, rsi_zone, signal,
                                    price, change24h, last_update, blocked_by_scope,
                                    has_existing_position, is_mature, blocked_by_exit_scam,
                                    blocked_by_rsi_time, trading_status, is_delisting,
                                    trend_analysis_json, enhanced_rsi_json, time_filter_info_json,
                                    exit_scam_info_json, extra_coin_data_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                                cache_id, symbol, rsi6h, trend6h, rsi_zone, signal,
                                price, change24h, last_update, blocked_by_scope,
                                has_existing_position, is_mature, blocked_by_exit_scam,
                                blocked_by_rsi_time, trading_status, is_delisting,
                                trend_analysis_json, enhanced_rsi_json, time_filter_info_json,
                                exit_scam_info_json, extra_coin_data_json
                            ))
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка сохранения монеты {symbol} в RSI кэш: {e}")
                            continue
                    
                    conn.commit()
            
            logger.debug("💾 RSI кэш сохранен в нормализованные таблицы БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения RSI кэша: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_rsi_cache(self, max_age_hours: float = 6.0) -> Optional[Dict]:
        """
        Загружает последний RSI кэш из нормализованных таблиц (если не старше max_age_hours)
        
        Args:
            max_age_hours: Максимальный возраст кэша в часах
        
        Returns:
            Словарь с данными кэша или None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, total_coins, successful_coins, failed_coins, extra_stats_json, created_at
                    FROM rsi_cache
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                cache_id = row[0]
                
                # Проверяем возраст кэша
                cache_time = datetime.fromisoformat(row[1])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    logger.debug(f"⚠️ RSI кэш устарел ({age_hours:.1f} часов)")
                    return None
                
                # Загружаем данные монет
                cursor.execute("""
                    SELECT symbol, rsi6h, trend6h, rsi_zone, signal, price, change24h,
                           last_update, blocked_by_scope, has_existing_position, is_mature,
                           blocked_by_exit_scam, blocked_by_rsi_time, trading_status, is_delisting,
                           trend_analysis_json, enhanced_rsi_json, time_filter_info_json,
                           exit_scam_info_json, extra_coin_data_json
                    FROM rsi_cache_coins
                    WHERE cache_id = ?
                """, (cache_id,))
                coin_rows = cursor.fetchall()
                
                coins_data = {}
                for coin_row in coin_rows:
                    symbol = coin_row[0]
                    coin_data = {
                        'symbol': symbol,
                        'rsi6h': coin_row[1],
                        'trend6h': coin_row[2],
                        'rsi_zone': coin_row[3],
                        'signal': coin_row[4],
                        'price': coin_row[5],
                        'change24h': coin_row[6],
                        'last_update': coin_row[7],
                        'blocked_by_scope': bool(coin_row[8]),
                        'has_existing_position': bool(coin_row[9]),
                        'is_mature': bool(coin_row[10]),
                        'blocked_by_exit_scam': bool(coin_row[11]),
                        'blocked_by_rsi_time': bool(coin_row[12]),
                        'trading_status': coin_row[13],
                        'is_delisting': bool(coin_row[14])
                    }
                    
                    # Удаляем None значения
                    coin_data = {k: v for k, v in coin_data.items() if v is not None}
                    
                    # Загружаем сложные структуры из JSON
                    if coin_row[15]:
                        try:
                            coin_data['trend_analysis'] = json.loads(coin_row[15])
                        except:
                            pass
                    if coin_row[16]:
                        try:
                            coin_data['enhanced_rsi'] = json.loads(coin_row[16])
                        except:
                            pass
                    if coin_row[17]:
                        try:
                            coin_data['time_filter_info'] = json.loads(coin_row[17])
                        except:
                            pass
                    if coin_row[18]:
                        try:
                            coin_data['exit_scam_info'] = json.loads(coin_row[18])
                        except:
                            pass
                    
                    # Загружаем extra_coin_data_json если есть
                    if coin_row[19]:
                        try:
                            extra_data = json.loads(coin_row[19])
                            coin_data.update(extra_data)
                        except:
                            pass
                    
                    coins_data[symbol] = coin_data
                
                # Собираем статистику
                stats = {
                    'total_coins': row[2],
                    'successful_coins': row[3],
                    'failed_coins': row[4]
                }
                
                # Загружаем extra_stats_json если есть
                if row[5]:
                    try:
                        extra_stats = json.loads(row[5])
                        stats.update(extra_stats)
                    except:
                        pass
                
                return {
                    'timestamp': row[1],
                    'coins': coins_data,
                    'stats': stats
                }
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки RSI кэша: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def clear_rsi_cache(self) -> bool:
        """Очищает RSI кэш"""
        try:
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM rsi_cache")
                    conn.commit()
            logger.info("✅ RSI кэш очищен в БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка очистки RSI кэша: {e}")
            return False
    
    # ==================== МЕТОДЫ ДЛЯ СОСТОЯНИЯ ПРОЦЕССОВ ====================
    
    def save_process_state(self, process_state: Dict) -> bool:
        """
        Сохраняет состояние процессов в нормализованные столбцы
        
        Args:
            process_state: Словарь {process_name: {active, last_update, ...}}
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Сохраняем каждый процесс отдельной строкой
                    for process_name, process_data in process_state.items():
                        try:
                            # Извлекаем поля процесса
                            active = 1 if process_data.get('active', False) else 0
                            initialized = 1 if process_data.get('initialized', False) else 0
                            last_update = process_data.get('last_update')
                            last_check = process_data.get('last_check')
                            last_save = process_data.get('last_save')
                            last_sync = process_data.get('last_sync')
                            update_count = process_data.get('update_count', 0)
                            check_count = process_data.get('check_count', 0)
                            save_count = process_data.get('save_count', 0)
                            connection_count = process_data.get('connection_count', 0)
                            signals_processed = process_data.get('signals_processed', 0)
                            bots_created = process_data.get('bots_created', 0)
                            last_error = process_data.get('last_error')
                            
                            # Собираем остальные поля в extra_process_data_json
                            extra_data = {}
                            known_fields = {
                                'active', 'initialized', 'last_update', 'last_check',
                                'last_save', 'last_sync', 'update_count', 'check_count',
                                'save_count', 'connection_count', 'signals_processed',
                                'bots_created', 'last_error'
                            }
                            
                            for key, value in process_data.items():
                                if key not in known_fields:
                                    extra_data[key] = value
                            
                            extra_process_data_json = json.dumps(extra_data) if extra_data else None
                            
                            # Получаем created_at из существующей записи или используем текущее время
                            cursor.execute("SELECT created_at FROM process_state WHERE process_name = ?", (process_name,))
                            existing = cursor.fetchone()
                            final_created_at = existing[0] if existing else now
                            
                            # Вставляем или обновляем процесс
                            cursor.execute("""
                                INSERT OR REPLACE INTO process_state (
                                    process_name, active, initialized, last_update,
                                    last_check, last_save, last_sync, update_count,
                                    check_count, save_count, connection_count,
                                    signals_processed, bots_created, last_error,
                                    extra_process_data_json, updated_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                process_name, active, initialized, last_update,
                                last_check, last_save, last_sync, update_count,
                                check_count, save_count, connection_count,
                                signals_processed, bots_created, last_error,
                                extra_process_data_json, now, final_created_at
                            ))
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка сохранения процесса {process_name}: {e}")
                            continue
                    
                    conn.commit()
            
            logger.debug("💾 Состояние процессов сохранено в нормализованные столбцы БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения состояния процессов: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_process_state(self) -> Dict:
        """
        Загружает состояние процессов из нормализованных столбцов
        
        Returns:
            Словарь {process_name: {active, last_update, ...}}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT process_name, active, initialized, last_update,
                           last_check, last_save, last_sync, update_count,
                           check_count, save_count, connection_count,
                           signals_processed, bots_created, last_error,
                           extra_process_data_json
                    FROM process_state
                """)
                rows = cursor.fetchall()
                
                process_state_dict = {}
                for row in rows:
                    process_name = row[0]
                    process_data = {
                        'active': bool(row[1]),
                        'initialized': bool(row[2]),
                        'last_update': row[3],
                        'last_check': row[4],
                        'last_save': row[5],
                        'last_sync': row[6],
                        'update_count': row[7],
                        'check_count': row[8],
                        'save_count': row[9],
                        'connection_count': row[10],
                        'signals_processed': row[11],
                        'bots_created': row[12],
                        'last_error': row[13]
                    }
                    
                    # Удаляем None значения
                    process_data = {k: v for k, v in process_data.items() if v is not None}
                    
                    # Загружаем extra_process_data_json если есть
                    if row[14]:
                        try:
                            extra_data = json.loads(row[14])
                            process_data.update(extra_data)
                        except:
                            pass
                    
                    process_state_dict[process_name] = process_data
                
                return process_state_dict
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки состояния процессов: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ ИНДИВИДУАЛЬНЫХ НАСТРОЕК ====================
    
    def save_individual_coin_settings(self, settings: Dict) -> bool:
        """
        Сохраняет индивидуальные настройки монет в нормализованные столбцы
        
        Args:
            settings: Словарь {symbol: settings_dict}
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Удаляем старые записи
                    cursor.execute("DELETE FROM individual_coin_settings")
                    
                    # Вставляем новые записи в нормализованном формате
                    for symbol, symbol_settings in settings.items():
                        # Извлекаем основные поля
                        extra_settings = {}
                        known_fields = {
                            'rsi_long_threshold', 'rsi_short_threshold',
                            'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                            'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                            'max_loss_percent', 'take_profit_percent',
                            'trailing_stop_activation', 'trailing_stop_distance',
                            'trailing_take_distance', 'trailing_update_interval',
                            'break_even_trigger', 'break_even_protection',
                            'max_position_hours', 'rsi_time_filter_enabled',
                            'rsi_time_filter_candles', 'rsi_time_filter_upper',
                            'rsi_time_filter_lower', 'avoid_down_trend'
                        }
                        
                        for key, value in symbol_settings.items():
                            if key not in known_fields:
                                extra_settings[key] = value
                        
                        extra_settings_json = json.dumps(extra_settings) if extra_settings else None
                        
                        # Получаем created_at из существующей записи или используем текущее время
                        cursor.execute("SELECT created_at FROM individual_coin_settings WHERE symbol = ?", (symbol,))
                        existing = cursor.fetchone()
                        created_at = existing[0] if existing else symbol_settings.get('created_at', now)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO individual_coin_settings (
                                symbol, rsi_long_threshold, rsi_short_threshold,
                                rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                                rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                                max_loss_percent, take_profit_percent,
                                trailing_stop_activation, trailing_stop_distance,
                                trailing_take_distance, trailing_update_interval,
                                break_even_trigger, break_even_protection,
                                max_position_hours, rsi_time_filter_enabled,
                                rsi_time_filter_candles, rsi_time_filter_upper,
                                rsi_time_filter_lower, avoid_down_trend,
                                extra_settings_json, updated_at, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT created_at FROM individual_coin_settings WHERE symbol = ?), ?), ?)
                        """, (
                            symbol,
                            symbol_settings.get('rsi_long_threshold'),
                            symbol_settings.get('rsi_short_threshold'),
                            symbol_settings.get('rsi_exit_long_with_trend'),
                            symbol_settings.get('rsi_exit_long_against_trend'),
                            symbol_settings.get('rsi_exit_short_with_trend'),
                            symbol_settings.get('rsi_exit_short_against_trend'),
                            symbol_settings.get('max_loss_percent'),
                            symbol_settings.get('take_profit_percent'),
                            symbol_settings.get('trailing_stop_activation'),
                            symbol_settings.get('trailing_stop_distance'),
                            symbol_settings.get('trailing_take_distance'),
                            symbol_settings.get('trailing_update_interval'),
                            symbol_settings.get('break_even_trigger'),
                            symbol_settings.get('break_even_protection'),
                            symbol_settings.get('max_position_hours'),
                            1 if symbol_settings.get('rsi_time_filter_enabled') else 0,
                            symbol_settings.get('rsi_time_filter_candles'),
                            symbol_settings.get('rsi_time_filter_upper'),
                            symbol_settings.get('rsi_time_filter_lower'),
                            1 if symbol_settings.get('avoid_down_trend') else 0,
                            extra_settings_json,
                            symbol,
                            created_at,
                            now
                        ))
                    
                    conn.commit()
            
            logger.debug(f"💾 Индивидуальные настройки сохранены в нормализованные столбцы БД ({len(settings)} монет)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения индивидуальных настроек: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_individual_coin_settings(self) -> Dict:
        """
        Загружает индивидуальные настройки монет из нормализованных столбцов
        
        Returns:
            Словарь {symbol: settings_dict}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, rsi_long_threshold, rsi_short_threshold,
                           rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                           rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                           max_loss_percent, take_profit_percent,
                           trailing_stop_activation, trailing_stop_distance,
                           trailing_take_distance, trailing_update_interval,
                           break_even_trigger, break_even_protection,
                           max_position_hours, rsi_time_filter_enabled,
                           rsi_time_filter_candles, rsi_time_filter_upper,
                           rsi_time_filter_lower, avoid_down_trend,
                           extra_settings_json
                    FROM individual_coin_settings
                """)
                rows = cursor.fetchall()
                
                settings = {}
                for row in rows:
                    symbol = row[0]
                    settings_dict = {
                        'rsi_long_threshold': row[1],
                        'rsi_short_threshold': row[2],
                        'rsi_exit_long_with_trend': row[3],
                        'rsi_exit_long_against_trend': row[4],
                        'rsi_exit_short_with_trend': row[5],
                        'rsi_exit_short_against_trend': row[6],
                        'max_loss_percent': row[7],
                        'take_profit_percent': row[8],
                        'trailing_stop_activation': row[9],
                        'trailing_stop_distance': row[10],
                        'trailing_take_distance': row[11],
                        'trailing_update_interval': row[12],
                        'break_even_trigger': row[13],
                        'break_even_protection': row[14],
                        'max_position_hours': row[15],
                        'rsi_time_filter_enabled': bool(row[16]),
                        'rsi_time_filter_candles': row[17],
                        'rsi_time_filter_upper': row[18],
                        'rsi_time_filter_lower': row[19],
                        'avoid_down_trend': bool(row[20])
                    }
                    
                    # Удаляем None значения
                    settings_dict = {k: v for k, v in settings_dict.items() if v is not None}
                    
                    # Загружаем extra_settings_json если есть
                    if row[21]:
                        try:
                            extra_settings = json.loads(row[21])
                            settings_dict.update(extra_settings)
                        except:
                            pass
                    
                    settings[symbol] = settings_dict
                
                return settings
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки индивидуальных настроек: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    def remove_all_individual_coin_settings(self) -> bool:
        """Удаляет все индивидуальные настройки"""
        try:
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM individual_coin_settings")
                    conn.commit()
            logger.info("✅ Все индивидуальные настройки удалены из БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка удаления индивидуальных настроек: {e}")
            return False
    
    # ==================== МЕТОДЫ ДЛЯ ЗРЕЛЫХ МОНЕТ ====================
    
    def save_mature_coins(self, mature_coins: Dict) -> bool:
        """
        Сохраняет зрелые монеты в нормализованные столбцы
        
        Args:
            mature_coins: Словарь {symbol: {timestamp: float, maturity_data: dict}}
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Удаляем старые записи
                    cursor.execute("DELETE FROM mature_coins")
                    
                    # Вставляем новые записи в нормализованном формате
                    for symbol, coin_data in mature_coins.items():
                        timestamp = coin_data.get('timestamp', 0.0)
                        maturity_data = coin_data.get('maturity_data', {})
                        
                        # Извлекаем основные поля
                        is_mature = 1 if maturity_data.get('is_mature', False) else 0
                        details = maturity_data.get('details', {})
                        candles_count = details.get('candles_count')
                        min_required = details.get('min_required')
                        config_min_rsi_low = details.get('config_min_rsi_low')
                        config_max_rsi_high = details.get('config_max_rsi_high')
                        
                        # Собираем остальные поля в extra_maturity_data_json
                        extra_data = {}
                        known_fields = {'is_mature', 'details'}
                        for key, value in maturity_data.items():
                            if key not in known_fields:
                                extra_data[key] = value
                        
                        # Также сохраняем неизвестные поля из details
                        known_details_fields = {'candles_count', 'min_required', 'config_min_rsi_low', 'config_max_rsi_high'}
                        for key, value in details.items():
                            if key not in known_details_fields:
                                if 'extra_details' not in extra_data:
                                    extra_data['extra_details'] = {}
                                extra_data['extra_details'][key] = value
                        
                        extra_maturity_data_json = json.dumps(extra_data) if extra_data else None
                        
                        # Получаем created_at из существующей записи или используем текущее время
                        cursor.execute("SELECT created_at FROM mature_coins WHERE symbol = ?", (symbol,))
                        existing = cursor.fetchone()
                        created_at = existing[0] if existing else coin_data.get('created_at', now)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO mature_coins (
                                symbol, timestamp, is_mature, candles_count,
                                min_required, config_min_rsi_low, config_max_rsi_high,
                                extra_maturity_data_json, updated_at, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT created_at FROM mature_coins WHERE symbol = ?), ?), ?)
                        """, (
                            symbol,
                            timestamp,
                            is_mature,
                            candles_count,
                            min_required,
                            config_min_rsi_low,
                            config_max_rsi_high,
                            extra_maturity_data_json,
                            symbol,
                            created_at,
                            now
                        ))
                    
                    conn.commit()
            
            logger.debug(f"💾 Зрелые монеты сохранены в нормализованные столбцы БД ({len(mature_coins)} монет)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения зрелых монет: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_mature_coins(self) -> Dict:
        """
        Загружает зрелые монеты из нормализованных столбцов
        
        Returns:
            Словарь {symbol: {timestamp: float, maturity_data: dict}}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, timestamp, is_mature, candles_count,
                           min_required, config_min_rsi_low, config_max_rsi_high,
                           extra_maturity_data_json
                    FROM mature_coins
                """)
                rows = cursor.fetchall()
                
                mature_coins = {}
                for row in rows:
                    symbol = row[0]
                    
                    # Собираем maturity_data из нормализованных столбцов
                    maturity_data = {
                        'is_mature': bool(row[2]),
                        'details': {
                            'candles_count': row[3],
                            'min_required': row[4],
                            'config_min_rsi_low': row[5],
                            'config_max_rsi_high': row[6]
                        }
                    }
                    
                    # Удаляем None значения из details
                    maturity_data['details'] = {k: v for k, v in maturity_data['details'].items() if v is not None}
                    
                    # Загружаем extra_maturity_data_json если есть
                    if row[7]:
                        try:
                            extra_data = json.loads(row[7])
                            # Добавляем поля из extra_data в maturity_data
                            for key, value in extra_data.items():
                                if key == 'extra_details':
                                    # Объединяем extra_details с details
                                    maturity_data['details'].update(value)
                                else:
                                    maturity_data[key] = value
                        except:
                            pass
                    
                    mature_coins[symbol] = {
                        'timestamp': row[1],
                        'maturity_data': maturity_data
                    }
                
                return mature_coins
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки зрелых монет: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ КЭША ПРОВЕРКИ ЗРЕЛОСТИ ====================
    
    def save_maturity_check_cache(self, coins_count: int, config_hash: str = None) -> bool:
        """
        Сохраняет кэш проверки зрелости в нормализованные столбцы
        
        Args:
            coins_count: Количество монет
            config_hash: Хеш конфигурации (JSON строка или словарь) (опционально)
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            # Парсим config_hash если он передан
            min_candles = None
            min_rsi_low = None
            max_rsi_high = None
            extra_config_json = None
            
            if config_hash:
                try:
                    # Если config_hash - это строка JSON, парсим её
                    if isinstance(config_hash, str):
                        config_data = json.loads(config_hash)
                    else:
                        config_data = config_hash
                    
                    # Извлекаем основные поля
                    min_candles = config_data.get('min_candles')
                    min_rsi_low = config_data.get('min_rsi_low')
                    max_rsi_high = config_data.get('max_rsi_high')
                    
                    # Собираем остальные поля в extra_config_json
                    extra_data = {}
                    known_fields = {'min_candles', 'min_rsi_low', 'max_rsi_high'}
                    for key, value in config_data.items():
                        if key not in known_fields:
                            extra_data[key] = value
                    
                    extra_config_json = json.dumps(extra_data) if extra_data else None
                except (json.JSONDecodeError, TypeError, AttributeError):
                    # Если не удалось распарсить - сохраняем как extra_config_json
                    extra_config_json = json.dumps({'config_hash': config_hash}) if config_hash else None
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Удаляем старые записи
                    cursor.execute("DELETE FROM maturity_check_cache")
                    
                    # Получаем created_at из существующей записи или используем текущее время
                    cursor.execute("SELECT created_at FROM maturity_check_cache LIMIT 1")
                    existing = cursor.fetchone()
                    created_at = existing[0] if existing else now
                    
                    # Вставляем новую запись в нормализованном формате
                    cursor.execute("""
                        INSERT INTO maturity_check_cache 
                        (coins_count, min_candles, min_rsi_low, max_rsi_high, extra_config_json, updated_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM maturity_check_cache LIMIT 1), ?))
                    """, (coins_count, min_candles, min_rsi_low, max_rsi_high, extra_config_json, now, created_at))
                    
                    conn.commit()
            
            logger.debug("💾 Кэш проверки зрелости сохранен в нормализованные столбцы БД")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения кэша проверки зрелости: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_maturity_check_cache(self) -> Dict:
        """
        Загружает кэш проверки зрелости из нормализованных столбцов
        
        Returns:
            Словарь {coins_count: int, config_hash: str} (config_hash собирается из нормализованных полей)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Проверяем, есть ли столбцы min_candles, min_rsi_low, max_rsi_high
                try:
                    cursor.execute("SELECT min_candles FROM maturity_check_cache LIMIT 1")
                    # Столбцы есть - используем нормализованную структуру
                    cursor.execute("""
                        SELECT coins_count, min_candles, min_rsi_low, max_rsi_high, extra_config_json
                        FROM maturity_check_cache
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)
                except sqlite3.OperationalError:
                    # Столбцов нет - используем старую структуру с config_hash
                    cursor.execute("""
                    SELECT coins_count, config_hash
                    FROM maturity_check_cache
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    # Определяем, какая структура (новая или старая)
                    if len(row) >= 5:
                        # Новая нормализованная структура
                        config_data = {}
                        if row[1] is not None:
                            config_data['min_candles'] = row[1]
                        if row[2] is not None:
                            config_data['min_rsi_low'] = row[2]
                        if row[3] is not None:
                            config_data['max_rsi_high'] = row[3]
                        
                        # Добавляем данные из extra_config_json если есть
                        if row[4]:
                            try:
                                extra_data = json.loads(row[4])
                                config_data.update(extra_data)
                            except:
                                pass
                        
                        # Формируем config_hash как JSON строку для обратной совместимости
                        config_hash = json.dumps(config_data) if config_data else None
                    else:
                        # Старая структура с config_hash
                        config_hash = row[1] if len(row) > 1 else None
                    
                    return {
                        'coins_count': row[0],
                        'config_hash': config_hash
                    }
                return {'coins_count': 0, 'config_hash': None}
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки кэша проверки зрелости: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {'coins_count': 0, 'config_hash': None}
    
    # ==================== МЕТОДЫ ДЛЯ ДЕЛИСТИРОВАННЫХ МОНЕТ ====================
    
    def save_delisted_coins(self, delisted: list) -> bool:
        """
        Сохраняет делистированные монеты
        
        Args:
            delisted: Список символов монет
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Удаляем старые записи
                    cursor.execute("DELETE FROM delisted")
                    
                    # Вставляем новые записи
                    for symbol in delisted:
                        cursor.execute("""
                            INSERT INTO delisted (symbol, delisted_at, created_at)
                            VALUES (?, ?, ?)
                        """, (symbol, now, now))
                    
                    conn.commit()
            
            logger.debug(f"💾 Делистированные монеты сохранены в БД ({len(delisted)} монет)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения делистированных монет: {e}")
            return False
    
    def load_delisted_coins(self) -> list:
        """
        Загружает делистированные монеты
        
        Returns:
            Список символов монет
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM delisted")
                rows = cursor.fetchall()
                
                return [row['symbol'] for row in rows]
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки делистированных монет: {e}")
            return []
    
    def is_coin_delisted(self, symbol: str) -> bool:
        """Проверяет, делистирована ли монета"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM delisted WHERE symbol = ?", (symbol,))
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"❌ Ошибка проверки делистирования: {e}")
            return False
    
    # ==================== МЕТОДЫ ДЛЯ КЭША СВЕЧЕЙ ====================
    
    def save_candles_cache(self, candles_cache: Dict) -> bool:
        """
        Сохраняет кэш свечей в нормализованные таблицы
        
        Args:
            candles_cache: Словарь {symbol: {candles: [], timeframe: '6h', ...}}
        
        Returns:
            True если успешно сохранено
        """
        try:
            now = datetime.now().isoformat()
            
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Проверяем, есть ли старая колонка candles_json (NOT NULL constraint)
                    try:
                        cursor.execute("SELECT candles_json FROM candles_cache LIMIT 1")
                        # Если запрос выполнился - значит старая структура, нужно пересоздать таблицу
                        logger.warning("⚠️ Обнаружена старая структура candles_cache с candles_json, пересоздаю таблицу...")
                        
                        # Сохраняем данные из старой таблицы
                        cursor.execute("""
                            SELECT id, symbol, timeframe, candles_count, first_candle_time, last_candle_time, updated_at, created_at
                            FROM candles_cache
                        """)
                        old_data = cursor.fetchall()
                        
                        # Сохраняем данные свечей (если таблица существует)
                        old_candles_data = []
                        try:
                            cursor.execute("SELECT cache_id, time, open, high, low, close, volume FROM candles_cache_data")
                            old_candles_data = cursor.fetchall()
                        except sqlite3.OperationalError:
                            # Таблица candles_cache_data не существует - это нормально для старой структуры
                            pass
                        
                        # Удаляем старую таблицу
                        cursor.execute("DROP TABLE IF EXISTS candles_cache")
                        cursor.execute("DROP TABLE IF EXISTS candles_cache_data")
                        
                        # Создаем новую таблицу без candles_json
                        cursor.execute("""
                            CREATE TABLE candles_cache (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                symbol TEXT UNIQUE NOT NULL,
                                timeframe TEXT NOT NULL DEFAULT '6h',
                                candles_count INTEGER DEFAULT 0,
                                first_candle_time INTEGER,
                                last_candle_time INTEGER,
                                updated_at TEXT NOT NULL,
                                created_at TEXT NOT NULL
                            )
                        """)
                        
                        # Создаем таблицу для данных свечей
                        cursor.execute("""
                            CREATE TABLE candles_cache_data (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                cache_id INTEGER NOT NULL,
                                time INTEGER NOT NULL,
                                open REAL NOT NULL,
                                high REAL NOT NULL,
                                low REAL NOT NULL,
                                close REAL NOT NULL,
                                volume REAL NOT NULL,
                                FOREIGN KEY (cache_id) REFERENCES candles_cache(id) ON DELETE CASCADE
                            )
                        """)
                        
                        # Восстанавливаем индексы
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_symbol ON candles_cache(symbol)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_updated ON candles_cache(updated_at)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_timeframe ON candles_cache(timeframe)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_id ON candles_cache_data(cache_id)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_time ON candles_cache_data(time)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_cache_data_cache_time ON candles_cache_data(cache_id, time)")
                        
                        # Восстанавливаем данные
                        for row in old_data:
                            cursor.execute("""
                                INSERT INTO candles_cache 
                                (id, symbol, timeframe, candles_count, first_candle_time, last_candle_time, updated_at, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, row)
                        
                        # Восстанавливаем данные свечей
                        for row in old_candles_data:
                            cursor.execute("""
                                INSERT INTO candles_cache_data 
                                (cache_id, time, open, high, low, close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, row)
                        
                        conn.commit()
                        logger.info("✅ Таблица candles_cache пересоздана без колонки candles_json")
                    except sqlite3.OperationalError:
                        # Колонка candles_json не существует - значит структура правильная
                        pass
                    
                    for symbol, cache_data in candles_cache.items():
                        candles = cache_data.get('candles', [])
                        timeframe = cache_data.get('timeframe', '6h')
                        
                        # Определяем временные границы
                        times = [c.get('time') for c in candles if c.get('time')]
                        first_time = min(times) if times else None
                        last_time = max(times) if times else None
                        candles_count = len(candles)
                        
                        # Сохраняем или обновляем метаданные кэша
                        cursor.execute("""
                            INSERT OR REPLACE INTO candles_cache 
                            (symbol, timeframe, candles_count, first_candle_time, last_candle_time, updated_at, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT created_at FROM candles_cache WHERE symbol = ?), ?))
                        """, (
                            symbol,
                            timeframe,
                            candles_count,
                            first_time,
                            last_time,
                            now,
                            symbol,
                            now
                        ))
                        
                        # Получаем cache_id
                        cursor.execute("SELECT id FROM candles_cache WHERE symbol = ?", (symbol,))
                        cache_row = cursor.fetchone()
                        if cache_row:
                            cache_id = cache_row[0]
                            
                            # Удаляем старые свечи для этого символа
                            cursor.execute("DELETE FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
                            
                            # Вставляем свечи в нормализованную таблицу
                            for candle in candles:
                                cursor.execute("""
                                    INSERT INTO candles_cache_data 
                                    (cache_id, time, open, high, low, close, volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    cache_id,
                                    candle.get('time'),
                                    candle.get('open'),
                                    candle.get('high'),
                                    candle.get('low'),
                                    candle.get('close'),
                                    candle.get('volume', 0)
                                ))
                    
                    conn.commit()
            
            logger.debug(f"💾 Кэш свечей сохранен в БД ({len(candles_cache)} символов)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения кэша свечей: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_candles_cache(self, symbol: Optional[str] = None) -> Dict:
        """
        Загружает кэш свечей из нормализованных таблиц
        
        Args:
            symbol: Символ монеты (если None, загружает все)
        
        Returns:
            Словарь {symbol: {candles: [], timeframe: '6h', ...}}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем, есть ли старая структура с candles_json
                try:
                    cursor.execute("SELECT candles_json FROM candles_cache LIMIT 1")
                    # Старая структура - используем её для обратной совместимости
                    if symbol:
                        cursor.execute("""
                            SELECT symbol, candles_json, timeframe, updated_at
                            FROM candles_cache
                            WHERE symbol = ?
                        """, (symbol,))
                    else:
                        cursor.execute("""
                            SELECT symbol, candles_json, timeframe, updated_at
                            FROM candles_cache
                        """)
                    
                    rows = cursor.fetchall()
                    result = {}
                    
                    for row in rows:
                        symbol_key = row['symbol']
                        candles = json.loads(row['candles_json']) if row['candles_json'] else []
                        timeframe = row['timeframe']
                        
                        result[symbol_key] = {
                            'candles': candles,
                            'timeframe': timeframe,
                            'updated_at': row['updated_at']
                        }
                    
                    return result
                except sqlite3.OperationalError:
                    # Новая нормализованная структура
                    if symbol:
                        cursor.execute("""
                            SELECT id, symbol, timeframe, updated_at
                            FROM candles_cache
                            WHERE symbol = ?
                        """, (symbol,))
                    else:
                        cursor.execute("""
                            SELECT id, symbol, timeframe, updated_at
                            FROM candles_cache
                        """)
                    
                    cache_rows = cursor.fetchall()
                    result = {}
                    
                    for cache_row in cache_rows:
                        cache_id = cache_row['id']
                        symbol_key = cache_row['symbol']
                        timeframe = cache_row['timeframe']
                        
                        # Загружаем свечи из нормализованной таблицы
                        cursor.execute("""
                            SELECT time, open, high, low, close, volume
                            FROM candles_cache_data
                            WHERE cache_id = ?
                            ORDER BY time ASC
                        """, (cache_id,))
                        
                        candle_rows = cursor.fetchall()
                        candles = []
                        for candle_row in candle_rows:
                            candles.append({
                                'time': candle_row['time'],
                                'open': candle_row['open'],
                                'high': candle_row['high'],
                                'low': candle_row['low'],
                                'close': candle_row['close'],
                                'volume': candle_row['volume']
                            })
                        
                        result[symbol_key] = {
                            'candles': candles,
                            'timeframe': timeframe,
                            'updated_at': cache_row['updated_at']
                        }
                    
                    return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки кэша свечей: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    def get_candles_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Получает свечи для конкретного символа
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Словарь с данными свечей или None
        """
        cache = self.load_candles_cache(symbol=symbol)
        return cache.get(symbol)
    
    def save_bot_trade_history(self, trade: Dict[str, Any]) -> Optional[int]:
        """
        Сохраняет историю сделки бота в БД
        
        Args:
            trade: Словарь с данными сделки
        
        Returns:
            ID сохраненной записи или None в случае ошибки
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                
                # Извлекаем значения с дефолтами
                bot_id = trade.get('bot_id') or trade.get('symbol', '')
                symbol = trade.get('symbol', '')
                direction = trade.get('direction', 'LONG')
                entry_price = trade.get('entry_price', 0.0)
                exit_price = trade.get('exit_price')
                entry_time = trade.get('entry_time', now)
                exit_time = trade.get('exit_time')
                entry_timestamp = trade.get('entry_timestamp') or trade.get('entry_timestamp_ms')
                exit_timestamp = trade.get('exit_timestamp') or trade.get('exit_timestamp_ms')
                position_size_usdt = trade.get('position_size_usdt')
                position_size_coins = trade.get('position_size_coins') or trade.get('size')
                pnl = trade.get('pnl')
                roi = trade.get('roi') or trade.get('roi_pct') or trade.get('closed_pnl_percent')
                status = trade.get('status', 'CLOSED')
                close_reason = trade.get('close_reason') or trade.get('reason')
                decision_source = trade.get('decision_source', 'SCRIPT')
                ai_decision_id = trade.get('ai_decision_id')
                ai_confidence = trade.get('ai_confidence')
                entry_rsi = trade.get('entry_rsi') or trade.get('rsi')
                exit_rsi = trade.get('exit_rsi')
                entry_trend = trade.get('entry_trend') or trade.get('trend')
                exit_trend = trade.get('exit_trend')
                entry_volatility = trade.get('entry_volatility')
                entry_volume_ratio = trade.get('entry_volume_ratio')
                is_successful = 1 if trade.get('is_successful', False) or (pnl and pnl > 0) else 0
                is_simulated = 1 if trade.get('is_simulated', False) else 0
                source = trade.get('source', 'bot')
                order_id = trade.get('order_id')
                
                # Обрабатываем extra_data_json
                extra_data = trade.get('extra_data') or trade.get('extra_data_json')
                if isinstance(extra_data, dict):
                    extra_data_json = json.dumps(extra_data, ensure_ascii=False) if extra_data else None
                elif isinstance(extra_data, str):
                    extra_data_json = extra_data if extra_data else None
                else:
                    extra_data_json = None
                
                # Конвертируем timestamps если нужно
                if entry_timestamp is None and entry_time:
                    try:
                        dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        entry_timestamp = dt.timestamp() * 1000
                    except:
                        pass
                
                if exit_timestamp is None and exit_time:
                    try:
                        dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                        exit_timestamp = dt.timestamp() * 1000
                    except:
                        pass
                
                # Проверяем на дубликаты (по bot_id, symbol, entry_price, entry_timestamp)
                if entry_timestamp:
                    cursor.execute("""
                        SELECT id FROM bot_trades_history
                        WHERE bot_id = ? AND symbol = ? AND entry_price = ? AND entry_timestamp = ?
                    """, (bot_id, symbol, entry_price, entry_timestamp))
                    existing = cursor.fetchone()
                    if existing:
                        # Обновляем существующую запись
                        cursor.execute("""
                            UPDATE bot_trades_history SET
                                exit_price = ?,
                                exit_time = ?,
                                exit_timestamp = ?,
                                pnl = ?,
                                roi = ?,
                                status = ?,
                                close_reason = ?,
                                exit_rsi = ?,
                                exit_trend = ?,
                                is_successful = ?,
                                updated_at = ?
                            WHERE id = ?
                        """, (exit_price, exit_time, exit_timestamp, pnl, roi, status, close_reason,
                              exit_rsi, exit_trend, is_successful, now, existing['id']))
                        conn.commit()
                        return existing['id']
                
                # Создаем новую запись
                cursor.execute("""
                    INSERT INTO bot_trades_history (
                        bot_id, symbol, direction, entry_price, exit_price,
                        entry_time, exit_time, entry_timestamp, exit_timestamp,
                        position_size_usdt, position_size_coins, pnl, roi,
                        status, close_reason, decision_source, ai_decision_id,
                        ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
                        entry_volatility, entry_volume_ratio, is_successful,
                        is_simulated, source, order_id, extra_data_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bot_id, symbol, direction, entry_price, exit_price,
                    entry_time, exit_time, entry_timestamp, exit_timestamp,
                    position_size_usdt, position_size_coins, pnl, roi,
                    status, close_reason, decision_source, ai_decision_id,
                    ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
                    entry_volatility, entry_volume_ratio, is_successful,
                    is_simulated, source, order_id, extra_data_json,
                    now, now
                ))
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения истории сделки: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def get_bot_trades_history(self, 
                              bot_id: Optional[str] = None,
                              symbol: Optional[str] = None,
                              status: Optional[str] = None,
                              decision_source: Optional[str] = None,
                              limit: Optional[int] = None,
                              offset: int = 0) -> List[Dict[str, Any]]:
        """
        Загружает историю сделок ботов из БД
        
        Args:
            bot_id: Фильтр по ID бота
            symbol: Фильтр по символу
            status: Фильтр по статусу (OPEN/CLOSED)
            decision_source: Фильтр по источнику решения (SCRIPT/AI/EXCHANGE_IMPORT)
            limit: Максимальное количество записей
            offset: Смещение для пагинации
        
        Returns:
            Список словарей с данными сделок
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Строим запрос с фильтрами
                query = "SELECT * FROM bot_trades_history WHERE 1=1"
                params = []
                
                if bot_id:
                    query += " AND bot_id = ?"
                    params.append(bot_id)
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if decision_source:
                    query += " AND decision_source = ?"
                    params.append(decision_source)
                
                query += " ORDER BY entry_timestamp DESC, created_at DESC"
                
                if limit:
                    query += " LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                result = []
                for row in rows:
                    trade = {
                        'id': row['id'],
                        'bot_id': row['bot_id'],
                        'symbol': row['symbol'],
                        'direction': row['direction'],
                        'entry_price': row['entry_price'],
                        'exit_price': row['exit_price'],
                        'entry_time': row['entry_time'],
                        'exit_time': row['exit_time'],
                        'entry_timestamp': row['entry_timestamp'],
                        'exit_timestamp': row['exit_timestamp'],
                        'position_size_usdt': row['position_size_usdt'],
                        'position_size_coins': row['position_size_coins'],
                        'pnl': row['pnl'],
                        'roi': row['roi'],
                        'status': row['status'],
                        'close_reason': row['close_reason'],
                        'decision_source': row['decision_source'],
                        'ai_decision_id': row['ai_decision_id'],
                        'ai_confidence': row['ai_confidence'],
                        'entry_rsi': row['entry_rsi'],
                        'exit_rsi': row['exit_rsi'],
                        'entry_trend': row['entry_trend'],
                        'exit_trend': row['exit_trend'],
                        'entry_volatility': row['entry_volatility'],
                        'entry_volume_ratio': row['entry_volume_ratio'],
                        'is_successful': bool(row['is_successful']),
                        'is_simulated': bool(row['is_simulated']),
                        'source': row['source'],
                        'order_id': row['order_id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    
                    # Парсим extra_data_json если есть
                    if row['extra_data_json']:
                        try:
                            trade['extra_data'] = json.loads(row['extra_data_json'])
                        except:
                            trade['extra_data'] = None
                    
                    result.append(trade)
                
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки истории сделок: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    # ==================== МЕТОДЫ МИГРАЦИИ ====================
    
    def _is_migration_needed(self) -> bool:
        """
        Проверяет, нужна ли миграция из JSON файлов
        
        Использует флаг в таблице db_metadata для отслеживания статуса миграции.
        
        Returns:
            True если миграция нужна (флаг = 0 или отсутствует), False если уже выполнена (флаг = 1)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем флаг миграции в метаданных БД
                try:
                    cursor.execute("""
                        SELECT value FROM db_metadata 
                        WHERE key = 'json_migration_completed'
                    """)
                    row = cursor.fetchone()
                    
                    if row:
                        migration_completed = row['value'] == '1'
                        if migration_completed:
                            logger.debug("ℹ️ Миграция из JSON уже выполнена (флаг в БД)")
                            return False
                        else:
                            logger.debug("ℹ️ Миграция из JSON еще не выполнена (флаг = 0)")
                            return True
                    else:
                        # Флага нет - значит БД новая, миграция нужна
                        logger.debug("ℹ️ Флаг миграции отсутствует - миграция нужна")
                        return True
                except sqlite3.OperationalError:
                    # Таблица db_metadata не существует - это старая БД без метаданных
                    # Проверяем наличие данных в основных таблицах как fallback
                    logger.debug("ℹ️ Таблица db_metadata не найдена, проверяем наличие данных...")
                    check_tables = [
                        'bots_state', 'bot_positions_registry', 'individual_coin_settings', 
                        'mature_coins', 'rsi_cache', 'process_state'
                    ]
                    
                    for table in check_tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            if count > 0:
                                # Есть данные - считаем что миграция уже выполнена
                                logger.debug(f"ℹ️ В таблице {table} есть {count} записей - миграция не требуется")
                                return False
                        except sqlite3.OperationalError:
                            continue
                    
                    # БД пуста - миграция нужна
                    return True
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки необходимости миграции: {e}")
            # В случае ошибки - выполняем миграцию на всякий случай
            return True
    
    def _set_migration_completed(self):
        """Устанавливает флаг что миграция из JSON выполнена"""
        self._set_metadata_flag('json_migration_completed', '1')
    
    def _set_metadata_flag(self, key: str, value: str):
        """
        Устанавливает флаг в метаданных БД
        
        Универсальный метод для установки любых флагов миграций или других метаданных.
        
        Args:
            key: Ключ флага (например, 'json_migration_completed', 'schema_v2_migrated')
            value: Значение флага (обычно '0' или '1', но может быть любое строковое значение)
        
        Example:
            ```python
            # Установить флаг миграции
            db._set_metadata_flag('json_migration_completed', '1')
            
            # Установить флаг миграции схемы
            db._set_metadata_flag('schema_v2_migrated', '1')
            
            # Установить версию БД
            db._set_metadata_flag('db_version', '2.0')
            ```
        """
        try:
            now = datetime.now().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO db_metadata (key, value, updated_at, created_at)
                    VALUES (?, ?, ?, 
                            COALESCE((SELECT created_at FROM db_metadata WHERE key = ?), ?))
                """, (key, value, now, key, now))
                conn.commit()
                logger.debug(f"✅ Флаг метаданных установлен: {key} = {value}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка установки флага метаданных {key}: {e}")
    
    def _get_metadata_flag(self, key: str, default: str = None) -> Optional[str]:
        """
        Получает значение флага из метаданных БД
        
        Универсальный метод для получения любых флагов миграций или других метаданных.
        
        Args:
            key: Ключ флага
            default: Значение по умолчанию если флаг не найден
        
        Returns:
            Значение флага или default
        
        Example:
            ```python
            # Проверить флаг миграции
            if db._get_metadata_flag('json_migration_completed') == '1':
                print("Миграция выполнена")
            
            # Получить версию БД
            version = db._get_metadata_flag('db_version', '1.0')
            ```
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM db_metadata WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return row['value']
                return default
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения флага метаданных {key}: {e}")
            return default
    
    def _is_migration_flag_set(self, flag_key: str) -> bool:
        """
        Проверяет, установлен ли флаг миграции
        
        Удобный метод для проверки флагов миграций.
        
        Args:
            flag_key: Ключ флага миграции
        
        Returns:
            True если флаг установлен в '1', False в противном случае
        
        Example:
            ```python
            # Проверить выполнена ли миграция JSON
            if not db._is_migration_flag_set('json_migration_completed'):
                # Выполнить миграцию
                db.migrate_json_to_database()
            
            # Проверить выполнена ли миграция схемы v2
            if not db._is_migration_flag_set('schema_v2_migrated'):
                # Выполнить миграцию схемы
                db.migrate_schema_v2()
            ```
        """
        flag_value = self._get_metadata_flag(flag_key, '0')
        return flag_value == '1'
    
    def migrate_json_to_database(self) -> Dict[str, int]:
        """
        Мигрирует данные из JSON файлов в БД (однократно)
        
        Проверяет наличие данных в БД перед миграцией - если данные уже есть,
        миграция не выполняется.
        
        Returns:
            Словарь с количеством мигрированных записей для каждого файла
        """
        # Проверяем, нужна ли миграция
        if not self._is_migration_needed():
            logger.debug("ℹ️ Миграция не требуется - данные уже есть в БД")
            return {}
        
        migration_stats = {}
        
        try:
            # Миграция bots_state.json
            bots_state_file = 'data/bots_state.json'
            if os.path.exists(bots_state_file):
                try:
                    with open(bots_state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data:
                            bots_data = data.get('bots', {})
                            auto_bot_config = data.get('auto_bot_config', {})
                            if self.save_bots_state(bots_data, auto_bot_config):
                                migration_stats['bots_state'] = 1
                                logger.info("📦 Мигрирован bots_state.json в БД")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции bots_state.json: {e}")
            
            # Миграция bot_positions_registry.json
            positions_file = 'data/bot_positions_registry.json'
            if os.path.exists(positions_file):
                try:
                    with open(positions_file, 'r', encoding='utf-8') as f:
                        registry = json.load(f)
                        if registry:
                            if self.save_bot_positions_registry(registry):
                                migration_stats['bot_positions_registry'] = len(registry)
                                logger.info(f"📦 Мигрирован bot_positions_registry.json в БД ({len(registry)} записей)")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции bot_positions_registry.json: {e}")
            
            # Миграция rsi_cache.json
            rsi_cache_file = 'data/rsi_cache.json'
            if os.path.exists(rsi_cache_file):
                try:
                    with open(rsi_cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        if cache_data:
                            coins_data = cache_data.get('coins', {})
                            stats = cache_data.get('stats', {})
                            if self.save_rsi_cache(coins_data, stats):
                                migration_stats['rsi_cache'] = 1
                                logger.info("📦 Мигрирован rsi_cache.json в БД")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции rsi_cache.json: {e}")
            
            # Миграция process_state.json
            process_state_file = 'data/process_state.json'
            if os.path.exists(process_state_file):
                try:
                    with open(process_state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data:
                            process_state = data.get('process_state', {})
                            if self.save_process_state(process_state):
                                migration_stats['process_state'] = 1
                                logger.info("📦 Мигрирован process_state.json в БД")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции process_state.json: {e}")
            
            # Миграция individual_coin_settings.json
            settings_file = 'data/individual_coin_settings.json'
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                        if settings:
                            if self.save_individual_coin_settings(settings):
                                migration_stats['individual_coin_settings'] = len(settings)
                                logger.info(f"📦 Мигрирован individual_coin_settings.json в БД ({len(settings)} записей)")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции individual_coin_settings.json: {e}")
            
            # Миграция mature_coins.json
            mature_coins_file = 'data/mature_coins.json'
            if os.path.exists(mature_coins_file):
                try:
                    with open(mature_coins_file, 'r', encoding='utf-8') as f:
                        mature_coins = json.load(f)
                        if mature_coins:
                            if self.save_mature_coins(mature_coins):
                                migration_stats['mature_coins'] = len(mature_coins)
                                logger.info(f"📦 Мигрирован mature_coins.json в БД ({len(mature_coins)} записей)")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции mature_coins.json: {e}")
            
            # Миграция maturity_check_cache.json
            maturity_cache_file = 'data/maturity_check_cache.json'
            if os.path.exists(maturity_cache_file):
                try:
                    with open(maturity_cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        if cache_data:
                            coins_count = cache_data.get('coins_count', 0)
                            config_hash = cache_data.get('config_hash')
                            if self.save_maturity_check_cache(coins_count, config_hash):
                                migration_stats['maturity_check_cache'] = 1
                                logger.info("📦 Мигрирован maturity_check_cache.json в БД")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции maturity_check_cache.json: {e}")
            
            # Миграция delisted.json
            delisted_file = 'data/delisted.json'
            if os.path.exists(delisted_file):
                try:
                    with open(delisted_file, 'r', encoding='utf-8') as f:
                        delisted = json.load(f)
                        if delisted and isinstance(delisted, list):
                            if self.save_delisted_coins(delisted):
                                migration_stats['delisted'] = len(delisted)
                                logger.info(f"📦 Мигрирован delisted.json в БД ({len(delisted)} записей)")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции delisted.json: {e}")
            
            # Миграция candles_cache.json
            candles_cache_file = 'data/candles_cache.json'
            if os.path.exists(candles_cache_file):
                try:
                    with open(candles_cache_file, 'r', encoding='utf-8') as f:
                        candles_cache = json.load(f)
                        if candles_cache and isinstance(candles_cache, dict):
                            if self.save_candles_cache(candles_cache):
                                migration_stats['candles_cache'] = len(candles_cache)
                                logger.info(f"📦 Мигрирован candles_cache.json в БД ({len(candles_cache)} символов)")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка миграции candles_cache.json: {e}")
            
            if migration_stats:
                logger.info(f"✅ Миграция завершена: {sum(migration_stats.values())} записей мигрировано")
                # Устанавливаем флаг что миграция выполнена
                self._set_migration_completed()
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции JSON в БД: {e}")
        
        return migration_stats
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Получает общую статистику базы данных
        
        Returns:
            Словарь со статистикой:
            {
                'bots_state_count': int,
                'bot_positions_registry_count': int,
                'rsi_cache_count': int,
                'process_state_count': int,
                'individual_coin_settings_count': int,
                'mature_coins_count': int,
                'maturity_check_cache_count': int,
                'delisted_count': int,
                'database_size_mb': float
            }
        
        Example:
            ```python
            db = get_bots_database()
            stats = db.get_database_stats()
            print(f"Ботов в БД: {stats['bots_state_count']}")
            print(f"Размер БД: {stats['database_size_mb']:.2f} MB")
            ```
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Подсчеты по таблицам
                tables = [
                    'bots_state', 'bot_positions_registry', 'rsi_cache', 
                    'candles_cache', 'process_state', 'individual_coin_settings', 
                    'mature_coins', 'maturity_check_cache', 'delisted'
                ]
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        stats[f"{table}_count"] = cursor.fetchone()[0]
                    except sqlite3.Error as e:
                        logger.debug(f"⚠️ Ошибка подсчета записей в {table}: {e}")
                        stats[f"{table}_count"] = 0
                
                # Размер базы данных (включая WAL файлы)
                db_size = 0
                if os.path.exists(self.db_path):
                    db_size += os.path.getsize(self.db_path)
                # Добавляем размер WAL файла если есть
                wal_path = f"{self.db_path}-wal"
                if os.path.exists(wal_path):
                    db_size += os.path.getsize(wal_path)
                # Добавляем размер SHM файла если есть
                shm_path = f"{self.db_path}-shm"
                if os.path.exists(shm_path):
                    db_size += os.path.getsize(shm_path)
                
                stats['database_size_mb'] = db_size / 1024 / 1024
                
                return stats
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики БД: {e}")
            return {}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Список доступных резервных копий БД
        
        Returns:
            Список словарей с информацией о резервных копиях
        """
        backups = []
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.basename(self.db_path)
        
        try:
            if not os.path.exists(db_dir):
                return backups
            
            # Ищем все файлы резервных копий
            for filename in os.listdir(db_dir):
                if filename.startswith(f"{db_name}.backup_") and not filename.endswith('-wal') and not filename.endswith('-shm'):
                    backup_path = os.path.join(db_dir, filename)
                    try:
                        file_size = os.path.getsize(backup_path)
                        # Извлекаем timestamp из имени файла
                        timestamp_str = filename.replace(f"{db_name}.backup_", "")
                        try:
                            backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except:
                            backup_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
                        
                        backups.append({
                            'path': backup_path,
                            'filename': filename,
                            'size_mb': file_size / 1024 / 1024,
                            'created_at': backup_time.isoformat(),
                            'timestamp': timestamp_str
                        })
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка обработки резервной копии {filename}: {e}")
            
            # Сортируем по дате создания (новые первыми)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            return backups
        except Exception as e:
            logger.error(f"❌ Ошибка получения списка резервных копий: {e}")
            return []
    
    def restore_from_backup(self, backup_path: str = None) -> bool:
        """
        Восстанавливает БД из резервной копии
        
        Args:
            backup_path: Путь к резервной копии (если None, используется последняя)
        
        Returns:
            True если восстановление успешно, False в противном случае
        """
        try:
            # Если путь не указан, используем последнюю резервную копию
            if backup_path is None:
                backups = self.list_backups()
                if not backups:
                    logger.error("❌ Нет доступных резервных копий")
                    return False
                backup_path = backups[0]['path']
                logger.info(f"📦 Используется последняя резервная копия: {backup_path}")
            
            if not os.path.exists(backup_path):
                logger.error(f"❌ Резервная копия не найдена: {backup_path}")
                return False
            
            logger.info(f"📦 Восстановление БД из резервной копии: {backup_path}")
            
            # Закрываем все соединения перед восстановлением
            # (в SQLite это не критично, но для чистоты)
            
            # Создаем резервную копию текущей БД перед восстановлением (на всякий случай)
            if os.path.exists(self.db_path):
                current_backup = self._backup_database()
                if current_backup:
                    logger.info(f"💾 Текущая БД сохранена в: {current_backup}")
            
            # Копируем резервную копию на место основной БД
            shutil.copy2(backup_path, self.db_path)
            
            # Восстанавливаем WAL и SHM файлы если есть
            wal_backup = f"{backup_path}-wal"
            shm_backup = f"{backup_path}-shm"
            wal_file = f"{self.db_path}-wal"
            shm_file = f"{self.db_path}-shm"
            
            if os.path.exists(wal_backup):
                shutil.copy2(wal_backup, wal_file)
                logger.debug("✅ Восстановлен WAL файл")
            elif os.path.exists(wal_file):
                # Удаляем старый WAL файл если нет резервной копии
                os.remove(wal_file)
                logger.debug("🗑️ Удален старый WAL файл")
            
            if os.path.exists(shm_backup):
                shutil.copy2(shm_backup, shm_file)
                logger.debug("✅ Восстановлен SHM файл")
            elif os.path.exists(shm_file):
                # Удаляем старый SHM файл если нет резервной копии
                os.remove(shm_file)
                logger.debug("🗑️ Удален старый SHM файл")
            
            # Проверяем целостность восстановленной БД
            is_ok, error_msg = self._check_integrity()
            if is_ok:
                logger.info("✅ БД успешно восстановлена из резервной копии")
                return True
            else:
                logger.error(f"❌ Восстановленная БД повреждена: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления БД из резервной копии: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False


# Глобальный экземпляр базы данных
_bots_database_instance = None
_bots_database_lock = threading.Lock()


def get_bots_database(db_path: str = None) -> BotsDatabase:
    """
    Получает глобальный экземпляр базы данных Bots
    
    База данных создается автоматически при первом вызове, если её еще нет.
    Все таблицы создаются автоматически. При первом запуске выполняется
    автоматическая миграция данных из JSON файлов в БД.
    
    Args:
        db_path: Путь к файлу базы данных (если None, используется data/bots_data.db)
    
    Returns:
        Экземпляр BotsDatabase
    """
    global _bots_database_instance
    
    with _bots_database_lock:
        if _bots_database_instance is None:
            logger.info("🔧 Инициализация Bots Database...")
            _bots_database_instance = BotsDatabase(db_path)
            
            # Автоматическая миграция при первом запуске (данные из JSON в БД)
            try:
                migration_stats = _bots_database_instance.migrate_json_to_database()
                if migration_stats:
                    logger.info(f"✅ Автоматическая миграция выполнена: {migration_stats}")
                else:
                    logger.debug("ℹ️ Миграция не требуется (нет данных в JSON или уже мигрировано)")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка автоматической миграции: {e}")
                # Продолжаем работу, даже если миграция не удалась
        
        return _bots_database_instance

