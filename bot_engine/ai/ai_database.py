#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реляционная база данных для хранения ВСЕХ данных AI модуля

Хранит:
- AI симуляции (simulated_trades)
- Реальные сделки ботов (bot_trades)
- История биржи (exchange_trades)
- Решения AI (ai_decisions)
- Сессии обучения (training_sessions)
- Метрики производительности (performance_metrics)
- Связи между данными для сложных запросов

Позволяет:
- Хранить миллиарды записей
- Делать JOIN запросы между таблицами
- Сравнивать данные из разных источников
- Анализировать паттерны
- Обучать ИИ на огромных объемах данных
"""

import sqlite3
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager
from functools import wraps
import logging

logger = logging.getLogger('AI.Database')


def _get_project_root() -> Path:
    """
    Определяет корень проекта относительно текущего файла.
    Корень проекта - директория, где лежит ai.py и bot_engine/
    """
    current = Path(__file__).resolve()
    # Поднимаемся от bot_engine/ai/ai_database.py до корня проекта
    # bot_engine/ai/ -> bot_engine/ -> корень
    for parent in [current.parent.parent.parent] + list(current.parents):
        if parent and (parent / 'ai.py').exists() and (parent / 'bot_engine').exists():
            return parent
    # Фолбек: поднимаемся на 2 уровня
    try:
        return current.parents[2]
    except IndexError:
        return current.parent


class AIDatabase:
    """
    Реляционная база данных для всех данных AI модуля
    """
    
    def __init__(self, db_path: str = None):
        """
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных (если None, используется data/ai_data.db)
        """
        if db_path is None:
            # ✅ ПУТЬ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
            project_root = _get_project_root()
            db_path = project_root / 'data' / 'ai_data.db'
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
        
        logger.info(f"✅ AI Database инициализирована: {db_path}")
    
    def _is_likely_corrupted(self) -> bool:
        """
        Проверяет, вероятно ли файл поврежден (только для очень очевидных случаев)
        НЕ удаляет БД автоматически - только предупреждает
        
        ВАЖНО: Не проверяем заголовок SQLite, так как это может давать ложные срабатывания
        при работе с удаленными БД, WAL режиме или когда файл открыт другим процессом.
        Полагаемся только на явную ошибку SQLite при подключении.
        """
        if not os.path.exists(self.db_path):
            return False
        
        try:
            # Проверяем только размер файла - если меньше 100 байт, это точно не БД
            # Это единственная безопасная проверка, которая не дает ложных срабатываний
            file_size = os.path.getsize(self.db_path)
            if file_size < 100:
                logger.warning(f"⚠️ Файл БД слишком маленький ({file_size} байт) - возможно поврежден")
                return True
            
            # НЕ проверяем заголовок - это может давать ложные срабатывания
            # SQLite сам проверит валидность при подключении
            
            return False
        except Exception as e:
            # Если не можем прочитать файл, не считаем его поврежденным
            # Возможно, он заблокирован другим процессом или на удаленном диске
            pass
            return False
    
    def _backup_database(self, max_retries: int = 3) -> Optional[str]:
        """
        Создает резервную копию БД в data/backups.
        
        Args:
            max_retries: Максимальное количество попыток при блокировке файла
        
        Returns:
            Путь к резервной копии или None если не удалось создать
        """
        if not os.path.exists(self.db_path):
            return None
        
        import shutil
        from datetime import datetime
        
        project_root = _get_project_root()
        backup_dir = project_root / 'data' / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"ai_data_{timestamp}.db"
        backup_path = str(backup_path)
        
        # Пытаемся создать резервную копию с retry логикой
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    try:
                        pass
                    except MemoryError:
                        pass
                    time.sleep(1.0 * attempt)
                
                try:
                    shutil.copy2(self.db_path, backup_path)
                except MemoryError:
                    print("⚠️ Нехватка памяти при создании резервной копии БД")
                    return None
                
                wal_file = self.db_path + '-wal'
                shm_file = self.db_path + '-shm'
                if os.path.exists(wal_file):
                    try:
                        shutil.copy2(wal_file, backup_path + '-wal')
                    except Exception as e:
                        pass
                if os.path.exists(shm_file):
                    try:
                        shutil.copy2(shm_file, backup_path + '-shm')
                    except Exception as e:
                        pass
                
                logger.warning(f"💾 Создана резервная копия БД: {backup_path}")
                return backup_path
            except MemoryError:
                # КРИТИЧНО: Нехватка памяти - не пытаемся создавать резервную копию
                print("⚠️ Нехватка памяти при создании резервной копии БД")
                return None
            except PermissionError as e:
                # Файл заблокирован другим процессом
                if attempt < max_retries - 1:
                    try:
                        pass
                    except MemoryError:
                        pass
                    continue
                else:
                    try:
                        logger.error(f"❌ Не удалось создать резервную копию БД после {max_retries} попыток: {e}")
                    except MemoryError:
                        print(f"❌ Не удалось создать резервную копию БД: {e}")
                    return None
            except Exception as e:
                error_str = str(e).lower()
                if "процесс не может получить доступ к файлу" in error_str or "file is locked" in error_str or "access" in error_str:
                    # Файл заблокирован
                    if attempt < max_retries - 1:
                        pass
                        continue
                    else:
                        logger.error(f"❌ Не удалось создать резервную копию БД после {max_retries} попыток: {e}")
                        return None
                else:
                    # Другая ошибка - не повторяем
                    try:
                        logger.error(f"❌ Ошибка создания резервной копии БД: {e}")
                    except MemoryError:
                        print(f"❌ Ошибка создания резервной копии БД: {e}")
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
            main_tables = ['simulated_trades', 'bot_trades', 'exchange_trades', 'candles_history']
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
            pass
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
    
    @contextmanager
    def _get_connection(self, retry_on_locked: bool = True, max_retries: int = 5):
        """
        Контекстный менеджер для работы с БД с поддержкой retry при блокировках
        
        Args:
            retry_on_locked: Повторять попытки при ошибке "database is locked"
            max_retries: Максимальное количество попыток при блокировке
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
                    # Обрабатываем ошибки блокировки (из with-блока).
                    # КРИТИЧНО: не делать continue — иначе генератор снова сделает yield и возникнет "generator didn't stop after throw()". Retry делает вызывающий код.
                    if "database is locked" in error_str or "locked" in error_str:
                        conn.rollback()
                        conn.close()
                        logger.warning(f"⚠️ БД заблокирована при записи (попытка {attempt + 1})")
                        raise
                    elif "disk i/o error" in error_str or "i/o error" in error_str:
                        # Критическая ошибка I/O - БД может быть повреждена
                        conn.rollback()
                        conn.close()
                        try:
                            logger.error(f"❌ КРИТИЧНО: Ошибка I/O при работе с БД: {e}")
                            logger.warning("🔧 Попытка автоматического исправления...")
                        except MemoryError:
                            print("❌ КРИТИЧНО: Ошибка I/O при работе с БД")
                        
                        if attempt == 0:
                            # Пытаемся исправить только один раз (если не MemoryError)
                            try:
                                if self._repair_database():
                                    try:
                                        logger.info("✅ БД исправлена, повторяем операцию...")
                                    except MemoryError:
                                        print("✅ БД исправлена, повторяем операцию...")
                                    time.sleep(1)  # Небольшая задержка перед повтором
                                    continue
                                else:
                                    try:
                                        logger.error("❌ Не удалось исправить БД после I/O ошибки")
                                    except MemoryError:
                                        print("❌ Не удалось исправить БД после I/O ошибки")
                                    raise
                            except MemoryError:
                                # Нехватка памяти - не пытаемся исправлять
                                print("⚠️ Нехватка памяти, пропускаем исправление БД")
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
                # Восстанавливаем БД при критических ошибках повреждения
                if "file is not a database" in error_str or ("not a database" in error_str and "unable to open" not in error_str):
                    logger.error(f"❌ Файл БД поврежден (явная ошибка SQLite): {self.db_path}")
                    logger.error(f"❌ Ошибка: {e}")
                    # Восстанавливаем БД только при явной ошибке
                    self._recreate_database()
                    # Пытаемся подключиться снова (только один раз)
                    if attempt == 0:
                        continue
                    else:
                        raise
                elif "database disk image is malformed" in error_str or "malformed" in error_str:
                    # Критическая ошибка - БД повреждена
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
                elif "disk i/o error" in error_str or "i/o error" in error_str:
                    # Критическая ошибка I/O
                    try:
                        logger.error(f"❌ КРИТИЧНО: Ошибка I/O при подключении к БД: {self.db_path}")
                        logger.error(f"❌ Ошибка: {e}")
                        logger.warning("🔧 Попытка автоматического исправления...")
                    except MemoryError:
                        print(f"❌ КРИТИЧНО: Ошибка I/O при подключении к БД")
                    
                    if attempt == 0:
                        # Пытаемся исправить только один раз (если не MemoryError)
                        try:
                            if self._repair_database():
                                try:
                                    logger.info("✅ БД исправлена, повторяем подключение...")
                                except MemoryError:
                                    print("✅ БД исправлена, повторяем подключение...")
                                time.sleep(1)  # Небольшая задержка перед повтором
                                continue
                            else:
                                try:
                                    logger.error("❌ Не удалось исправить БД после I/O ошибки")
                                except MemoryError:
                                    print("❌ Не удалось исправить БД после I/O ошибки")
                                raise
                        except MemoryError:
                            # Нехватка памяти - не пытаемся исправлять
                            print("⚠️ Нехватка памяти, пропускаем исправление БД")
                            raise
                    else:
                        raise
                elif "database is locked" in error_str or "locked" in error_str:
                    # Ошибка блокировки (или исключение проброшено из inner except).
                    # КРИТИЧНО: не делать continue — иначе "generator didn't stop after throw()".
                    last_error = e
                    logger.warning(f"⚠️ БД заблокирована при подключении после {max_retries} попыток")
                    raise
                else:
                    # Другие ошибки - не повторяем
                    raise
        
        # Если дошли сюда, значит все попытки исчерпаны
        if last_error:
            raise last_error
    
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
        
        pass
        
        try:
            # Сначала проверяем, не заблокирована ли БД другим процессом
            # Пытаемся простое подключение с коротким таймаутом
            pass
            try:
                test_conn = sqlite3.connect(self.db_path, timeout=1.0)
                test_conn.close()
                pass
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # БД заблокирована - пропускаем проверку, чтобы не блокировать запуск
                    pass
                    return True, None
                raise
            
            # ⚡ ИСПРАВЛЕНО: Создаем новое соединение для каждой операции
            # Это гарантирует, что все операции выполняются в том же потоке
            pass
            try:
                # Получаем размер БД перед проверкой
                try:
                    db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                    db_size_gb = db_size_mb / 1024  # GB
                    pass
                    
                    # Пропускаем проверку целостности для очень больших БД (>1 GB)
                    if db_size_mb > 1024:  # Больше 1 GB
                        logger.info(f"   [3/4] ⚠️ БД очень большая ({db_size_gb:.2f} GB), пропускаем проверку целостности для ускорения запуска")
                        return True, None
                except Exception as e:
                    pass
                
                # Создаем новое соединение для проверки режима журнала
                conn1 = sqlite3.connect(self.db_path, timeout=5.0)
                cursor1 = conn1.cursor()
                
                # Проверяем режим журнала
                pass
                cursor1.execute("PRAGMA journal_mode")
                journal_mode = cursor1.fetchone()[0]
                pass
                
                # Если WAL режим - делаем checkpoint для синхронизации
                if journal_mode.upper() == 'WAL':
                    pass
                    try:
                        cursor1.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        conn1.commit()
                        pass
                    except Exception as e:
                        pass
                
                conn1.close()
                
                # Создаем новое соединение для проверки целостности
                pass
                conn2 = sqlite3.connect(self.db_path, timeout=5.0)
                cursor2 = conn2.cursor()
                
                # Получаем информацию о БД перед проверкой
                try:
                    cursor2.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor2.fetchone()[0]
                    pass
                except Exception as e:
                    pass
                
                # Устанавливаем таймаут для операции
                pass
                cursor2.execute("PRAGMA busy_timeout = 2000")  # 2 секунды
                pass
                
                # ⚡ ИСПРАВЛЕНО: Выполняем проверку целостности в том же потоке
                import time
                pass
                start_time = time.time()
                
                try:
                    # Выполняем проверку напрямую в текущем потоке
                    cursor2.execute("PRAGMA quick_check")
                    result = cursor2.fetchone()[0]
                    elapsed = time.time() - start_time
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"   [4/4] ❌ Ошибка при выполнении PRAGMA quick_check (после {elapsed:.2f}s): {e}")
                    conn2.close()
                    return True, None  # Считаем БД валидной при ошибке
                
                pass
                pass
                
                if result == "ok":
                    pass
                else:
                    logger.warning(f"   [4/4] ⚠️ Обнаружены проблемы в БД: {result[:200]}")
                
                conn2.close()
                pass
                
                if result == "ok":
                    pass
                    return True, None
                else:
                    # Есть проблемы - но не делаем полную проверку (она может быть очень долгой)
                    logger.warning(f"⚠️ Обнаружены проблемы в БД: {result}")
                    return False, result
                    
            except sqlite3.OperationalError as e:
                error_str = str(e).lower()
                if "locked" in error_str:
                    # БД заблокирована - пропускаем проверку
                    pass
                    return True, None
                # Другие ошибки - считаем БД валидной, чтобы не блокировать запуск
                logger.warning(f"⚠️ Ошибка проверки целостности БД: {e}, продолжаем работу...")
                return True, None
                
        except Exception as e:
            # В случае ошибки считаем БД валидной, чтобы не блокировать запуск
            pass
            return True, None  # Возвращаем True, чтобы не блокировать запуск

    def _migrate_corrupted_to_fresh(self) -> bool:
        """
        Миграция при критическом повреждении: сохраняем повреждённую БД как .corrupted_*,
        создаём новую пустую ai_data.db при следующем подключении.
        Для всех пользователей при обновлении — приложение стартует без падения.
        """
        if not os.path.exists(self.db_path):
            return False
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupted_path = f"{self.db_path}.corrupted_{ts}"
            os.rename(self.db_path, corrupted_path)
            for ext in ('-wal', '-shm'):
                p = self.db_path + ext
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        try:
                            os.rename(p, f"{p}.corrupted_{ts}")
                        except OSError:
                            pass
            logger.warning(f"🔄 Миграция: повреждённая БД сохранена как {corrupted_path}")
            logger.info("✅ Создаётся новая ai_data.db (схема будет применена при инициализации)")
            return True
        except OSError as e:
            logger.warning(f"⚠️ Миграция повреждённой БД не удалась (файл занят?): {e}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Миграция повреждённой БД не удалась: {e}")
            return False

    def _repair_database(self) -> bool:
        """
        Пытается исправить поврежденную БД
        
        Returns:
            True если удалось исправить, False в противном случае
        """
        try:
            # КРИТИЧНО: Не пытаемся исправлять при нехватке памяти
            try:
                logger.warning("🔧 Попытка исправления БД...")
            except MemoryError:
                # Используем print вместо logger при MemoryError
                print("⚠️ КРИТИЧНО: Нехватка памяти, пропускаем исправление БД")
                return False
            
            # В процессе ai.py бэкапы создаёт планировщик в app.py — не создаём здесь,
            # иначе при частых ошибках БД (locked/I/O) получаются сотни бэкапов в секунду
            backup_created = False
            if os.environ.get('INFOBOT_AI_PROCESS') != 'true':
                try:
                    backup_path = self._backup_database(max_retries=3)
                    backup_created = backup_path is not None
                except MemoryError:
                    print("⚠️ Нехватка памяти при создании резервной копии, пропускаем...")
                    backup_created = False

            if not backup_created:
                try:
                    logger.warning("⚠️ Не удалось создать резервную копию перед исправлением (файл может быть заблокирован)")
                    logger.info("💡 Попробую использовать существующие резервные копии для восстановления...")
                except MemoryError:
                    print("⚠️ Не удалось создать резервную копию перед исправлением")
            
            # Пытаемся использовать VACUUM для исправления (только если БД не слишком повреждена)
            vacuum_tried = False
            vacuum_failed_malformed = False
            try:
                # Подключаемся без retry для VACUUM (может быть долго)
                conn = sqlite3.connect(self.db_path, timeout=300.0)  # 5 минут для VACUUM
                cursor = conn.cursor()
                try:
                    logger.info("🔧 Выполняю VACUUM для исправления БД (это может занять время)...")
                except MemoryError:
                    print("🔧 Выполняю VACUUM для исправления БД...")
                cursor.execute("VACUUM")
                conn.commit()
                conn.close()
                try:
                    logger.info("✅ VACUUM выполнен")
                except MemoryError:
                    print("✅ VACUUM выполнен")
                vacuum_tried = True
            except MemoryError:
                # Нехватка памяти - пропускаем VACUUM
                print("⚠️ Нехватка памяти при выполнении VACUUM, пропускаем...")
                vacuum_tried = False
            except Exception as vacuum_error:
                error_str = str(vacuum_error).lower()
                if "malformed" in error_str or "disk i/o error" in error_str:
                    vacuum_failed_malformed = True
                    try:
                        logger.warning(f"⚠️ VACUUM невозможен из-за критического повреждения: {vacuum_error}")
                        logger.info("💡 Пропускаю VACUUM, при малформном повреждении бэкапы — копия битой БД, восстановление не выполняю.")
                    except MemoryError:
                        print("⚠️ VACUUM невозможен из-за критического повреждения")
                else:
                    try:
                        logger.warning(f"⚠️ VACUUM не помог: {vacuum_error}")
                    except MemoryError:
                        print("⚠️ VACUUM не помог")
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
            
            # При критическом повреждении (malformed) бэкапы — копия битой БД. Восстановление не делаем, сразу миграция.
            if vacuum_failed_malformed:
                try:
                    logger.warning("🔄 Пропуск восстановления из бэкапа (бэкапы созданы из повреждённой БД). Миграция на новую ai_data.db.")
                except MemoryError:
                    print("🔄 Миграция на новую ai_data.db (бэкапы — копия битой БД).")
                if self._migrate_corrupted_to_fresh():
                    return True
                try:
                    logger.error("⚠️ Миграция повреждённой БД не удалась (файл занят?). Закройте процессы и перезапустите.")
                except MemoryError:
                    pass
                return False

            # Пытаемся восстановить из резервной копии (только если VACUUM не падал с malformed)
            try:
                logger.info("🔄 Попытка восстановления из резервной копии...")
            except MemoryError:
                print("🔄 Попытка восстановления из резервной копии...")

            try:
                backups = self.list_backups()
            except MemoryError:
                print("⚠️ Нехватка памяти при получении списка резервных копий")
                backups = []

            restored_ok = False
            if backups:
                # Если мы создали резервную копию только что, используем более старую
                if backup_created and len(backups) > 1:
                    older_backup = backups[1]['path']
                    try:
                        logger.info(f"📦 Восстанавливаю из более старой резервной копии: {older_backup}")
                    except MemoryError:
                        print("📦 Восстанавливаю из более старой резервной копии")
                    try:
                        restored_ok = self.restore_from_backup(older_backup)
                    except MemoryError:
                        print("⚠️ Нехватка памяти при восстановлении из резервной копии")
                else:
                    latest_backup = backups[0]['path']
                    try:
                        logger.info(f"📦 Восстанавливаю из резервной копии: {latest_backup}")
                    except MemoryError:
                        print("📦 Восстанавливаю из резервной копии")
                    try:
                        restored_ok = self.restore_from_backup(latest_backup)
                    except MemoryError:
                        print("⚠️ Нехватка памяти при восстановлении из резервной копии")

            if restored_ok:
                is_ok, _ = self._check_integrity()
                if not is_ok:
                    try:
                        logger.warning("⚠️ Восстановленная БД повреждена. Миграция на новую ai_data.db.")
                    except MemoryError:
                        print("⚠️ Восстановленная БД повреждена, миграция на новую.")
                    if self._migrate_corrupted_to_fresh():
                        return True
                    return False
                return True

            if not backups:
                try:
                    logger.error("❌ Нет доступных резервных копий для восстановления")
                    if not backup_created:
                        logger.error("❌ КРИТИЧНО: Не удалось создать резервную копию и нет существующих копий!")
                except MemoryError:
                    print("❌ Нет доступных резервных копий для восстановления")
            else:
                try:
                    logger.error("❌ Не удалось восстановить БД из резервной копии")
                except MemoryError:
                    print("❌ Не удалось восстановить БД из резервной копии")

            if self._migrate_corrupted_to_fresh():
                return True
            try:
                logger.error("⚠️ БД останется поврежденной. Рекомендуется:")
                logger.error("   1. Закрыть все процессы, использующие БД")
                logger.error("   2. Попробовать восстановить вручную: db.restore_from_backup()")
                logger.error("   3. Или удалить ai_data.db и перезапустить (будет создана новая)")
            except MemoryError:
                pass
            return False
        except MemoryError:
            # КРИТИЧНО: Не логируем при MemoryError (это вызывает рекурсию)
            print("⚠️ КРИТИЧНО: Нехватка памяти при исправлении БД, пропускаем...")
            return False
        except Exception as e:
            # Используем безопасное логирование
            try:
                logger.error(f"❌ Ошибка исправления БД: {e}")
            except MemoryError:
                print(f"❌ Ошибка исправления БД: {e}")
            try:
                import traceback
                pass
            except MemoryError:
                pass
            return False
    
    def _init_database(self):
        """Создает все таблицы и индексы"""
        # Проверяем целостность БД при каждом запуске
        if os.path.exists(self.db_path):
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
                pass
        
        # SQLite автоматически создает файл БД при первом подключении
        # Не нужно создавать пустой файл через touch() - это создает невалидную БД
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Миграция: добавляем новые поля если их нет
            self._migrate_schema(cursor, conn)
            
            # ==================== ТАБЛИЦА: AI СИМУЛЯЦИИ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все параметры из JSON в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time INTEGER NOT NULL,
                    exit_time INTEGER NOT NULL,
                    entry_rsi REAL,
                    exit_rsi REAL,
                    entry_trend TEXT,
                    exit_trend TEXT,
                    entry_volatility REAL,
                    entry_volume_ratio REAL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    roi REAL,
                    exit_reason TEXT,
                    is_successful INTEGER NOT NULL DEFAULT 0,
                    duration_candles INTEGER,
                    entry_idx INTEGER,
                    exit_idx INTEGER,
                    simulation_timestamp TEXT NOT NULL,
                    training_session_id INTEGER,
                    -- RSI параметры (из rsi_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Risk параметры (из risk_params_json)
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    trailing_stop_activation REAL,
                    trailing_stop_distance REAL,
                    trailing_take_distance REAL,
                    trailing_update_interval REAL,
                    break_even_trigger REAL,
                    break_even_protection REAL,
                    max_position_hours REAL,
                    -- Дополнительные JSON для сложных структур (оставляем для обратной совместимости)
                    config_params_json TEXT,
                    filters_params_json TEXT,
                    entry_conditions_json TEXT,
                    exit_conditions_json TEXT,
                    restrictions_json TEXT,
                    extra_params_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (training_session_id) REFERENCES training_sessions(id)
                )
            """)
            
            # Индексы для simulated_trades
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_symbol ON simulated_trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_entry_time ON simulated_trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_exit_time ON simulated_trades(exit_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_pnl ON simulated_trades(pnl)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_successful ON simulated_trades(is_successful)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_session ON simulated_trades(training_session_id)")
            
            # ==================== ТАБЛИЦА: РЕАЛЬНЫЕ СДЕЛКИ БОТОВ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все параметры из JSON в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    bot_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    roi REAL,
                    status TEXT NOT NULL,
                    decision_source TEXT NOT NULL,
                    ai_decision_id TEXT,
                    ai_confidence REAL,
                    entry_rsi REAL,
                    exit_rsi REAL,
                    entry_trend TEXT,
                    exit_trend TEXT,
                    entry_volatility REAL,
                    entry_volume_ratio REAL,
                    close_reason TEXT,
                    position_size_usdt REAL,
                    position_size_coins REAL,
                    -- RSI параметры (из config_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Risk параметры (из config_params_json)
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    trailing_stop_activation REAL,
                    trailing_stop_distance REAL,
                    trailing_take_distance REAL,
                    trailing_update_interval REAL,
                    break_even_trigger REAL,
                    break_even_protection REAL,
                    max_position_hours REAL,
                    -- Дополнительные JSON для сложных структур (оставляем для обратной совместимости)
                    entry_data_json TEXT,
                    exit_market_data_json TEXT,
                    filters_params_json TEXT,
                    entry_conditions_json TEXT,
                    exit_conditions_json TEXT,
                    restrictions_json TEXT,
                    extra_config_json TEXT,
                    is_simulated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для bot_trades
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_symbol ON bot_trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_bot_id ON bot_trades(bot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_status ON bot_trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_decision_source ON bot_trades(decision_source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_pnl ON bot_trades(pnl)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_entry_time ON bot_trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_ai_decision ON bot_trades(ai_decision_id)")
            
            # ==================== ТАБЛИЦА: ИСТОРИЯ БИРЖИ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchange_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    roi REAL NOT NULL,
                    position_size_usdt REAL,
                    position_size_coins REAL,
                    order_id TEXT,
                    source TEXT NOT NULL,
                    saved_timestamp TEXT NOT NULL,
                    is_real INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для exchange_trades
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange_trades_symbol ON exchange_trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange_trades_entry_time ON exchange_trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange_trades_exit_time ON exchange_trades(exit_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange_trades_pnl ON exchange_trades(pnl)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange_trades_order_id ON exchange_trades(order_id)")
            
            # ==================== ТАБЛИЦА: РЕШЕНИЯ AI (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: основные поля из JSON в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    rsi REAL,
                    trend TEXT,
                    price REAL,
                    -- Дополнительные поля из market_data (если есть)
                    volume REAL,
                    volatility REAL,
                    volume_ratio REAL,
                    -- Дополнительные поля из decision_params (если есть)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    -- Дополнительные JSON для сложных структур
                    market_data_json TEXT,
                    decision_params_json TEXT,
                    extra_market_data_json TEXT,
                    extra_decision_params_json TEXT,
                    created_at TEXT NOT NULL,
                    executed_at TEXT,
                    result_pnl REAL,
                    result_successful INTEGER
                )
            """)
            
            # Индексы для ai_decisions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol ON ai_decisions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_decision_id ON ai_decisions(decision_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_created_at ON ai_decisions(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_result ON ai_decisions(result_successful)")
            
            # ==================== ТАБЛИЦА: ПОСЛЕДНИЕ РЕКОМЕНДАЦИИ AI (для чтения из bots.py) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_recommendations (
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    should_open INTEGER NOT NULL,
                    signal TEXT,
                    confidence REAL,
                    reason TEXT,
                    ai_used INTEGER,
                    smc_used INTEGER,
                    data_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, direction)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_recommendations_updated ON ai_recommendations(updated_at)")
            
            # ==================== ТАБЛИЦА: СЕССИИ ОБУЧЕНИЯ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_type TEXT NOT NULL,
                    training_seed INTEGER,
                    coins_processed INTEGER DEFAULT 0,
                    models_saved INTEGER DEFAULT 0,
                    candles_processed INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    failed_trades INTEGER DEFAULT 0,
                    win_rate REAL,
                    total_pnl REAL,
                    accuracy REAL,
                    mse REAL,
                    params_used INTEGER DEFAULT 0,
                    params_total INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    metadata_json TEXT
                )
            """)
            
            # Индексы для training_sessions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_sessions_type ON training_sessions(session_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_sessions_started_at ON training_sessions(started_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status)")
            
            # ==================== ТАБЛИЦА: МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data_json TEXT,
                    recorded_at TEXT NOT NULL,
                    training_session_id INTEGER,
                    FOREIGN KEY (training_session_id) REFERENCES training_sessions(id)
                )
            """)
            
            # Индексы для performance_metrics
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_metrics_symbol ON performance_metrics(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_metrics_type ON performance_metrics(metric_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_metrics_recorded_at ON performance_metrics(recorded_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_metrics_session ON performance_metrics(training_session_id)")
            
            # ==================== ТАБЛИЦА: ОБРАЗЦЫ ДЛЯ ОБУЧЕНИЯ ПРЕДСКАЗАТЕЛЯ КАЧЕСТВА ПАРАМЕТРОВ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все параметры из JSON в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- RSI параметры (из rsi_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Risk параметры (из risk_params_json)
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    trailing_stop_activation REAL,
                    trailing_stop_distance REAL,
                    trailing_take_distance REAL,
                    trailing_update_interval REAL,
                    break_even_trigger REAL,
                    break_even_protection REAL,
                    max_position_hours REAL,
                    -- Метрики
                    win_rate REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    quality REAL NOT NULL,
                    blocked INTEGER NOT NULL DEFAULT 0,
                    rsi_entered_zones INTEGER DEFAULT 0,
                    filters_blocked INTEGER DEFAULT 0,
                    -- Дополнительные JSON для сложных структур
                    block_reasons_json TEXT,
                    extra_rsi_params_json TEXT,
                    extra_risk_params_json TEXT,
                    symbol TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для parameter_training_samples
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_param_samples_symbol ON parameter_training_samples(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_param_samples_quality ON parameter_training_samples(quality)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_param_samples_blocked ON parameter_training_samples(blocked)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_param_samples_created_at ON parameter_training_samples(created_at)")
            
            # ==================== ТАБЛИЦА: ИСПОЛЬЗОВАННЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все RSI параметры в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS used_training_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_hash TEXT UNIQUE NOT NULL,
                    -- RSI параметры (из rsi_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Дополнительные RSI параметры
                    extra_rsi_params_json TEXT,
                    -- Метрики
                    training_seed INTEGER,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    signal_accuracy REAL DEFAULT 0.0,
                    trades_count INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0.0,
                    symbol TEXT,
                    used_at TEXT NOT NULL,
                    update_count INTEGER DEFAULT 1
                )
            """)
            
            # Индексы для used_training_parameters
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_used_params_hash ON used_training_parameters(param_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_used_params_symbol ON used_training_parameters(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_used_params_rating ON used_training_parameters(rating)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_used_params_win_rate ON used_training_parameters(win_rate)")
            
            # ==================== ТАБЛИЦА: ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ МОНЕТ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все RSI параметры в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS best_params_per_symbol (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    -- RSI параметры (из rsi_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Дополнительные RSI параметры
                    extra_rsi_params_json TEXT,
                    -- Метрики
                    rating REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для best_params_per_symbol
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_best_params_symbol ON best_params_per_symbol(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_best_params_rating ON best_params_per_symbol(rating)")
            
            # ==================== ТАБЛИЦА: ЗАБЛОКИРОВАННЫЕ ПАРАМЕТРЫ (НОРМАЛИЗОВАННАЯ) ====================
            # НОВАЯ НОРМАЛИЗОВАННАЯ СТРУКТУРА: все RSI параметры в отдельных столбцах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocked_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_hash TEXT,
                    -- RSI параметры (из rsi_params_json)
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Дополнительные RSI параметры
                    extra_rsi_params_json TEXT,
                    block_reasons_json TEXT,
                    blocked_attempts INTEGER DEFAULT 0,
                    blocked_long INTEGER DEFAULT 0,
                    blocked_short INTEGER DEFAULT 0,
                    symbol TEXT,
                    blocked_at TEXT NOT NULL
                )
            """)
            
            # Индексы для blocked_params
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_params_symbol ON blocked_params(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_params_blocked_at ON blocked_params(blocked_at)")
            
            # ==================== ТАБЛИЦА: ЦЕЛЕВЫЕ ЗНАЧЕНИЯ WIN RATE ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS win_rate_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    target_win_rate REAL NOT NULL,
                    current_win_rate REAL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для win_rate_targets
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_win_rate_targets_symbol ON win_rate_targets(symbol)")
            
            # ==================== ТАБЛИЦА: БЛОКИРОВКИ ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_locks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    process_id TEXT NOT NULL,
                    hostname TEXT,
                    locked_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PROCESSING',
                    UNIQUE(symbol)
                )
            """)
            
            # Индексы для training_locks
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_locks_symbol ON training_locks(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_locks_expires_at ON training_locks(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_locks_status ON training_locks(status)")
            
            # ==================== ТАБЛИЦА: ИСТОРИЯ СВЕЧЕЙ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '6h',
                    candle_time INTEGER NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, candle_time)
                )
            """)
            
            # Индексы для candles_history
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles_history(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timeframe ON candles_history(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_time ON candles_history(candle_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles_history(symbol, candle_time)")
            
            # ==================== ТАБЛИЦА: ВЕРСИИ МОДЕЛЕЙ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE NOT NULL,
                    model_type TEXT NOT NULL,
                    version_number TEXT,
                    model_path TEXT,
                    accuracy REAL,
                    mse REAL,
                    win_rate REAL,
                    total_pnl REAL,
                    training_samples INTEGER,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для model_versions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON model_versions(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_model_type ON model_versions(model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at)")
            
            # ==================== ТАБЛИЦА: СНИМКИ ДАННЫХ БОТОВ ====================
            # ВАЖНО: Таблица bots_data_snapshots БОЛЬШЕ НЕ ИСПОЛЬЗУЕТСЯ!
            # Все данные ботов уже есть в нормализованных таблицах:
            # - bots_data.db → bots (текущее состояние ботов)
            # - bots_data.db → rsi_cache_coins (RSI данные)
            # Снапшоты - это избыточное дублирование данных!
            # Таблица будет удалена при миграции (см. ниже)
            
            # ==================== ТАБЛИЦА: АНАЛИЗ СТРАТЕГИЙ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    symbol TEXT,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для strategy_analysis
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_analysis_type ON strategy_analysis(analysis_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_analysis_symbol ON strategy_analysis(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_analysis_created_at ON strategy_analysis(created_at)")
            
            # ==================== ТАБЛИЦА: ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimized_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    -- RSI параметры
                    rsi_long_threshold REAL,
                    rsi_short_threshold REAL,
                    rsi_exit_long_with_trend REAL,
                    rsi_exit_long_against_trend REAL,
                    rsi_exit_short_with_trend REAL,
                    rsi_exit_short_against_trend REAL,
                    -- Risk параметры
                    max_loss_percent REAL,
                    take_profit_percent REAL,
                    trailing_stop_activation REAL,
                    trailing_stop_distance REAL,
                    trailing_take_distance REAL,
                    trailing_update_interval REAL,
                    break_even_trigger REAL,
                    break_even_protection REAL,
                    max_position_hours REAL,
                    -- Дополнительные параметры
                    optimization_type TEXT,
                    win_rate REAL,
                    total_pnl REAL,
                    params_json TEXT,
                    extra_params_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для optimized_params
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimized_params_symbol ON optimized_params(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimized_params_type ON optimized_params(optimization_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimized_params_created_at ON optimized_params(created_at)")
            
            # ==================== ТАБЛИЦА: СТАТУС СЕРВИСА ДАННЫХ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_service_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL UNIQUE,
                    last_collection TEXT,
                    trades_count INTEGER DEFAULT 0,
                    candles_count INTEGER DEFAULT 0,
                    ready INTEGER DEFAULT 0,
                    history_loaded INTEGER DEFAULT 0,
                    timestamp TEXT,
                    extra_status_json TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для data_service_status
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_service_name ON data_service_status(service_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_service_updated_at ON data_service_status(updated_at)")
            
            # ==================== ТАБЛИЦА: ПАТТЕРНЫ И ИНСАЙТЫ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    symbol TEXT,
                    rsi_range TEXT,
                    trend_condition TEXT,
                    volatility_range TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_pnl REAL,
                    avg_duration REAL,
                    pattern_data_json TEXT,
                    discovered_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
            """)
            
            # Индексы для trading_patterns
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON trading_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON trading_patterns(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_rsi_range ON trading_patterns(rsi_range)")
            
            # ==================== ТАБЛИЦА: РЕЗУЛЬТАТЫ БЭКТЕСТОВ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_name TEXT,
                    symbol TEXT,
                    -- Основные метрики
                    period_days INTEGER,
                    initial_balance REAL,
                    final_balance REAL,
                    total_return REAL,
                    total_pnl REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    -- Дополнительные данные
                    results_json TEXT,
                    extra_results_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Индексы для backtest_results
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON backtest_results(created_at)")
            
            # ==================== ТАБЛИЦА: БАЗА ЗНАНИЙ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_type TEXT NOT NULL,
                    knowledge_data_json TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    UNIQUE(knowledge_type)
                )
            """)
            
            # Индексы для knowledge_base
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_base(knowledge_type)")
            
            # ==================== ТАБЛИЦА: ДАННЫЕ ОБУЧЕНИЯ ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    symbol TEXT,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для training_data
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_type ON training_data(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_symbol ON training_data(symbol)")
            
            # ==================== ТАБЛИЦА: КОНФИГИ БОТОВ (НОРМАЛИЗОВАННАЯ) ====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
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
                    config_json TEXT,
                    extra_config_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Индексы для bot_configs
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_configs_symbol ON bot_configs(symbol)")
            
            conn.commit()
            
            pass
    
    def _migrate_schema(self, cursor, conn):
        """Миграция схемы БД: добавляет новые поля если их нет"""
        try:
            # ==================== МИГРАЦИЯ: data_service_status из JSON в нормализованные столбцы ====================
            # Проверяем, есть ли старая структура (с status_json)
            try:
                cursor.execute("SELECT status_json FROM data_service_status LIMIT 1")
                # Если запрос выполнился - значит старая структура
                logger.info("📦 Обнаружена старая JSON структура data_service_status, выполняю миграцию...")
                
                # Загружаем все данные из старой структуры
                cursor.execute("SELECT service_name, status_json, updated_at FROM data_service_status")
                old_rows = cursor.fetchall()
                
                if old_rows:
                    # Добавляем новые колонки если их нет
                    try:
                        cursor.execute("SELECT last_collection FROM data_service_status LIMIT 1")
                    except sqlite3.OperationalError:
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN last_collection TEXT")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN trades_count INTEGER DEFAULT 0")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN candles_count INTEGER DEFAULT 0")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN ready INTEGER DEFAULT 0")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN history_loaded INTEGER DEFAULT 0")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN timestamp TEXT")
                        cursor.execute("ALTER TABLE data_service_status ADD COLUMN extra_status_json TEXT")
                    
                    # Мигрируем данные
                    for old_row in old_rows:
                        service_name = old_row['service_name']
                        status_json = old_row['status_json']
                        updated_at = old_row['updated_at']
                        
                        try:
                            status = json.loads(status_json) if status_json else {}
                            
                            # Извлекаем основные поля
                            last_collection = status.get('last_collection')
                            trades_count = status.get('trades', 0)
                            candles_count = status.get('candles', 0)
                            ready = 1 if status.get('ready', False) else 0
                            history_loaded = 1 if status.get('history_loaded', False) else 0
                            timestamp = status.get('timestamp')
                            
                            # Собираем остальные поля в extra_status_json
                            extra_status = {}
                            known_fields = {
                                'last_collection', 'trades', 'candles', 'ready', 
                                'history_loaded', 'timestamp'
                            }
                            for key, value in status.items():
                                if key not in known_fields:
                                    extra_status[key] = value
                            
                            extra_status_json = json.dumps(extra_status, ensure_ascii=False) if extra_status else None
                            
                            # Обновляем запись
                            cursor.execute("""
                                UPDATE data_service_status 
                                SET last_collection = ?, trades_count = ?, candles_count = ?,
                                    ready = ?, history_loaded = ?, timestamp = ?, extra_status_json = ?
                                WHERE service_name = ?
                            """, (
                                last_collection, trades_count, candles_count,
                                ready, history_loaded, timestamp, extra_status_json,
                                service_name
                            ))
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка миграции статуса для {service_name}: {e}")
                            continue
                    
                    logger.info("✅ Миграция data_service_status завершена: данные перенесены из JSON в нормализованные столбцы")
                    
                    # Удаляем старую колонку status_json (SQLite не поддерживает DROP COLUMN, пересоздаем таблицу)
                    try:
                        # Создаем временную таблицу с новой структурой
                        cursor.execute("""
                            CREATE TABLE data_service_status_new (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                service_name TEXT NOT NULL UNIQUE,
                                last_collection TEXT,
                                trades_count INTEGER DEFAULT 0,
                                candles_count INTEGER DEFAULT 0,
                                ready INTEGER DEFAULT 0,
                                history_loaded INTEGER DEFAULT 0,
                                timestamp TEXT,
                                extra_status_json TEXT,
                                updated_at TEXT NOT NULL
                            )
                        """)
                        
                        # Копируем данные из старой таблицы в новую
                        cursor.execute("""
                            INSERT INTO data_service_status_new (
                                id, service_name, last_collection, trades_count, candles_count,
                                ready, history_loaded, timestamp, extra_status_json, updated_at
                            )
                            SELECT 
                                id, service_name, last_collection, trades_count, candles_count,
                                ready, history_loaded, timestamp, extra_status_json, updated_at
                            FROM data_service_status
                        """)
                        
                        # Удаляем старую таблицу
                        cursor.execute("DROP TABLE data_service_status")
                        
                        # Переименовываем новую таблицу
                        cursor.execute("ALTER TABLE data_service_status_new RENAME TO data_service_status")
                        
                        # Восстанавливаем индексы
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_service_name ON data_service_status(service_name)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_service_updated_at ON data_service_status(updated_at)")
                        
                        conn.commit()
                        logger.info("✅ Колонка status_json удалена из data_service_status")
                    except Exception as drop_error:
                        logger.warning(f"⚠️ Не удалось удалить колонку status_json: {drop_error}")
                        # Продолжаем работу - данные уже мигрированы
                    
            except sqlite3.OperationalError:
                # Колонка status_json не существует - значит уже мигрировано или новая структура
                pass
            except Exception as e:
                logger.warning(f"⚠️ Ошибка миграции data_service_status: {e}")
            
            # ==================== МИГРАЦИЯ: Удаление таблицы bots_data_snapshots ====================
            # ВАЖНО: Снапшоты больше не нужны - данные уже в нормализованных таблицах!
            # - bots_data.db → bots (текущее состояние ботов)
            # - bots_data.db → rsi_cache_coins (RSI данные)
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bots_data_snapshots'")
                if cursor.fetchone():
                    # Таблица существует - удаляем её
                    cursor.execute("DROP TABLE IF EXISTS bots_data_snapshots")
                    logger.info("🗑️ Таблица bots_data_snapshots удалена (снапшоты больше не используются - данные в нормализованных таблицах)")
            except Exception as e:
                pass
            # Проверяем и добавляем entry_volatility и entry_volume_ratio в simulated_trades
            try:
                cursor.execute("SELECT entry_volatility FROM simulated_trades LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📦 Миграция: добавляем entry_volatility и entry_volume_ratio в simulated_trades")
                cursor.execute("ALTER TABLE simulated_trades ADD COLUMN entry_volatility REAL")
                cursor.execute("ALTER TABLE simulated_trades ADD COLUMN entry_volume_ratio REAL")
            
            # Проверяем и добавляем entry_volatility и entry_volume_ratio в bot_trades
            try:
                cursor.execute("SELECT entry_volatility FROM bot_trades LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📦 Миграция: добавляем entry_volatility и entry_volume_ratio в bot_trades")
                cursor.execute("ALTER TABLE bot_trades ADD COLUMN entry_volatility REAL")
                cursor.execute("ALTER TABLE bot_trades ADD COLUMN entry_volume_ratio REAL")
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для simulated_trades ====================
            # Добавляем новые нормализованные столбцы если их нет
            rsi_fields = [
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('rsi_exit_long_with_trend', 'REAL'),
                ('rsi_exit_long_against_trend', 'REAL'),
                ('rsi_exit_short_with_trend', 'REAL'),
                ('rsi_exit_short_against_trend', 'REAL')
            ]
            risk_fields = [
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('trailing_stop_activation', 'REAL'),
                ('trailing_stop_distance', 'REAL'),
                ('trailing_take_distance', 'REAL'),
                ('trailing_update_interval', 'REAL'),
                ('break_even_trigger', 'REAL'),
                ('break_even_protection', 'REAL'),
                ('max_position_hours', 'REAL')
            ]
            extra_fields = [('extra_params_json', 'TEXT')]
            
            all_new_fields = rsi_fields + risk_fields + extra_fields
            for field_name, field_type in all_new_fields:
                try:
                    cursor.execute(f"SELECT {field_name} FROM simulated_trades LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в simulated_trades")
                    cursor.execute(f"ALTER TABLE simulated_trades ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                # Проверяем, существует ли колонка rsi_params_json
                cursor.execute("PRAGMA table_info(simulated_trades)")
                columns = [row[1] for row in cursor.fetchall()]
                has_rsi_json = 'rsi_params_json' in columns
                has_risk_json = 'risk_params_json' in columns
                
                if has_rsi_json or has_risk_json:
                    cursor.execute(f"SELECT id, {', rsi_params_json' if has_rsi_json else ''}{', risk_params_json' if has_risk_json else ''} FROM simulated_trades WHERE {'rsi_params_json IS NOT NULL OR ' if has_rsi_json else ''}{'risk_params_json IS NOT NULL' if has_risk_json else '1=0'} LIMIT 1")
                    if cursor.fetchone():
                        logger.info("📦 Обнаружены JSON данные в simulated_trades, выполняю миграцию в нормализованные столбцы...")
                        
                        query = f"SELECT id{', rsi_params_json' if has_rsi_json else ''}{', risk_params_json' if has_risk_json else ''} FROM simulated_trades"
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        
                        migrated_count = 0
                        for row in rows:
                            try:
                                trade_id = row[0]
                                rsi_params_json = row[1] if has_rsi_json else None
                                risk_params_json = row[2] if has_risk_json else (row[1] if has_rsi_json and has_risk_json else None)
                                
                                # Парсим JSON
                                rsi_params = json.loads(rsi_params_json) if rsi_params_json else {}
                                risk_params = json.loads(risk_params_json) if risk_params_json else {}
                                
                                # Извлекаем значения (поддерживаем оба формата ключей)
                                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                                
                                max_loss = risk_params.get('max_loss_percent')
                                take_profit = risk_params.get('take_profit_percent')
                                trailing_activation = risk_params.get('trailing_stop_activation')
                                trailing_distance = risk_params.get('trailing_stop_distance')
                                trailing_take = risk_params.get('trailing_take_distance')
                                trailing_interval = risk_params.get('trailing_update_interval')
                                break_even_trigger = risk_params.get('break_even_trigger')
                                break_even_protection = risk_params.get('break_even_protection')
                                max_hours = risk_params.get('max_position_hours')
                                
                                # Обновляем запись
                                cursor.execute("""
                                    UPDATE simulated_trades SET
                                        rsi_long_threshold = ?,
                                        rsi_short_threshold = ?,
                                        rsi_exit_long_with_trend = ?,
                                        rsi_exit_long_against_trend = ?,
                                        rsi_exit_short_with_trend = ?,
                                        rsi_exit_short_against_trend = ?,
                                        max_loss_percent = ?,
                                        take_profit_percent = ?,
                                        trailing_stop_activation = ?,
                                        trailing_stop_distance = ?,
                                        trailing_take_distance = ?,
                                        trailing_update_interval = ?,
                                        break_even_trigger = ?,
                                        break_even_protection = ?,
                                        max_position_hours = ?
                                    WHERE id = ?
                                """, (
                                    rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                                    rsi_exit_short_with, rsi_exit_short_against,
                                    max_loss, take_profit, trailing_activation, trailing_distance,
                                    trailing_take, trailing_interval, break_even_trigger,
                                    break_even_protection, max_hours, trade_id
                                ))
                                migrated_count += 1
                            except Exception as e:
                                pass
                                continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция simulated_trades завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            # Проверяем и добавляем параметры конфига в simulated_trades (оставляем для обратной совместимости)
            new_fields_sim = [
                ('config_params_json', 'TEXT'),
                ('filters_params_json', 'TEXT'),
                ('entry_conditions_json', 'TEXT'),
                ('exit_conditions_json', 'TEXT'),
                ('restrictions_json', 'TEXT')
            ]
            for field_name, field_type in new_fields_sim:
                try:
                    cursor.execute(f"SELECT {field_name} FROM simulated_trades LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в simulated_trades")
                    cursor.execute(f"ALTER TABLE simulated_trades ADD COLUMN {field_name} {field_type}")
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для bot_trades ====================
            # Добавляем новые нормализованные столбцы если их нет
            rsi_fields_bot = [
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('rsi_exit_long_with_trend', 'REAL'),
                ('rsi_exit_long_against_trend', 'REAL'),
                ('rsi_exit_short_with_trend', 'REAL'),
                ('rsi_exit_short_against_trend', 'REAL')
            ]
            risk_fields_bot = [
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('trailing_stop_activation', 'REAL'),
                ('trailing_stop_distance', 'REAL'),
                ('trailing_take_distance', 'REAL'),
                ('trailing_update_interval', 'REAL'),
                ('break_even_trigger', 'REAL'),
                ('break_even_protection', 'REAL'),
                ('max_position_hours', 'REAL')
            ]
            extra_fields_bot = [('extra_config_json', 'TEXT')]
            
            all_new_fields_bot = rsi_fields_bot + risk_fields_bot + extra_fields_bot
            for field_name, field_type in all_new_fields_bot:
                try:
                    cursor.execute(f"SELECT {field_name} FROM bot_trades LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в bot_trades")
                    cursor.execute(f"ALTER TABLE bot_trades ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                # Проверяем, существует ли колонка config_params_json
                cursor.execute("PRAGMA table_info(bot_trades)")
                columns = [row[1] for row in cursor.fetchall()]
                has_config_json = 'config_params_json' in columns
                
                if has_config_json:
                    cursor.execute("SELECT id, config_params_json FROM bot_trades WHERE config_params_json IS NOT NULL LIMIT 1")
                    if cursor.fetchone():
                        logger.info("📦 Обнаружены JSON данные в bot_trades, выполняю миграцию в нормализованные столбцы...")
                        
                        cursor.execute("SELECT id, config_params_json FROM bot_trades WHERE config_params_json IS NOT NULL")
                        rows = cursor.fetchall()
                        
                        migrated_count = 0
                        for row in rows:
                            try:
                                trade_id = row[0]
                                config_params_json = row[1]
                                
                                # Парсим JSON
                                config_params = json.loads(config_params_json) if config_params_json else {}
                                
                                # Извлекаем RSI параметры (поддерживаем оба формата)
                                rsi_params = config_params.get('rsi_params', {}) if isinstance(config_params.get('rsi_params'), dict) else {}
                                if not rsi_params:
                                    # Пытаемся извлечь напрямую из config_params
                                    rsi_params = {k: v for k, v in config_params.items() if 'rsi' in k.lower() or k in ['oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend']}
                                
                                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold') or config_params.get('rsi_long_threshold')
                                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold') or config_params.get('rsi_short_threshold')
                                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend') or config_params.get('rsi_exit_long_with_trend')
                                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend') or config_params.get('rsi_exit_long_against_trend')
                                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend') or config_params.get('rsi_exit_short_with_trend')
                                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend') or config_params.get('rsi_exit_short_against_trend')
                                
                                # Извлекаем Risk параметры
                                risk_params = config_params.get('risk_params', {}) if isinstance(config_params.get('risk_params'), dict) else {}
                                if not risk_params:
                                    # Пытаемся извлечь напрямую из config_params
                                    risk_params = {k: v for k, v in config_params.items() if k in ['max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours']}
                                
                                max_loss = risk_params.get('max_loss_percent') or config_params.get('max_loss_percent')
                                take_profit = risk_params.get('take_profit_percent') or config_params.get('take_profit_percent')
                                trailing_activation = risk_params.get('trailing_stop_activation') or config_params.get('trailing_stop_activation')
                                trailing_distance = risk_params.get('trailing_stop_distance') or config_params.get('trailing_stop_distance')
                                trailing_take = risk_params.get('trailing_take_distance') or config_params.get('trailing_take_distance')
                                trailing_interval = risk_params.get('trailing_update_interval') or config_params.get('trailing_update_interval')
                                break_even_trigger = risk_params.get('break_even_trigger') or config_params.get('break_even_trigger')
                                break_even_protection = risk_params.get('break_even_protection') or config_params.get('break_even_protection')
                                max_hours = risk_params.get('max_position_hours') or config_params.get('max_position_hours')
                                
                                # Собираем остальные параметры в extra_config_json
                                extra_config = {}
                                known_fields = {
                                    'rsi_params', 'risk_params', 'rsi_long_threshold', 'rsi_short_threshold',
                                    'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                                    'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                                    'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                                    'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                                    'break_even_trigger', 'break_even_protection', 'max_position_hours',
                                    'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                                    'exit_short_with_trend', 'exit_short_against_trend'
                                }
                                for key, value in config_params.items():
                                    if key not in known_fields:
                                        extra_config[key] = value
                                
                                extra_config_json = json.dumps(extra_config, ensure_ascii=False) if extra_config else None
                                
                                # Обновляем запись
                                cursor.execute("""
                                    UPDATE bot_trades SET
                                        rsi_long_threshold = ?,
                                        rsi_short_threshold = ?,
                                        rsi_exit_long_with_trend = ?,
                                        rsi_exit_long_against_trend = ?,
                                        rsi_exit_short_with_trend = ?,
                                        rsi_exit_short_against_trend = ?,
                                        max_loss_percent = ?,
                                        take_profit_percent = ?,
                                        trailing_stop_activation = ?,
                                        trailing_stop_distance = ?,
                                        trailing_take_distance = ?,
                                        trailing_update_interval = ?,
                                        break_even_trigger = ?,
                                        break_even_protection = ?,
                                        max_position_hours = ?,
                                        extra_config_json = ?
                                    WHERE id = ?
                                """, (
                                    rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                                    rsi_exit_short_with, rsi_exit_short_against,
                                    max_loss, take_profit, trailing_activation, trailing_distance,
                                    trailing_take, trailing_interval, break_even_trigger,
                                    break_even_protection, max_hours, extra_config_json, trade_id
                                ))
                                migrated_count += 1
                            except Exception as e:
                                pass
                                continue
                        
                        if migrated_count > 0:
                            logger.info(f"✅ Миграция bot_trades завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            # Проверяем и добавляем параметры конфига в bot_trades (оставляем для обратной совместимости)
            new_fields_bot = [
                ('filters_params_json', 'TEXT'),
                ('entry_conditions_json', 'TEXT'),
                ('exit_conditions_json', 'TEXT'),
                ('restrictions_json', 'TEXT')
            ]
            for field_name, field_type in new_fields_bot:
                try:
                    cursor.execute(f"SELECT {field_name} FROM bot_trades LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в bot_trades")
                    cursor.execute(f"ALTER TABLE bot_trades ADD COLUMN {field_name} {field_type}")
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для parameter_training_samples ====================
            rsi_fields_param_samples = [
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('rsi_exit_long_with_trend', 'REAL'),
                ('rsi_exit_long_against_trend', 'REAL'),
                ('rsi_exit_short_with_trend', 'REAL'),
                ('rsi_exit_short_against_trend', 'REAL')
            ]
            risk_fields_param_samples = [
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('trailing_stop_activation', 'REAL'),
                ('trailing_stop_distance', 'REAL'),
                ('trailing_take_distance', 'REAL'),
                ('trailing_update_interval', 'REAL'),
                ('break_even_trigger', 'REAL'),
                ('break_even_protection', 'REAL'),
                ('max_position_hours', 'REAL')
            ]
            extra_fields_param_samples = [('extra_rsi_params_json', 'TEXT'), ('extra_risk_params_json', 'TEXT')]
            
            all_new_fields_param_samples = rsi_fields_param_samples + risk_fields_param_samples + extra_fields_param_samples
            for field_name, field_type in all_new_fields_param_samples:
                try:
                    cursor.execute(f"SELECT {field_name} FROM parameter_training_samples LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в parameter_training_samples")
                    cursor.execute(f"ALTER TABLE parameter_training_samples ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы для parameter_training_samples
            try:
                # Проверяем, существует ли колонка rsi_params_json
                cursor.execute("PRAGMA table_info(parameter_training_samples)")
                columns = [row[1] for row in cursor.fetchall()]
                has_rsi_json = 'rsi_params_json' in columns
                has_risk_json = 'risk_params_json' in columns
                
                if has_rsi_json or has_risk_json:
                    cursor.execute(f"SELECT id{', rsi_params_json' if has_rsi_json else ''}{', risk_params_json' if has_risk_json else ''} FROM parameter_training_samples WHERE {'rsi_params_json IS NOT NULL' if has_rsi_json else '1=0'} LIMIT 1")
                    if cursor.fetchone():
                        logger.info("📦 Обнаружены JSON данные в parameter_training_samples, выполняю миграцию...")
                        
                        query = f"SELECT id{', rsi_params_json' if has_rsi_json else ''}{', risk_params_json' if has_risk_json else ''} FROM parameter_training_samples WHERE {'rsi_params_json IS NOT NULL' if has_rsi_json else '1=0'}"
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        
                        migrated_count = 0
                        for row in rows:
                            try:
                                sample_id = row[0]
                                rsi_params_json = row[1] if has_rsi_json else None
                                risk_params_json = row[2] if has_risk_json else (row[1] if has_rsi_json and has_risk_json else None)
                                
                                # Парсим RSI параметры
                                rsi_params = json.loads(rsi_params_json) if rsi_params_json else {}
                                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                                
                                # Собираем остальные RSI параметры
                                extra_rsi = {}
                                known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                                for key, value in rsi_params.items():
                                    if key not in known_rsi:
                                        extra_rsi[key] = value
                                extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                                
                                # Парсим Risk параметры
                                risk_params = json.loads(risk_params_json) if risk_params_json else {}
                                max_loss = risk_params.get('max_loss_percent')
                                take_profit = risk_params.get('take_profit_percent')
                                trailing_activation = risk_params.get('trailing_stop_activation')
                                trailing_distance = risk_params.get('trailing_stop_distance')
                                trailing_take = risk_params.get('trailing_take_distance')
                                trailing_interval = risk_params.get('trailing_update_interval')
                                break_even_trigger = risk_params.get('break_even_trigger')
                                break_even_protection = risk_params.get('break_even_protection')
                                max_hours = risk_params.get('max_position_hours')
                                
                                # Собираем остальные Risk параметры
                                extra_risk = {}
                                known_risk = {'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours'}
                                for key, value in risk_params.items():
                                    if key not in known_risk:
                                        extra_risk[key] = value
                                extra_risk_json = json.dumps(extra_risk, ensure_ascii=False) if extra_risk else None
                                
                                # Обновляем запись
                                cursor.execute("""
                                UPDATE parameter_training_samples SET
                                    rsi_long_threshold = ?,
                                    rsi_short_threshold = ?,
                                    rsi_exit_long_with_trend = ?,
                                    rsi_exit_long_against_trend = ?,
                                    rsi_exit_short_with_trend = ?,
                                    rsi_exit_short_against_trend = ?,
                                    max_loss_percent = ?,
                                    take_profit_percent = ?,
                                    trailing_stop_activation = ?,
                                    trailing_stop_distance = ?,
                                    trailing_take_distance = ?,
                                    trailing_update_interval = ?,
                                    break_even_trigger = ?,
                                    break_even_protection = ?,
                                    max_position_hours = ?,
                                    extra_rsi_params_json = ?,
                                    extra_risk_params_json = ?
                                WHERE id = ?
                            """, (
                                rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                                rsi_exit_short_with, rsi_exit_short_against,
                                max_loss, take_profit, trailing_activation, trailing_distance,
                                trailing_take, trailing_interval, break_even_trigger,
                                break_even_protection, max_hours, extra_rsi_json, extra_risk_json, sample_id
                            ))
                                migrated_count += 1
                            except Exception as e:
                                pass
                                continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция parameter_training_samples завершена: {migrated_count} записей мигрировано")
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для used_training_parameters, best_params_per_symbol, blocked_params ====================
            rsi_fields_common = [
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('rsi_exit_long_with_trend', 'REAL'),
                ('rsi_exit_long_against_trend', 'REAL'),
                ('rsi_exit_short_with_trend', 'REAL'),
                ('rsi_exit_short_against_trend', 'REAL'),
                ('extra_rsi_params_json', 'TEXT')
            ]
            
            for table_name in ['used_training_parameters', 'best_params_per_symbol', 'blocked_params']:
                for field_name, field_type in rsi_fields_common:
                    try:
                        cursor.execute(f"SELECT {field_name} FROM {table_name} LIMIT 1")
                    except sqlite3.OperationalError:
                        logger.info(f"📦 Миграция: добавляем {field_name} в {table_name}")
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {field_name} {field_type}")
                
                # Мигрируем данные из JSON в столбцы
                try:
                    # Проверяем, существует ли колонка rsi_params_json
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in cursor.fetchall()]
                    has_rsi_json = 'rsi_params_json' in columns
                    
                    if has_rsi_json:
                        cursor.execute(f"SELECT id, rsi_params_json FROM {table_name} WHERE rsi_params_json IS NOT NULL LIMIT 1")
                        if cursor.fetchone():
                            logger.info(f"📦 Обнаружены JSON данные в {table_name}, выполняю миграцию...")
                            
                            cursor.execute(f"SELECT id, rsi_params_json FROM {table_name} WHERE rsi_params_json IS NOT NULL")
                            rows = cursor.fetchall()
                            
                            migrated_count = 0
                            for row in rows:
                                try:
                                    record_id = row[0]
                                    rsi_params_json = row[1]
                                    
                                    # Парсим RSI параметры
                                    rsi_params = json.loads(rsi_params_json) if rsi_params_json else {}
                                    rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                                    rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                                    rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                                    rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                                    rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                                    rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                                    
                                    # Собираем остальные RSI параметры
                                    extra_rsi = {}
                                    known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                                    for key, value in rsi_params.items():
                                        if key not in known_rsi:
                                            extra_rsi[key] = value
                                    extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                                    
                                    # Обновляем запись
                                    cursor.execute(f"""
                                        UPDATE {table_name} SET
                                            rsi_long_threshold = ?,
                                            rsi_short_threshold = ?,
                                            rsi_exit_long_with_trend = ?,
                                            rsi_exit_long_against_trend = ?,
                                            rsi_exit_short_with_trend = ?,
                                            rsi_exit_short_against_trend = ?,
                                            extra_rsi_params_json = ?
                                        WHERE id = ?
                                    """, (
                                        rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                                        rsi_exit_short_with, rsi_exit_short_against, extra_rsi_json, record_id
                                    ))
                                    migrated_count += 1
                                except Exception as e:
                                    pass
                                    continue
                            
                            if migrated_count > 0:
                                logger.info(f"✅ Миграция {table_name} завершена: {migrated_count} записей мигрировано")
                except Exception as e:
                    pass
            
            # Проверяем и добавляем поля в blocked_params
            new_fields_blocked = [
                ('param_hash', 'TEXT'),
                ('blocked_attempts', 'INTEGER DEFAULT 0'),
                ('blocked_long', 'INTEGER DEFAULT 0'),
                ('blocked_short', 'INTEGER DEFAULT 0')
            ]
            for field_name, field_type in new_fields_blocked:
                try:
                    cursor.execute(f"SELECT {field_name} FROM blocked_params LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в bot_trades")
                    cursor.execute(f"ALTER TABLE bot_trades ADD COLUMN {field_name} {field_type}")
            
            # Проверяем и добавляем поля в blocked_params
            new_fields_blocked = [
                ('param_hash', 'TEXT'),
                ('blocked_attempts', 'INTEGER DEFAULT 0'),
                ('blocked_long', 'INTEGER DEFAULT 0'),
                ('blocked_short', 'INTEGER DEFAULT 0')
            ]
            for field_name, field_type in new_fields_blocked:
                try:
                    cursor.execute(f"SELECT {field_name} FROM blocked_params LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в blocked_params")
                    cursor.execute(f"ALTER TABLE blocked_params ADD COLUMN {field_name} {field_type}")
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для optimized_params ====================
            # Добавляем новые нормализованные столбцы если их нет
            optimized_params_fields = [
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('rsi_exit_long_with_trend', 'REAL'),
                ('rsi_exit_long_against_trend', 'REAL'),
                ('rsi_exit_short_with_trend', 'REAL'),
                ('rsi_exit_short_against_trend', 'REAL'),
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('trailing_stop_activation', 'REAL'),
                ('trailing_stop_distance', 'REAL'),
                ('trailing_take_distance', 'REAL'),
                ('trailing_update_interval', 'REAL'),
                ('break_even_trigger', 'REAL'),
                ('break_even_protection', 'REAL'),
                ('max_position_hours', 'REAL'),
                ('extra_params_json', 'TEXT')
            ]
            for field_name, field_type in optimized_params_fields:
                try:
                    cursor.execute(f"SELECT {field_name} FROM optimized_params LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в optimized_params")
                    cursor.execute(f"ALTER TABLE optimized_params ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                cursor.execute("SELECT id, params_json FROM optimized_params WHERE params_json IS NOT NULL AND rsi_long_threshold IS NULL LIMIT 1")
                if cursor.fetchone():
                    logger.info("📦 Обнаружены JSON данные в optimized_params, выполняю миграцию в нормализованные столбцы...")
                    
                    cursor.execute("SELECT id, params_json FROM optimized_params WHERE params_json IS NOT NULL")
                    rows = cursor.fetchall()
                    
                    migrated_count = 0
                    for row in rows:
                        try:
                            param_id = row[0]
                            params_json = row[1]
                            
                            # Парсим params
                            params = {}
                            if params_json:
                                try:
                                    params = json.loads(params_json)
                                except:
                                    params = {}
                            
                            # Извлекаем RSI параметры
                            rsi_params = params.get('rsi_params', {}) if isinstance(params.get('rsi_params'), dict) else {}
                            if not rsi_params:
                                rsi_params = {k: v for k, v in params.items() if 'rsi' in k.lower() or k in ['oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend']}
                            
                            rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold') or params.get('rsi_long_threshold')
                            rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold') or params.get('rsi_short_threshold')
                            rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend') or params.get('rsi_exit_long_with_trend')
                            rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend') or params.get('rsi_exit_long_against_trend')
                            rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend') or params.get('rsi_exit_short_with_trend')
                            rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend') or params.get('rsi_exit_short_against_trend')
                            
                            # Извлекаем Risk параметры
                            risk_params = params.get('risk_params', {}) if isinstance(params.get('risk_params'), dict) else {}
                            if not risk_params:
                                risk_params = {k: v for k, v in params.items() if k in ['max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours']}
                            
                            max_loss = risk_params.get('max_loss_percent') or params.get('max_loss_percent')
                            take_profit = risk_params.get('take_profit_percent') or params.get('take_profit_percent')
                            trailing_activation = risk_params.get('trailing_stop_activation') or params.get('trailing_stop_activation')
                            trailing_distance = risk_params.get('trailing_stop_distance') or params.get('trailing_stop_distance')
                            trailing_take = risk_params.get('trailing_take_distance') or params.get('trailing_take_distance')
                            trailing_interval = risk_params.get('trailing_update_interval') or params.get('trailing_update_interval')
                            break_even_trigger = risk_params.get('break_even_trigger') or params.get('break_even_trigger')
                            break_even_protection = risk_params.get('break_even_protection') or params.get('break_even_protection')
                            max_hours = risk_params.get('max_position_hours') or params.get('max_position_hours')
                            
                            # Собираем остальные параметры в extra_params_json
                            extra_params = {}
                            known_fields = {
                                'rsi_params', 'risk_params', 'rsi_long_threshold', 'rsi_short_threshold',
                                'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                                'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                                'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                                'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                                'break_even_trigger', 'break_even_protection', 'max_position_hours',
                                'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                                'exit_short_with_trend', 'exit_short_against_trend', 'win_rate', 'total_pnl'
                            }
                            for key, value in params.items():
                                if key not in known_fields:
                                    extra_params[key] = value
                            
                            extra_params_json = json.dumps(extra_params, ensure_ascii=False) if extra_params else None
                            
                            # Обновляем запись
                            cursor.execute("""
                                UPDATE optimized_params SET
                                    rsi_long_threshold = COALESCE(rsi_long_threshold, ?),
                                    rsi_short_threshold = COALESCE(rsi_short_threshold, ?),
                                    rsi_exit_long_with_trend = COALESCE(rsi_exit_long_with_trend, ?),
                                    rsi_exit_long_against_trend = COALESCE(rsi_exit_long_against_trend, ?),
                                    rsi_exit_short_with_trend = COALESCE(rsi_exit_short_with_trend, ?),
                                    rsi_exit_short_against_trend = COALESCE(rsi_exit_short_against_trend, ?),
                                    max_loss_percent = COALESCE(max_loss_percent, ?),
                                    take_profit_percent = COALESCE(take_profit_percent, ?),
                                    trailing_stop_activation = COALESCE(trailing_stop_activation, ?),
                                    trailing_stop_distance = COALESCE(trailing_stop_distance, ?),
                                    trailing_take_distance = COALESCE(trailing_take_distance, ?),
                                    trailing_update_interval = COALESCE(trailing_update_interval, ?),
                                    break_even_trigger = COALESCE(break_even_trigger, ?),
                                    break_even_protection = COALESCE(break_even_protection, ?),
                                    max_position_hours = COALESCE(max_position_hours, ?),
                                    extra_params_json = COALESCE(extra_params_json, ?)
                                WHERE id = ? AND rsi_long_threshold IS NULL
                            """, (
                                rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                                rsi_exit_short_with, rsi_exit_short_against,
                                max_loss, take_profit, trailing_activation, trailing_distance,
                                trailing_take, trailing_interval, break_even_trigger,
                                break_even_protection, max_hours, extra_params_json,
                                param_id
                            ))
                            if cursor.rowcount > 0:
                                migrated_count += 1
                        except Exception as e:
                            pass
                            continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция optimized_params завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для backtest_results ====================
            # Добавляем новые нормализованные столбцы если их нет
            backtest_fields = [
                ('period_days', 'INTEGER'),
                ('initial_balance', 'REAL'),
                ('final_balance', 'REAL'),
                ('total_pnl', 'REAL'),
                ('winning_trades', 'INTEGER'),
                ('losing_trades', 'INTEGER'),
                ('avg_win', 'REAL'),
                ('avg_loss', 'REAL'),
                ('profit_factor', 'REAL'),
                ('extra_results_json', 'TEXT')
            ]
            for field_name, field_type in backtest_fields:
                try:
                    cursor.execute(f"SELECT {field_name} FROM backtest_results LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в backtest_results")
                    cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                cursor.execute("SELECT id, results_json FROM backtest_results WHERE results_json IS NOT NULL AND period_days IS NULL LIMIT 1")
                if cursor.fetchone():
                    logger.info("📦 Обнаружены JSON данные в backtest_results, выполняю миграцию в нормализованные столбцы...")
                    
                    cursor.execute("SELECT id, results_json FROM backtest_results WHERE results_json IS NOT NULL")
                    rows = cursor.fetchall()
                    
                    migrated_count = 0
                    for row in rows:
                        try:
                            result_id = row[0]
                            results_json = row[1]
                            
                            # Парсим results
                            results = {}
                            if results_json:
                                try:
                                    results = json.loads(results_json)
                                except:
                                    results = {}
                            
                            period_days = results.get('period_days')
                            initial_balance = results.get('initial_balance')
                            final_balance = results.get('final_balance')
                            total_pnl = results.get('total_pnl')
                            winning_trades = results.get('winning_trades')
                            losing_trades = results.get('losing_trades')
                            avg_win = results.get('avg_win')
                            avg_loss = results.get('avg_loss')
                            profit_factor = results.get('profit_factor')
                            
                            # Собираем остальные поля в extra_results_json
                            extra_results = {}
                            known_fields = {
                                'period_days', 'initial_balance', 'final_balance', 'total_return',
                                'total_pnl', 'total_trades', 'winning_trades', 'losing_trades',
                                'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'timestamp'
                            }
                            for key, value in results.items():
                                if key not in known_fields:
                                    extra_results[key] = value
                            
                            extra_results_json = json.dumps(extra_results, ensure_ascii=False) if extra_results else None
                            
                            # Обновляем запись
                            cursor.execute("""
                                UPDATE backtest_results SET
                                    period_days = COALESCE(period_days, ?),
                                    initial_balance = COALESCE(initial_balance, ?),
                                    final_balance = COALESCE(final_balance, ?),
                                    total_pnl = COALESCE(total_pnl, ?),
                                    winning_trades = COALESCE(winning_trades, ?),
                                    losing_trades = COALESCE(losing_trades, ?),
                                    avg_win = COALESCE(avg_win, ?),
                                    avg_loss = COALESCE(avg_loss, ?),
                                    profit_factor = COALESCE(profit_factor, ?),
                                    extra_results_json = COALESCE(extra_results_json, ?)
                                WHERE id = ? AND period_days IS NULL
                            """, (
                                period_days, initial_balance, final_balance, total_pnl,
                                winning_trades, losing_trades, avg_win, avg_loss, profit_factor,
                                extra_results_json, result_id
                            ))
                            if cursor.rowcount > 0:
                                migrated_count += 1
                        except Exception as e:
                            pass
                            continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция backtest_results завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для ai_decisions ====================
            # Добавляем новые нормализованные столбцы если их нет
            ai_decisions_fields = [
                ('volume', 'REAL'),
                ('volatility', 'REAL'),
                ('volume_ratio', 'REAL'),
                ('rsi_long_threshold', 'REAL'),
                ('rsi_short_threshold', 'REAL'),
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('extra_market_data_json', 'TEXT'),
                ('extra_decision_params_json', 'TEXT')
            ]
            for field_name, field_type in ai_decisions_fields:
                try:
                    cursor.execute(f"SELECT {field_name} FROM ai_decisions LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в ai_decisions")
                    cursor.execute(f"ALTER TABLE ai_decisions ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                cursor.execute("SELECT id, market_data_json, decision_params_json FROM ai_decisions WHERE (market_data_json IS NOT NULL OR decision_params_json IS NOT NULL) AND (volume IS NULL OR rsi_long_threshold IS NULL) LIMIT 1")
                if cursor.fetchone():
                    logger.info("📦 Обнаружены JSON данные в ai_decisions, выполняю миграцию в нормализованные столбцы...")
                    
                    cursor.execute("SELECT id, market_data_json, decision_params_json FROM ai_decisions WHERE market_data_json IS NOT NULL OR decision_params_json IS NOT NULL")
                    rows = cursor.fetchall()
                    
                    migrated_count = 0
                    for row in rows:
                        try:
                            decision_id = row[0]
                            market_data_json = row[1]
                            decision_params_json = row[2] if len(row) > 2 else None
                            
                            # Парсим market_data
                            market_data = {}
                            if market_data_json:
                                try:
                                    market_data = json.loads(market_data_json)
                                except:
                                    market_data = {}
                            
                            volume = market_data.get('volume') if isinstance(market_data, dict) else None
                            volatility = market_data.get('volatility') if isinstance(market_data, dict) else None
                            volume_ratio = market_data.get('volume_ratio') if isinstance(market_data, dict) else None
                            
                            # Собираем остальные поля market_data в extra_market_data_json
                            extra_market_data = {}
                            if isinstance(market_data, dict):
                                known_market_keys = {'volume', 'volatility', 'volume_ratio', 'rsi', 'trend', 'price', 'signal', 'confidence'}
                                for key, value in market_data.items():
                                    if key not in known_market_keys:
                                        extra_market_data[key] = value
                            
                            extra_market_data_json = json.dumps(extra_market_data, ensure_ascii=False) if extra_market_data else None
                            
                            # Парсим decision_params
                            decision_params = {}
                            if decision_params_json:
                                try:
                                    decision_params = json.loads(decision_params_json)
                                except:
                                    decision_params = {}
                            
                            rsi_long_threshold = decision_params.get('rsi_long_threshold') if isinstance(decision_params, dict) else None
                            rsi_short_threshold = decision_params.get('rsi_short_threshold') if isinstance(decision_params, dict) else None
                            max_loss_percent = decision_params.get('max_loss_percent') if isinstance(decision_params, dict) else None
                            take_profit_percent = decision_params.get('take_profit_percent') if isinstance(decision_params, dict) else None
                            
                            # Собираем остальные поля decision_params в extra_decision_params_json
                            extra_decision_params = {}
                            if isinstance(decision_params, dict):
                                known_params_keys = {'rsi_long_threshold', 'rsi_short_threshold', 'max_loss_percent', 'take_profit_percent'}
                                for key, value in decision_params.items():
                                    if key not in known_params_keys:
                                        extra_decision_params[key] = value
                            
                            extra_decision_params_json = json.dumps(extra_decision_params, ensure_ascii=False) if extra_decision_params else None
                            
                            # Обновляем запись только если поля еще не заполнены
                            cursor.execute("""
                                UPDATE ai_decisions SET
                                    volume = COALESCE(volume, ?),
                                    volatility = COALESCE(volatility, ?),
                                    volume_ratio = COALESCE(volume_ratio, ?),
                                    rsi_long_threshold = COALESCE(rsi_long_threshold, ?),
                                    rsi_short_threshold = COALESCE(rsi_short_threshold, ?),
                                    max_loss_percent = COALESCE(max_loss_percent, ?),
                                    take_profit_percent = COALESCE(take_profit_percent, ?),
                                    extra_market_data_json = COALESCE(extra_market_data_json, ?),
                                    extra_decision_params_json = COALESCE(extra_decision_params_json, ?)
                                WHERE id = ? AND (volume IS NULL OR rsi_long_threshold IS NULL)
                            """, (
                                volume, volatility, volume_ratio,
                                rsi_long_threshold, rsi_short_threshold,
                                max_loss_percent, take_profit_percent,
                                extra_market_data_json, extra_decision_params_json,
                                decision_id
                            ))
                            if cursor.rowcount > 0:
                                migrated_count += 1
                        except Exception as e:
                            pass
                            continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция ai_decisions завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            # ==================== МИГРАЦИЯ: Нормализация JSON параметров в столбцы для bot_configs ====================
            # Добавляем новые нормализованные столбцы если их нет
            bot_configs_fields = [
                ('rsi_long_threshold', 'INTEGER'),
                ('rsi_short_threshold', 'INTEGER'),
                ('rsi_exit_long_with_trend', 'INTEGER'),
                ('rsi_exit_long_against_trend', 'INTEGER'),
                ('rsi_exit_short_with_trend', 'INTEGER'),
                ('rsi_exit_short_against_trend', 'INTEGER'),
                ('max_loss_percent', 'REAL'),
                ('take_profit_percent', 'REAL'),
                ('trailing_stop_activation', 'REAL'),
                ('trailing_stop_distance', 'REAL'),
                ('trailing_take_distance', 'REAL'),
                ('trailing_update_interval', 'REAL'),
                ('break_even_trigger', 'REAL'),
                ('break_even_protection', 'REAL'),
                ('max_position_hours', 'REAL'),
                ('rsi_time_filter_enabled', 'INTEGER DEFAULT 0'),
                ('rsi_time_filter_candles', 'INTEGER'),
                ('rsi_time_filter_upper', 'INTEGER'),
                ('rsi_time_filter_lower', 'INTEGER'),
                ('avoid_down_trend', 'INTEGER DEFAULT 0'),
                ('extra_config_json', 'TEXT')
            ]
            for field_name, field_type in bot_configs_fields:
                try:
                    cursor.execute(f"SELECT {field_name} FROM bot_configs LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📦 Миграция: добавляем {field_name} в bot_configs")
                    cursor.execute(f"ALTER TABLE bot_configs ADD COLUMN {field_name} {field_type}")
            
            # Мигрируем данные из JSON в столбцы (если есть старые данные)
            try:
                cursor.execute("SELECT id, symbol, config_json FROM bot_configs WHERE config_json IS NOT NULL AND rsi_long_threshold IS NULL LIMIT 1")
                if cursor.fetchone():
                    logger.info("📦 Обнаружены JSON данные в bot_configs, выполняю миграцию в нормализованные столбцы...")
                    
                    cursor.execute("SELECT id, symbol, config_json FROM bot_configs WHERE config_json IS NOT NULL")
                    rows = cursor.fetchall()
                    
                    migrated_count = 0
                    for row in rows:
                        try:
                            config_id = row[0]
                            symbol = row[1]
                            config_json = row[2]
                            
                            # Парсим config
                            config = {}
                            if config_json:
                                try:
                                    config = json.loads(config_json)
                                except:
                                    config = {}
                            
                            # Извлекаем поля
                            rsi_long_threshold = config.get('rsi_long_threshold')
                            rsi_short_threshold = config.get('rsi_short_threshold')
                            rsi_exit_long_with_trend = config.get('rsi_exit_long_with_trend')
                            rsi_exit_long_against_trend = config.get('rsi_exit_long_against_trend')
                            rsi_exit_short_with_trend = config.get('rsi_exit_short_with_trend')
                            rsi_exit_short_against_trend = config.get('rsi_exit_short_against_trend')
                            max_loss_percent = config.get('max_loss_percent')
                            take_profit_percent = config.get('take_profit_percent')
                            trailing_stop_activation = config.get('trailing_stop_activation')
                            trailing_stop_distance = config.get('trailing_stop_distance')
                            trailing_take_distance = config.get('trailing_take_distance')
                            trailing_update_interval = config.get('trailing_update_interval')
                            break_even_trigger = config.get('break_even_trigger')
                            break_even_protection = config.get('break_even_protection')
                            max_position_hours = config.get('max_position_hours')
                            rsi_time_filter_enabled = 1 if config.get('rsi_time_filter_enabled') else 0
                            rsi_time_filter_candles = config.get('rsi_time_filter_candles')
                            rsi_time_filter_upper = config.get('rsi_time_filter_upper')
                            rsi_time_filter_lower = config.get('rsi_time_filter_lower')
                            avoid_down_trend = 1 if config.get('avoid_down_trend') else 0
                            
                            # Собираем остальные поля в extra_config_json
                            extra_config = {}
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
                            for key, value in config.items():
                                if key not in known_fields:
                                    extra_config[key] = value
                            
                            extra_config_json = json.dumps(extra_config, ensure_ascii=False) if extra_config else None
                            
                            # Обновляем запись
                            cursor.execute("""
                                UPDATE bot_configs SET
                                    rsi_long_threshold = COALESCE(rsi_long_threshold, ?),
                                    rsi_short_threshold = COALESCE(rsi_short_threshold, ?),
                                    rsi_exit_long_with_trend = COALESCE(rsi_exit_long_with_trend, ?),
                                    rsi_exit_long_against_trend = COALESCE(rsi_exit_long_against_trend, ?),
                                    rsi_exit_short_with_trend = COALESCE(rsi_exit_short_with_trend, ?),
                                    rsi_exit_short_against_trend = COALESCE(rsi_exit_short_against_trend, ?),
                                    max_loss_percent = COALESCE(max_loss_percent, ?),
                                    take_profit_percent = COALESCE(take_profit_percent, ?),
                                    trailing_stop_activation = COALESCE(trailing_stop_activation, ?),
                                    trailing_stop_distance = COALESCE(trailing_stop_distance, ?),
                                    trailing_take_distance = COALESCE(trailing_take_distance, ?),
                                    trailing_update_interval = COALESCE(trailing_update_interval, ?),
                                    break_even_trigger = COALESCE(break_even_trigger, ?),
                                    break_even_protection = COALESCE(break_even_protection, ?),
                                    max_position_hours = COALESCE(max_position_hours, ?),
                                    rsi_time_filter_enabled = COALESCE(rsi_time_filter_enabled, ?),
                                    rsi_time_filter_candles = COALESCE(rsi_time_filter_candles, ?),
                                    rsi_time_filter_upper = COALESCE(rsi_time_filter_upper, ?),
                                    rsi_time_filter_lower = COALESCE(rsi_time_filter_lower, ?),
                                    avoid_down_trend = COALESCE(avoid_down_trend, ?),
                                    extra_config_json = COALESCE(extra_config_json, ?)
                                WHERE symbol = ? AND rsi_long_threshold IS NULL
                            """, (
                                rsi_long_threshold, rsi_short_threshold,
                                rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                                rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                                max_loss_percent, take_profit_percent,
                                trailing_stop_activation, trailing_stop_distance,
                                trailing_take_distance, trailing_update_interval,
                                break_even_trigger, break_even_protection,
                                max_position_hours, rsi_time_filter_enabled,
                                rsi_time_filter_candles, rsi_time_filter_upper,
                                rsi_time_filter_lower, avoid_down_trend,
                                extra_config_json, symbol
                            ))
                            if cursor.rowcount > 0:
                                migrated_count += 1
                        except Exception as e:
                            pass
                            continue
                    
                    if migrated_count > 0:
                        logger.info(f"✅ Миграция bot_configs завершена: {migrated_count} записей мигрировано из JSON в нормализованные столбцы")
            except Exception as e:
                pass
            
            conn.commit()
        except Exception as e:
            pass
    
    # ==================== МЕТОДЫ ДЛЯ СИМУЛЯЦИЙ ====================
    
    def save_simulated_trades(self, trades: List[Dict[str, Any]], training_session_id: Optional[int] = None) -> int:
        """
        Сохраняет симулированные сделки в БД
        
        Args:
            trades: Список симулированных сделок
            training_session_id: ID сессии обучения (опционально)
        
        Returns:
            Количество сохраненных сделок
        """
        if not trades:
            return 0
        
        saved_count = 0
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                for trade in trades:
                    try:
                        # Извлекаем параметры из JSON или напрямую
                        rsi_params = trade.get('rsi_params', {})
                        risk_params = trade.get('risk_params', {})
                        
                        # Поддерживаем оба формата ключей
                        rsi_long = rsi_params.get('oversold') if isinstance(rsi_params, dict) else None
                        if rsi_long is None:
                            rsi_long = rsi_params.get('rsi_long_threshold') if isinstance(rsi_params, dict) else None
                        if rsi_long is None:
                            rsi_long = trade.get('rsi_long_threshold')
                        
                        rsi_short = rsi_params.get('overbought') if isinstance(rsi_params, dict) else None
                        if rsi_short is None:
                            rsi_short = rsi_params.get('rsi_short_threshold') if isinstance(rsi_params, dict) else None
                        if rsi_short is None:
                            rsi_short = trade.get('rsi_short_threshold')
                        
                        rsi_exit_long_with = rsi_params.get('exit_long_with_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_long_with is None:
                            rsi_exit_long_with = rsi_params.get('rsi_exit_long_with_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_long_with is None:
                            rsi_exit_long_with = trade.get('rsi_exit_long_with_trend')
                        
                        rsi_exit_long_against = rsi_params.get('exit_long_against_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_long_against is None:
                            rsi_exit_long_against = rsi_params.get('rsi_exit_long_against_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_long_against is None:
                            rsi_exit_long_against = trade.get('rsi_exit_long_against_trend')
                        
                        rsi_exit_short_with = rsi_params.get('exit_short_with_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_short_with is None:
                            rsi_exit_short_with = rsi_params.get('rsi_exit_short_with_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_short_with is None:
                            rsi_exit_short_with = trade.get('rsi_exit_short_with_trend')
                        
                        rsi_exit_short_against = rsi_params.get('exit_short_against_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_short_against is None:
                            rsi_exit_short_against = rsi_params.get('rsi_exit_short_against_trend') if isinstance(rsi_params, dict) else None
                        if rsi_exit_short_against is None:
                            rsi_exit_short_against = trade.get('rsi_exit_short_against_trend')
                        
                        # Risk параметры
                        max_loss = risk_params.get('max_loss_percent') if isinstance(risk_params, dict) else trade.get('max_loss_percent')
                        take_profit = risk_params.get('take_profit_percent') if isinstance(risk_params, dict) else trade.get('take_profit_percent')
                        trailing_activation = risk_params.get('trailing_stop_activation') if isinstance(risk_params, dict) else trade.get('trailing_stop_activation')
                        trailing_distance = risk_params.get('trailing_stop_distance') if isinstance(risk_params, dict) else trade.get('trailing_stop_distance')
                        trailing_take = risk_params.get('trailing_take_distance') if isinstance(risk_params, dict) else trade.get('trailing_take_distance')
                        trailing_interval = risk_params.get('trailing_update_interval') if isinstance(risk_params, dict) else trade.get('trailing_update_interval')
                        break_even_trigger = risk_params.get('break_even_trigger') if isinstance(risk_params, dict) else trade.get('break_even_trigger')
                        break_even_protection = risk_params.get('break_even_protection') if isinstance(risk_params, dict) else trade.get('break_even_protection')
                        max_hours = risk_params.get('max_position_hours') if isinstance(risk_params, dict) else trade.get('max_position_hours')
                        
                        # Собираем остальные параметры в extra_params_json
                        extra_params = {}
                        if isinstance(rsi_params, dict):
                            known_rsi_keys = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                                            'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold',
                                            'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                                            'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                            for key, value in rsi_params.items():
                                if key not in known_rsi_keys:
                                    extra_params[key] = value
                        if isinstance(risk_params, dict):
                            known_risk_keys = {'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                                             'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                                             'break_even_trigger', 'break_even_protection', 'max_position_hours'}
                            for key, value in risk_params.items():
                                if key not in known_risk_keys:
                                    extra_params[key] = value
                        
                        extra_params_json = json.dumps(extra_params, ensure_ascii=False) if extra_params else None
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO simulated_trades (
                                symbol, direction, entry_price, exit_price,
                                entry_time, exit_time, entry_rsi, exit_rsi,
                                entry_trend, exit_trend, entry_volatility, entry_volume_ratio,
                                pnl, pnl_pct, roi,
                                exit_reason, is_successful, duration_candles,
                                entry_idx, exit_idx, simulation_timestamp,
                                training_session_id,
                                rsi_long_threshold, rsi_short_threshold,
                                rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                                rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                                max_loss_percent, take_profit_percent,
                                trailing_stop_activation, trailing_stop_distance,
                                trailing_take_distance, trailing_update_interval,
                                break_even_trigger, break_even_protection, max_position_hours,
                                config_params_json, filters_params_json, entry_conditions_json,
                                exit_conditions_json, restrictions_json, extra_params_json,
                                created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trade.get('symbol'),
                            trade.get('direction'),
                            trade.get('entry_price'),
                            trade.get('exit_price'),
                            trade.get('entry_time'),
                            trade.get('exit_time'),
                            trade.get('entry_rsi'),
                            trade.get('exit_rsi'),
                            trade.get('entry_trend'),
                            trade.get('exit_trend'),
                            trade.get('entry_volatility'),
                            trade.get('entry_volume_ratio'),
                            trade.get('pnl'),
                            trade.get('pnl_pct'),
                            trade.get('roi'),
                            trade.get('exit_reason'),
                            1 if trade.get('is_successful', False) else 0,
                            trade.get('duration_candles'),
                            trade.get('entry_idx'),
                            trade.get('exit_idx'),
                            trade.get('simulation_timestamp', now),
                            training_session_id,
                            rsi_long, rsi_short,
                            rsi_exit_long_with, rsi_exit_long_against,
                            rsi_exit_short_with, rsi_exit_short_against,
                            max_loss, take_profit,
                            trailing_activation, trailing_distance,
                            trailing_take, trailing_interval,
                            break_even_trigger, break_even_protection, max_hours,
                            json.dumps(trade.get('config_params'), ensure_ascii=False) if trade.get('config_params') else None,
                            json.dumps(trade.get('filters_params'), ensure_ascii=False) if trade.get('filters_params') else None,
                            json.dumps(trade.get('entry_conditions'), ensure_ascii=False) if trade.get('entry_conditions') else None,
                            json.dumps(trade.get('exit_conditions'), ensure_ascii=False) if trade.get('exit_conditions') else None,
                            json.dumps(trade.get('restrictions'), ensure_ascii=False) if trade.get('restrictions') else None,
                            extra_params_json,
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
    
    def get_simulated_trades(self, 
                            symbol: Optional[str] = None,
                            min_pnl: Optional[float] = None,
                            max_pnl: Optional[float] = None,
                            is_successful: Optional[bool] = None,
                            limit: Optional[int] = None,
                            offset: int = 0) -> List[Dict[str, Any]]:
        """
        Получает симулированные сделки с фильтрацией
        
        Args:
            symbol: Фильтр по символу
            min_pnl: Минимальный PnL
            max_pnl: Максимальный PnL
            is_successful: Фильтр по успешности
            limit: Лимит записей
            offset: Смещение
        
        Returns:
            Список сделок
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM simulated_trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if min_pnl is not None:
                query += " AND pnl >= ?"
                params.append(min_pnl)
            
            if max_pnl is not None:
                query += " AND pnl <= ?"
                params.append(max_pnl)
            
            if is_successful is not None:
                query += " AND is_successful = ?"
                params.append(1 if is_successful else 0)
            
            query += " ORDER BY entry_time DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                trade = dict(row)
                
                # Восстанавливаем rsi_params из нормализованных столбцов
                rsi_params = {}
                if trade.get('rsi_long_threshold') is not None:
                    rsi_params['oversold'] = trade['rsi_long_threshold']
                    rsi_params['rsi_long_threshold'] = trade['rsi_long_threshold']
                if trade.get('rsi_short_threshold') is not None:
                    rsi_params['overbought'] = trade['rsi_short_threshold']
                    rsi_params['rsi_short_threshold'] = trade['rsi_short_threshold']
                if trade.get('rsi_exit_long_with_trend') is not None:
                    rsi_params['exit_long_with_trend'] = trade['rsi_exit_long_with_trend']
                    rsi_params['rsi_exit_long_with_trend'] = trade['rsi_exit_long_with_trend']
                if trade.get('rsi_exit_long_against_trend') is not None:
                    rsi_params['exit_long_against_trend'] = trade['rsi_exit_long_against_trend']
                    rsi_params['rsi_exit_long_against_trend'] = trade['rsi_exit_long_against_trend']
                if trade.get('rsi_exit_short_with_trend') is not None:
                    rsi_params['exit_short_with_trend'] = trade['rsi_exit_short_with_trend']
                    rsi_params['rsi_exit_short_with_trend'] = trade['rsi_exit_short_with_trend']
                if trade.get('rsi_exit_short_against_trend') is not None:
                    rsi_params['exit_short_against_trend'] = trade['rsi_exit_short_against_trend']
                    rsi_params['rsi_exit_short_against_trend'] = trade['rsi_exit_short_against_trend']
                
                # Загружаем extra_params_json если есть
                if trade.get('extra_params_json'):
                    try:
                        extra_params = json.loads(trade['extra_params_json'])
                        rsi_params.update(extra_params)
                    except:
                        pass
                
                if rsi_params:
                    trade['rsi_params'] = rsi_params
                    trade['rsi_params_json'] = json.dumps(rsi_params, ensure_ascii=False)  # Для обратной совместимости
                
                # Восстанавливаем risk_params из нормализованных столбцов
                risk_params = {}
                if trade.get('max_loss_percent') is not None:
                    risk_params['max_loss_percent'] = trade['max_loss_percent']
                if trade.get('take_profit_percent') is not None:
                    risk_params['take_profit_percent'] = trade['take_profit_percent']
                if trade.get('trailing_stop_activation') is not None:
                    risk_params['trailing_stop_activation'] = trade['trailing_stop_activation']
                if trade.get('trailing_stop_distance') is not None:
                    risk_params['trailing_stop_distance'] = trade['trailing_stop_distance']
                if trade.get('trailing_take_distance') is not None:
                    risk_params['trailing_take_distance'] = trade['trailing_take_distance']
                if trade.get('trailing_update_interval') is not None:
                    risk_params['trailing_update_interval'] = trade['trailing_update_interval']
                if trade.get('break_even_trigger') is not None:
                    risk_params['break_even_trigger'] = trade['break_even_trigger']
                if trade.get('break_even_protection') is not None:
                    risk_params['break_even_protection'] = trade['break_even_protection']
                if trade.get('max_position_hours') is not None:
                    risk_params['max_position_hours'] = trade['max_position_hours']
                
                # Загружаем extra_params_json если есть (может содержать дополнительные risk параметры)
                if trade.get('extra_params_json'):
                    try:
                        extra_params = json.loads(trade['extra_params_json'])
                        # Добавляем только risk параметры
                        known_risk_keys = {'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                                         'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                                         'break_even_trigger', 'break_even_protection', 'max_position_hours'}
                        for key, value in extra_params.items():
                            if key not in known_risk_keys and key not in rsi_params:
                                risk_params[key] = value
                    except:
                        pass
                
                if risk_params:
                    trade['risk_params'] = risk_params
                    trade['risk_params_json'] = json.dumps(risk_params, ensure_ascii=False)  # Для обратной совместимости
                
                result.append(trade)
            
            return result
    
    def count_simulated_trades(self, symbol: Optional[str] = None) -> int:
        """Подсчитывает количество симуляций"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("SELECT COUNT(*) FROM simulated_trades WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("SELECT COUNT(*) FROM simulated_trades")
            
            return cursor.fetchone()[0]
    
    # ==================== МЕТОДЫ ДЛЯ РЕАЛЬНЫХ СДЕЛОК БОТОВ ====================
    
    def save_bot_trade(self, trade: Dict[str, Any]) -> Optional[int]:
        """Сохраняет или обновляет сделку бота"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Проверяем, существует ли сделка
                trade_id = trade.get('id') or trade.get('trade_id')
                if trade_id:
                    cursor.execute("SELECT id FROM bot_trades WHERE trade_id = ?", (trade_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Извлекаем volatility и volume_ratio из entry_data если есть
                        entry_data = trade.get('entry_data', {})
                        if isinstance(entry_data, str):
                            try:
                                entry_data = json.loads(entry_data)
                            except:
                                entry_data = {}
                        elif not isinstance(entry_data, dict):
                            entry_data = {}
                        
                        entry_volatility = trade.get('entry_volatility') or entry_data.get('volatility')
                        entry_volume_ratio = trade.get('entry_volume_ratio') or entry_data.get('volume_ratio')
                        
                        # Обновляем существующую
                        cursor.execute("""
                            UPDATE bot_trades SET
                                symbol = ?, direction = ?, entry_price = ?, exit_price = ?,
                                pnl = ?, roi = ?, status = ?, exit_rsi = ?, exit_trend = ?,
                                entry_volatility = ?, entry_volume_ratio = ?,
                                close_reason = ?, exit_market_data_json = ?, updated_at = ?
                            WHERE trade_id = ?
                        """, (
                            trade.get('symbol'),
                            trade.get('direction'),
                            trade.get('entry_price'),
                            trade.get('exit_price'),
                            trade.get('pnl'),
                            trade.get('roi'),
                            trade.get('status'),
                            trade.get('exit_rsi'),
                            trade.get('exit_trend'),
                            entry_volatility,
                            entry_volume_ratio,
                            trade.get('close_reason'),
                            json.dumps(trade.get('exit_market_data'), ensure_ascii=False) if trade.get('exit_market_data') else None,
                            now,
                            trade_id
                        ))
                        return existing[0]
                
                # Создаем новую
                # Извлекаем volatility и volume_ratio из entry_data если есть
                entry_data = trade.get('entry_data', {})
                if isinstance(entry_data, str):
                    try:
                        entry_data = json.loads(entry_data)
                    except:
                        entry_data = {}
                elif not isinstance(entry_data, dict):
                    entry_data = {}
                
                entry_volatility = trade.get('entry_volatility') or entry_data.get('volatility')
                entry_volume_ratio = trade.get('entry_volume_ratio') or entry_data.get('volume_ratio')
                
                # Извлекаем все параметры конфига из trade или entry_data
                config_params = trade.get('config_params') or trade.get('config') or entry_data.get('config')
                filters_params = trade.get('filters_params') or trade.get('filters') or entry_data.get('filters')
                entry_conditions = trade.get('entry_conditions') or entry_data.get('entry_conditions')
                exit_market_data = trade.get('exit_market_data') or trade.get('market_data', {})
                if isinstance(exit_market_data, str):
                    try:
                        exit_market_data = json.loads(exit_market_data)
                    except:
                        exit_market_data = {}
                elif not isinstance(exit_market_data, dict):
                    exit_market_data = {}
                exit_conditions = trade.get('exit_conditions') or exit_market_data.get('exit_conditions')
                restrictions = trade.get('restrictions') or entry_data.get('restrictions')
                
                # Извлекаем RSI и Risk параметры из config_params
                rsi_params = config_params.get('rsi_params', {}) if isinstance(config_params, dict) and isinstance(config_params.get('rsi_params'), dict) else {}
                if not rsi_params and isinstance(config_params, dict):
                    # Пытаемся извлечь напрямую из config_params
                    rsi_params = {k: v for k, v in config_params.items() if 'rsi' in k.lower() or k in ['oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend']}
                
                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold') or (config_params.get('rsi_long_threshold') if isinstance(config_params, dict) else None) or trade.get('rsi_long_threshold')
                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold') or (config_params.get('rsi_short_threshold') if isinstance(config_params, dict) else None) or trade.get('rsi_short_threshold')
                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend') or (config_params.get('rsi_exit_long_with_trend') if isinstance(config_params, dict) else None) or trade.get('rsi_exit_long_with_trend')
                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend') or (config_params.get('rsi_exit_long_against_trend') if isinstance(config_params, dict) else None) or trade.get('rsi_exit_long_against_trend')
                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend') or (config_params.get('rsi_exit_short_with_trend') if isinstance(config_params, dict) else None) or trade.get('rsi_exit_short_with_trend')
                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend') or (config_params.get('rsi_exit_short_against_trend') if isinstance(config_params, dict) else None) or trade.get('rsi_exit_short_against_trend')
                
                # Risk параметры
                risk_params = config_params.get('risk_params', {}) if isinstance(config_params, dict) and isinstance(config_params.get('risk_params'), dict) else {}
                if not risk_params and isinstance(config_params, dict):
                    # Пытаемся извлечь напрямую из config_params
                    risk_params = {k: v for k, v in config_params.items() if k in ['max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours']}
                
                max_loss = risk_params.get('max_loss_percent') or (config_params.get('max_loss_percent') if isinstance(config_params, dict) else None) or trade.get('max_loss_percent')
                take_profit = risk_params.get('take_profit_percent') or (config_params.get('take_profit_percent') if isinstance(config_params, dict) else None) or trade.get('take_profit_percent')
                trailing_activation = risk_params.get('trailing_stop_activation') or (config_params.get('trailing_stop_activation') if isinstance(config_params, dict) else None) or trade.get('trailing_stop_activation')
                trailing_distance = risk_params.get('trailing_stop_distance') or (config_params.get('trailing_stop_distance') if isinstance(config_params, dict) else None) or trade.get('trailing_stop_distance')
                trailing_take = risk_params.get('trailing_take_distance') or (config_params.get('trailing_take_distance') if isinstance(config_params, dict) else None) or trade.get('trailing_take_distance')
                trailing_interval = risk_params.get('trailing_update_interval') or (config_params.get('trailing_update_interval') if isinstance(config_params, dict) else None) or trade.get('trailing_update_interval')
                break_even_trigger = risk_params.get('break_even_trigger') or (config_params.get('break_even_trigger') if isinstance(config_params, dict) else None) or trade.get('break_even_trigger')
                break_even_protection = risk_params.get('break_even_protection') or (config_params.get('break_even_protection') if isinstance(config_params, dict) else None) or trade.get('break_even_protection')
                max_hours = risk_params.get('max_position_hours') or (config_params.get('max_position_hours') if isinstance(config_params, dict) else None) or trade.get('max_position_hours')
                
                # Собираем остальные параметры в extra_config_json
                extra_config = {}
                if isinstance(config_params, dict):
                    known_fields = {
                        'rsi_params', 'risk_params', 'rsi_long_threshold', 'rsi_short_threshold',
                        'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                        'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                        'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                        'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                        'break_even_trigger', 'break_even_protection', 'max_position_hours',
                        'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                        'exit_short_with_trend', 'exit_short_against_trend'
                    }
                    for key, value in config_params.items():
                        if key not in known_fields:
                            extra_config[key] = value
                
                extra_config_json = json.dumps(extra_config, ensure_ascii=False) if extra_config else None
                
                cursor.execute("""
                    INSERT OR IGNORE INTO bot_trades (
                        trade_id, bot_id, symbol, direction, entry_price, exit_price,
                        entry_time, exit_time, pnl, roi, status, decision_source,
                        ai_decision_id, ai_confidence, entry_rsi, exit_rsi,
                        entry_trend, exit_trend, entry_volatility, entry_volume_ratio,
                        close_reason,
                        position_size_usdt, position_size_coins,
                        rsi_long_threshold, rsi_short_threshold,
                        rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                        rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                        max_loss_percent, take_profit_percent,
                        trailing_stop_activation, trailing_stop_distance,
                        trailing_take_distance, trailing_update_interval,
                        break_even_trigger, break_even_protection, max_position_hours,
                        entry_data_json, exit_market_data_json,
                        filters_params_json, entry_conditions_json,
                        exit_conditions_json, restrictions_json, extra_config_json,
                        is_simulated,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    trade.get('bot_id'),
                    trade.get('symbol'),
                    trade.get('direction'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('timestamp') or trade.get('entry_time'),
                    trade.get('close_timestamp') or trade.get('exit_time'),
                    trade.get('pnl'),
                    trade.get('roi'),
                    trade.get('status', 'CLOSED'),
                    trade.get('decision_source', 'SCRIPT'),
                    trade.get('ai_decision_id'),
                    trade.get('ai_confidence'),
                    trade.get('entry_rsi') or entry_data.get('rsi'),
                    trade.get('exit_rsi') or exit_market_data.get('rsi'),
                    trade.get('entry_trend') or entry_data.get('trend'),
                    trade.get('exit_trend') or exit_market_data.get('trend'),
                    entry_volatility,
                    entry_volume_ratio,
                    trade.get('close_reason'),
                    trade.get('position_size_usdt'),
                    trade.get('position_size_coins'),
                    rsi_long, rsi_short,
                    rsi_exit_long_with, rsi_exit_long_against,
                    rsi_exit_short_with, rsi_exit_short_against,
                    max_loss, take_profit,
                    trailing_activation, trailing_distance,
                    trailing_take, trailing_interval,
                    break_even_trigger, break_even_protection, max_hours,
                    json.dumps(trade.get('entry_data'), ensure_ascii=False) if trade.get('entry_data') else None,
                    json.dumps(trade.get('exit_market_data') or trade.get('market_data'), ensure_ascii=False) if (trade.get('exit_market_data') or trade.get('market_data')) else None,
                    json.dumps(filters_params, ensure_ascii=False) if filters_params else None,
                    json.dumps(entry_conditions, ensure_ascii=False) if entry_conditions else None,
                    json.dumps(exit_conditions, ensure_ascii=False) if exit_conditions else None,
                    json.dumps(restrictions, ensure_ascii=False) if restrictions else None,
                    extra_config_json,
                    1 if trade.get('is_simulated', False) else 0,
                    now,
                    now
                ))
                
                return cursor.lastrowid
    
    def get_bot_trades(self,
                       symbol: Optional[str] = None,
                       bot_id: Optional[str] = None,
                       status: Optional[str] = None,
                       decision_source: Optional[str] = None,
                       min_pnl: Optional[float] = None,
                       max_pnl: Optional[float] = None,
                       limit: Optional[int] = None,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """Получает сделки ботов с фильтрацией"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM bot_trades WHERE is_simulated = 0"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if bot_id:
                query += " AND bot_id = ?"
                params.append(bot_id)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if decision_source:
                query += " AND decision_source = ?"
                params.append(decision_source)
            
            if min_pnl is not None:
                query += " AND pnl >= ?"
                params.append(min_pnl)
            
            if max_pnl is not None:
                query += " AND pnl <= ?"
                params.append(max_pnl)
            
            query += " ORDER BY entry_time DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                trade = dict(row)
                
                # Восстанавливаем config_params из нормализованных столбцов
                config_params = {}
                rsi_params = {}
                if trade.get('rsi_long_threshold') is not None:
                    rsi_params['oversold'] = trade['rsi_long_threshold']
                    rsi_params['rsi_long_threshold'] = trade['rsi_long_threshold']
                if trade.get('rsi_short_threshold') is not None:
                    rsi_params['overbought'] = trade['rsi_short_threshold']
                    rsi_params['rsi_short_threshold'] = trade['rsi_short_threshold']
                if trade.get('rsi_exit_long_with_trend') is not None:
                    rsi_params['exit_long_with_trend'] = trade['rsi_exit_long_with_trend']
                    rsi_params['rsi_exit_long_with_trend'] = trade['rsi_exit_long_with_trend']
                if trade.get('rsi_exit_long_against_trend') is not None:
                    rsi_params['exit_long_against_trend'] = trade['rsi_exit_long_against_trend']
                    rsi_params['rsi_exit_long_against_trend'] = trade['rsi_exit_long_against_trend']
                if trade.get('rsi_exit_short_with_trend') is not None:
                    rsi_params['exit_short_with_trend'] = trade['rsi_exit_short_with_trend']
                    rsi_params['rsi_exit_short_with_trend'] = trade['rsi_exit_short_with_trend']
                if trade.get('rsi_exit_short_against_trend') is not None:
                    rsi_params['exit_short_against_trend'] = trade['rsi_exit_short_against_trend']
                    rsi_params['rsi_exit_short_against_trend'] = trade['rsi_exit_short_against_trend']
                
                risk_params = {}
                if trade.get('max_loss_percent') is not None:
                    risk_params['max_loss_percent'] = trade['max_loss_percent']
                if trade.get('take_profit_percent') is not None:
                    risk_params['take_profit_percent'] = trade['take_profit_percent']
                if trade.get('trailing_stop_activation') is not None:
                    risk_params['trailing_stop_activation'] = trade['trailing_stop_activation']
                if trade.get('trailing_stop_distance') is not None:
                    risk_params['trailing_stop_distance'] = trade['trailing_stop_distance']
                if trade.get('trailing_take_distance') is not None:
                    risk_params['trailing_take_distance'] = trade['trailing_take_distance']
                if trade.get('trailing_update_interval') is not None:
                    risk_params['trailing_update_interval'] = trade['trailing_update_interval']
                if trade.get('break_even_trigger') is not None:
                    risk_params['break_even_trigger'] = trade['break_even_trigger']
                if trade.get('break_even_protection') is not None:
                    risk_params['break_even_protection'] = trade['break_even_protection']
                if trade.get('max_position_hours') is not None:
                    risk_params['max_position_hours'] = trade['max_position_hours']
                
                # Загружаем extra_config_json если есть
                if trade.get('extra_config_json'):
                    try:
                        extra_config = json.loads(trade['extra_config_json'])
                        config_params.update(extra_config)
                    except:
                        pass
                
                if rsi_params:
                    config_params['rsi_params'] = rsi_params
                if risk_params:
                    config_params['risk_params'] = risk_params
                
                if config_params:
                    trade['config_params'] = config_params
                    trade['config_params_json'] = json.dumps(config_params, ensure_ascii=False)  # Для обратной совместимости
                
                # Восстанавливаем JSON поля
                if trade.get('entry_data_json'):
                    trade['entry_data'] = json.loads(trade['entry_data_json'])
                if trade.get('exit_market_data_json'):
                    trade['exit_market_data'] = json.loads(trade['exit_market_data_json'])
                result.append(trade)
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ ИСТОРИИ БИРЖИ ====================
    
    def save_exchange_trades(self, trades: List[Dict[str, Any]]) -> int:
        """Сохраняет сделки с биржи"""
        if not trades:
            return 0
        
        saved_count = 0
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                for trade in trades:
                    try:
                        trade_id = trade.get('id') or trade.get('orderId') or f"exchange_{trade.get('symbol')}_{trade.get('timestamp')}"
                        cursor.execute("""
                            INSERT OR IGNORE INTO exchange_trades (
                                trade_id, symbol, direction, entry_price, exit_price,
                                entry_time, exit_time, pnl, roi,
                                position_size_usdt, position_size_coins,
                                order_id, source, saved_timestamp, is_real, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trade_id,
                            trade.get('symbol'),
                            trade.get('direction'),
                            trade.get('entry_price'),
                            trade.get('exit_price'),
                            trade.get('timestamp'),
                            trade.get('close_timestamp'),
                            trade.get('pnl'),
                            trade.get('roi'),
                            trade.get('position_size_usdt'),
                            trade.get('position_size_coins'),
                            trade.get('orderId'),
                            trade.get('source', 'exchange_api'),
                            trade.get('saved_timestamp', now),
                            1,
                            now
                        ))
                        if cursor.rowcount > 0:
                            saved_count += 1
                    except Exception as e:
                        pass
                        continue
                
                conn.commit()
        
        return saved_count
    
    def count_exchange_trades(self) -> int:
        """Подсчитывает количество сделок биржи"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM exchange_trades")
            return cursor.fetchone()[0]
    
    def count_bot_trades(self, symbol: Optional[str] = None, is_simulated: Optional[bool] = None) -> int:
        """Подсчитывает количество сделок ботов"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT COUNT(*) FROM bot_trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if is_simulated is not None:
                query += " AND is_simulated = ?"
                params.append(1 if is_simulated else 0)
            
            cursor.execute(query, params)
            return cursor.fetchone()[0]
    
    # ==================== МЕТОДЫ ДЛЯ РЕШЕНИЙ AI ====================
    
    def save_ai_decision(self, decision: Dict[str, Any]) -> int:
        """Сохраняет решение AI с нормализованными полями"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Извлекаем данные из market_data
                market_data = decision.get('market_data', {})
                if isinstance(market_data, str):
                    try:
                        market_data = json.loads(market_data)
                    except:
                        market_data = {}
                
                volume = market_data.get('volume') if isinstance(market_data, dict) else None
                volatility = market_data.get('volatility') if isinstance(market_data, dict) else None
                volume_ratio = market_data.get('volume_ratio') if isinstance(market_data, dict) else None
                
                # Собираем остальные поля market_data в extra_market_data_json
                extra_market_data = {}
                if isinstance(market_data, dict):
                    known_market_keys = {'volume', 'volatility', 'volume_ratio', 'rsi', 'trend', 'price', 'signal', 'confidence'}
                    for key, value in market_data.items():
                        if key not in known_market_keys:
                            extra_market_data[key] = value
                
                extra_market_data_json = json.dumps(extra_market_data, ensure_ascii=False) if extra_market_data else None
                
                # Извлекаем данные из decision_params/params
                decision_params = decision.get('params') or decision.get('decision_params', {})
                if isinstance(decision_params, str):
                    try:
                        decision_params = json.loads(decision_params)
                    except:
                        decision_params = {}
                
                rsi_long_threshold = decision_params.get('rsi_long_threshold') if isinstance(decision_params, dict) else None
                rsi_short_threshold = decision_params.get('rsi_short_threshold') if isinstance(decision_params, dict) else None
                max_loss_percent = decision_params.get('max_loss_percent') if isinstance(decision_params, dict) else None
                take_profit_percent = decision_params.get('take_profit_percent') if isinstance(decision_params, dict) else None
                
                # Собираем остальные поля decision_params в extra_decision_params_json
                extra_decision_params = {}
                if isinstance(decision_params, dict):
                    known_params_keys = {'rsi_long_threshold', 'rsi_short_threshold', 'max_loss_percent', 'take_profit_percent'}
                    for key, value in decision_params.items():
                        if key not in known_params_keys:
                            extra_decision_params[key] = value
                
                extra_decision_params_json = json.dumps(extra_decision_params, ensure_ascii=False) if extra_decision_params else None
                
                # Сохраняем полные JSON для обратной совместимости
                market_data_json = json.dumps(market_data, ensure_ascii=False) if market_data else None
                decision_params_json = json.dumps(decision_params, ensure_ascii=False) if decision_params else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ai_decisions (
                        decision_id, symbol, decision_type, signal, confidence,
                        rsi, trend, price,
                        volume, volatility, volume_ratio,
                        rsi_long_threshold, rsi_short_threshold,
                        max_loss_percent, take_profit_percent,
                        market_data_json, decision_params_json,
                        extra_market_data_json, extra_decision_params_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.get('decision_id'),
                    decision.get('symbol'),
                    decision.get('decision_type', 'SIGNAL'),
                    decision.get('signal'),
                    decision.get('confidence'),
                    decision.get('rsi'),
                    decision.get('trend'),
                    decision.get('price'),
                    volume,
                    volatility,
                    volume_ratio,
                    rsi_long_threshold,
                    rsi_short_threshold,
                    max_loss_percent,
                    take_profit_percent,
                    market_data_json,
                    decision_params_json,
                    extra_market_data_json,
                    extra_decision_params_json,
                    now
                ))
                
                return cursor.lastrowid
    
    def update_ai_decision_result(self, decision_id: str, pnl: float, is_successful: bool):
        """Обновляет результат решения AI"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute("""
                    UPDATE ai_decisions SET
                        result_pnl = ?, result_successful = ?, executed_at = ?
                    WHERE decision_id = ?
                """, (pnl, 1 if is_successful else 0, now, decision_id))
    
    def get_ai_decisions(self, status: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Получает решения AI с фильтрацией, восстанавливая структуру market_data и params"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM ai_decisions WHERE 1=1"
            params = []
            
            if status:
                query += " AND result_successful = ?"
                params.append(1 if status == 'SUCCESS' else 0)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                decision = dict(row)
                
                # Восстанавливаем market_data из нормализованных полей или JSON
                market_data = {}
                if decision.get('market_data_json'):
                    try:
                        market_data = json.loads(decision['market_data_json'])
                    except:
                        market_data = {}
                
                # Добавляем нормализованные поля в market_data
                if decision.get('volume') is not None:
                    market_data['volume'] = decision['volume']
                if decision.get('volatility') is not None:
                    market_data['volatility'] = decision['volatility']
                if decision.get('volume_ratio') is not None:
                    market_data['volume_ratio'] = decision['volume_ratio']
                
                # Добавляем extra_market_data
                if decision.get('extra_market_data_json'):
                    try:
                        extra_market_data = json.loads(decision['extra_market_data_json'])
                        market_data.update(extra_market_data)
                    except:
                        pass
                
                decision['market_data'] = market_data if market_data else None
                
                # Восстанавливаем params из нормализованных полей или JSON
                decision_params = {}
                if decision.get('decision_params_json'):
                    try:
                        decision_params = json.loads(decision['decision_params_json'])
                    except:
                        decision_params = {}
                
                # Добавляем нормализованные поля в params
                if decision.get('rsi_long_threshold') is not None:
                    decision_params['rsi_long_threshold'] = decision['rsi_long_threshold']
                if decision.get('rsi_short_threshold') is not None:
                    decision_params['rsi_short_threshold'] = decision['rsi_short_threshold']
                if decision.get('max_loss_percent') is not None:
                    decision_params['max_loss_percent'] = decision['max_loss_percent']
                if decision.get('take_profit_percent') is not None:
                    decision_params['take_profit_percent'] = decision['take_profit_percent']
                
                # Добавляем extra_decision_params
                if decision.get('extra_decision_params_json'):
                    try:
                        extra_decision_params = json.loads(decision['extra_decision_params_json'])
                        decision_params.update(extra_decision_params)
                    except:
                        pass
                
                decision['params'] = decision_params if decision_params else None
                decision['status'] = 'SUCCESS' if decision.get('result_successful') else 'FAILED' if decision.get('result_successful') is not None else 'PENDING'
                result.append(decision)
            
            return result
    
    # ==================== РЕКОМЕНДАЦИИ AI (чтение из bots.py, запись из ai.py) ====================
    
    def save_ai_recommendation(self, symbol: str, direction: str, data: Dict[str, Any]) -> None:
        """Сохраняет последнюю рекомендацию AI по символу и направлению (пишет только ai.py)."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                data_json = json.dumps(data, ensure_ascii=False) if data else None
                cursor.execute("""
                    INSERT OR REPLACE INTO ai_recommendations (
                        symbol, direction, should_open, signal, confidence, reason,
                        ai_used, smc_used, data_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    direction,
                    1 if data.get('should_open', True) else 0,
                    data.get('signal') or data.get('ai_signal'),
                    data.get('confidence') or data.get('ai_confidence'),
                    data.get('reason'),
                    1 if data.get('ai_used', False) else 0,
                    1 if data.get('smc_used', False) else 0,
                    data_json,
                    now,
                ))
    
    def get_latest_ai_recommendation(self, symbol: str, direction: str) -> Optional[Dict[str, Any]]:
        """Возвращает последнюю рекомендацию AI по символу и направлению (читает bots.py)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT should_open, signal, confidence, reason, ai_used, smc_used, data_json, updated_at
                FROM ai_recommendations WHERE symbol = ? AND direction = ?
            """, (symbol, direction))
            row = cursor.fetchone()
            if not row:
                return None
            try:
                data = json.loads(row[6]) if row[6] else {}
            except Exception:
                data = {}
            return {
                'should_open': bool(row[0]),
                'signal': row[1],
                'confidence': row[2] or 0,
                'reason': row[3],
                'ai_used': bool(row[4]),
                'smc_used': bool(row[5]),
                'updated_at': row[7],
                **data,
            }
    
    # ==================== МЕТОДЫ ДЛЯ СЕССИЙ ОБУЧЕНИЯ ====================
    
    def create_training_session(self, session_type: str, training_seed: Optional[int] = None, metadata: Optional[Dict] = None) -> int:
        """Создает новую сессию обучения"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT INTO training_sessions (
                        session_type, training_seed, started_at, status, metadata_json
                    ) VALUES (?, ?, ?, 'RUNNING', ?)
                """, (
                    session_type,
                    training_seed,
                    now,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None
                ))
                
                return cursor.lastrowid
    
    def update_training_session(self, session_id: int, **kwargs):
        """Обновляет сессию обучения"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                updates = []
                params = []
                
                for key, value in kwargs.items():
                    if key == 'metadata' and isinstance(value, dict):
                        updates.append("metadata_json = ?")
                        params.append(json.dumps(value, ensure_ascii=False))
                    elif key in ('coins_processed', 'models_saved', 'candles_processed', 
                                'total_trades', 'successful_trades', 'failed_trades',
                                'params_used', 'params_total'):
                        updates.append(f"{key} = ?")
                        params.append(value)
                    elif key in ('win_rate', 'total_pnl', 'accuracy', 'mse'):
                        updates.append(f"{key} = ?")
                        params.append(value)
                    elif key == 'status':
                        updates.append("status = ?")
                        params.append(value)
                        if value in ('COMPLETED', 'FAILED'):
                            updates.append("completed_at = ?")
                            params.append(now)
                
                if updates:
                    params.append(session_id)
                    cursor.execute(f"""
                        UPDATE training_sessions SET {', '.join(updates)}
                        WHERE id = ?
                    """, params)
    
    # ==================== СЛОЖНЫЕ ЗАПРОСЫ И АНАЛИЗ ====================
    
    def compare_simulated_vs_real(self, symbol: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        """
        Сравнивает симулированные и реальные сделки
        
        Returns:
            Статистика сравнения
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Статистика симуляций
            sim_query = "SELECT AVG(pnl) as avg_pnl, COUNT(*) as count, AVG(CASE WHEN is_successful = 1 THEN 1.0 ELSE 0.0 END) as win_rate FROM simulated_trades"
            sim_params = []
            if symbol:
                sim_query += " WHERE symbol = ?"
                sim_params.append(symbol)
            
            cursor.execute(sim_query, sim_params)
            sim_stats = dict(cursor.fetchone())
            
            # Статистика реальных сделок (с win_rate)
            real_query = """
                SELECT 
                    AVG(pnl) as avg_pnl, 
                    COUNT(*) as count,
                    AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(pnl) as total_pnl
                FROM bot_trades 
                WHERE is_simulated = 0 AND status = 'CLOSED' AND pnl IS NOT NULL
            """
            real_params = []
            if symbol:
                real_query += " AND symbol = ?"
                real_params.append(symbol)
            
            cursor.execute(real_query, real_params)
            real_row = cursor.fetchone()
            real_stats = dict(real_row) if real_row else {'avg_pnl': 0, 'count': 0, 'win_rate': 0, 'total_pnl': 0}
            
            # Вычисляем разницу производительности
            sim_win_rate = sim_stats.get('win_rate') or 0
            real_win_rate = real_stats.get('win_rate') or 0
            win_rate_diff = sim_win_rate - real_win_rate
            
            return {
                'simulated': sim_stats,
                'real': real_stats,
                'comparison': {
                    'pnl_diff': (sim_stats.get('avg_pnl') or 0) - (real_stats.get('avg_pnl') or 0),
                    'count_ratio': (sim_stats.get('count') or 0) / max(real_stats.get('count') or 1, 1),
                    'win_rate_diff': win_rate_diff,
                    'win_rate_simulated': sim_win_rate,
                    'win_rate_real': real_win_rate
                }
            }
    
    def get_trades_for_training(self,
                               include_simulated: bool = True,
                               include_real: bool = True,
                               include_exchange: bool = True,
                               min_trades: int = 10,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получает все сделки для обучения ИИ (объединенные из разных источников)
        
        ВАЖНО: AI система НЕ совершает реальные сделки, она только симулирует!
        Для обучения ИИ нужны сделки с RSI/трендом/волатильностью:
        - entry_rsi, entry_trend, entry_volatility (ОБЯЗАТЕЛЬНО для обучения!)
        - exit_rsi, exit_trend
        
        Источники данных:
        - ai_data.db -> simulated_trades (симуляции с полными данными)
        - ai_data.db -> bot_trades (реальные сделки ботов с RSI/трендом, если сохраняются)
        - ai_data.db -> exchange_trades (сделки с биржи, но может не быть RSI)
        - bots_data.db -> bot_trades_history (история торговли ботов с RSI/трендом)
        
        НЕ ИСПОЛЬЗУЕМ:
        - app_data.db -> closed_pnl (НЕТ RSI/тренда, только статистика PnL)
        - bots_data.db -> bots (только текущее состояние, не история закрытых сделок)
        
        Args:
            include_simulated: Включить симуляции (ai_data.db -> simulated_trades)
            include_real: Включить реальные сделки ботов (ai_data.db -> bot_trades)
            include_exchange: Включить сделки с биржи (ai_data.db -> exchange_trades)
            min_trades: Минимальное количество сделок для символа
            limit: Лимит на общее количество
        
        Returns:
            Список сделок для обучения (только с RSI/трендом)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Объединяем все источники через UNION
            queries = []
            params = []
            
            if include_simulated:
                queries.append("""
                    SELECT 
                        'SIMULATED' as source,
                        symbol, direction, entry_price, exit_price,
                        entry_rsi as rsi, entry_trend as trend,
                        entry_volatility, entry_volume_ratio,
                        pnl, pnl_pct as roi, is_successful,
                        entry_time as timestamp, exit_time as close_timestamp,
                        exit_reason as close_reason,
                        NULL as ai_decision_id, NULL as ai_confidence
                    FROM simulated_trades
                    WHERE exit_price IS NOT NULL
                """)
            
            if include_real:
                queries.append("""
                    SELECT 
                        'BOT' as source,
                        symbol, direction, entry_price, exit_price,
                        entry_rsi as rsi, entry_trend as trend,
                        entry_volatility, entry_volume_ratio,
                        pnl, roi, CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
                        entry_time as timestamp, exit_time as close_timestamp,
                        close_reason, ai_decision_id, ai_confidence
                    FROM bot_trades
                    WHERE is_simulated = 0 AND status = 'CLOSED' AND pnl IS NOT NULL
                """)
            
            if include_exchange:
                queries.append("""
                    SELECT 
                        'EXCHANGE' as source,
                        symbol, direction, entry_price, exit_price,
                        NULL as rsi, NULL as trend,
                        NULL as entry_volatility, NULL as entry_volume_ratio,
                        pnl, roi, CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
                        entry_time as timestamp, exit_time as close_timestamp,
                        NULL as close_reason, NULL as ai_decision_id, NULL as ai_confidence
                    FROM exchange_trades
                    WHERE pnl IS NOT NULL
                """)
            
            if not queries:
                return []
            
            # Объединяем запросы
            union_query = " UNION ALL ".join(queries)
            
            # Группируем по символам и фильтруем по минимальному количеству
            # ВАЖНО: Если min_trades=0, НЕ фильтруем по символам - возвращаем ВСЕ сделки
            if min_trades > 0:
                final_query = f"""
                    WITH all_trades AS ({union_query})
                    SELECT * FROM all_trades
                    WHERE symbol IN (
                        SELECT symbol FROM all_trades
                        GROUP BY symbol
                        HAVING COUNT(*) >= ?
                    )
                    ORDER BY timestamp DESC
                """
                params.append(min_trades)
            else:
                # min_trades=0 - возвращаем ВСЕ сделки без фильтрации по символам
                final_query = f"""
                    WITH all_trades AS ({union_query})
                    SELECT * FROM all_trades
                    ORDER BY timestamp DESC
                """
            
            if limit:
                final_query += " LIMIT ?"
                params.append(limit)
            
            conn.row_factory = sqlite3.Row
            cursor.execute(final_query, params)
            rows = cursor.fetchall()
            
            # Преобразуем Row в dict
            result = [dict(row) for row in rows]
            
            # КРИТИЧНО: Также загружаем сделки из bots_data.db -> bot_trades_history
            # Это история торговли ботов, которая теперь хранится в bots_data.db
            if include_real:
                try:
                    from bot_engine.bots_database import get_bots_database
                    bots_db = get_bots_database()
                    
                    # Загружаем закрытые сделки из bot_trades_history
                    bots_trades = bots_db.get_bot_trades_history(
                        status='CLOSED',
                        decision_source=None,  # Все источники
                        limit=None
                    )
                    
                    # Конвертируем формат в формат для обучения
                    for trade in bots_trades:
                        # Пропускаем симуляции
                        if trade.get('is_simulated'):
                            continue
                        
                        # Пропускаем если нет PnL
                        if trade.get('pnl') is None:
                            continue
                        
                        # Формируем запись в формате для обучения
                        converted_trade = {
                            'source': 'BOTS_HISTORY',
                            'symbol': trade.get('symbol', ''),
                            'direction': trade.get('direction', 'LONG'),
                            'entry_price': trade.get('entry_price', 0.0),
                            'exit_price': trade.get('exit_price'),
                            'rsi': trade.get('entry_rsi'),  # RSI на входе
                            'trend': trade.get('entry_trend'),  # Тренд на входе
                            'entry_volatility': trade.get('entry_volatility'),
                            'entry_volume_ratio': trade.get('entry_volume_ratio'),
                            'pnl': trade.get('pnl'),
                            'roi': trade.get('roi'),
                            'is_successful': 1 if trade.get('is_successful') else 0,
                            'timestamp': trade.get('entry_time') or trade.get('entry_timestamp'),
                            'close_timestamp': trade.get('exit_time') or trade.get('exit_timestamp'),
                            'close_reason': trade.get('close_reason'),
                            'ai_decision_id': trade.get('ai_decision_id'),
                            'ai_confidence': trade.get('ai_confidence')
                        }
                        
                        result.append(converted_trade)
                    
                    pass
                except Exception as e:
                    pass
            
            # ВАЖНО: НЕ загружаем closed_pnl из app_data.db для обучения!
            # Причина: в closed_pnl НЕТ RSI/тренда/волатильности, которые ОБЯЗАТЕЛЬНЫ для обучения ИИ
            # ИИ использует entry_rsi, entry_trend, entry_volatility для подготовки features
            # Без этих данных сделки будут пропущены (см. ai_trainer.py:1266 - if not entry_rsi: continue)
            # 
            # Для обучения используем только:
            # - simulated_trades (симуляции с полными данными)
            # - bot_trades (реальные сделки ботов с RSI/трендом)
            # - exchange_trades (сделки с биржи, но может не быть RSI)
            # - bot_trades_history (история торговли ботов из bots_data.db)
            
            # Фильтруем по min_trades если нужно (только для сделок из ai_data.db, bots_trades уже загружены)
            if min_trades > 0:
                # Группируем по символам
                symbol_counts = {}
                for trade in result:
                    symbol = trade.get('symbol', '')
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
                # Фильтруем только символы с >= min_trades
                result = [trade for trade in result if symbol_counts.get(trade.get('symbol', ''), 0) >= min_trades]
            
            # Сортируем по timestamp
            result.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
            
            # Применяем limit если указан
            if limit:
                result = result[:limit]
            
            pass
            return result
    
    def get_open_positions_for_ai(self) -> List[Dict[str, Any]]:
        """
        Получает открытые позиции из app_data.db и обогащает их данными для ИИ
        
        ВАЖНО: Открытые позиции используются для:
        1. Обучения ИИ (как примеры текущих сделок)
        2. Получения рекомендаций ИИ в реальном времени (точки выхода, стопы)
        
        Обогащает позиции данными из:
        - bots_data.db -> bots (entry_price, entry_time, entry_rsi, entry_trend)
        - bots_data.db -> rsi_cache_coins (текущий RSI, тренд)
        
        Returns:
            Список открытых позиций с полными данными для ИИ
        """
        try:
            from bot_engine.app_database import AppDatabase
            from bot_engine.bots_database import BotsDatabase
            
            app_db = AppDatabase()
            bots_db = BotsDatabase()
            
            # Загружаем открытые позиции из app_data.db
            positions_data = app_db.load_positions_data()
            all_positions = []
            
            for category in ['high_profitable', 'profitable', 'losing']:
                positions = positions_data.get(category, [])
                all_positions.extend(positions)
            
            if not all_positions:
                pass
                return []
            
            # Загружаем текущий RSI cache
            rsi_cache = bots_db.load_rsi_cache(max_age_hours=6.0)
            coins_rsi_data = rsi_cache.get('coins', {}) if rsi_cache else {}
            
            # Загружаем состояние ботов для получения entry_price, entry_time, entry_rsi
            bots_state = bots_db.load_bots_state()
            bots_data = bots_state.get('bots', {})
            
            enriched_positions = []
            for position in all_positions:
                symbol = position.get('symbol', '')
                if not symbol:
                    continue
                
                # Получаем данные бота для этой позиции
                bot_data = bots_data.get(symbol, {})
                
                # Получаем текущий RSI/тренд из cache
                coin_rsi_data = coins_rsi_data.get(symbol, {})
                # Получаем RSI и тренд с учетом текущего таймфрейма
                from bot_engine.bot_config import get_rsi_from_coin_data, get_trend_from_coin_data
                current_rsi = get_rsi_from_coin_data(coin_rsi_data)
                current_trend = get_trend_from_coin_data(coin_rsi_data)
                current_price = coin_rsi_data.get('price')
                
                # Получаем данные входа из бота
                entry_price = bot_data.get('entry_price') or position.get('entry_price')
                entry_time = bot_data.get('entry_time')
                entry_timestamp = bot_data.get('entry_timestamp')
                entry_rsi = bot_data.get('last_rsi')  # Последний RSI при входе
                entry_trend = bot_data.get('entry_trend') or bot_data.get('last_trend', 'NEUTRAL')
                position_side = bot_data.get('position_side') or position.get('side', 'LONG')
                
                # Если нет entry_price из бота, пытаемся вычислить из текущей цены и PnL
                if not entry_price and current_price and position.get('pnl') and position.get('size'):
                    # Приблизительный расчет entry_price из PnL
                    pnl = position.get('pnl', 0)
                    size = position.get('size', 0)
                    if size > 0:
                        if position_side == 'LONG':
                            entry_price = current_price - (pnl / size)
                        else:
                            entry_price = current_price + (pnl / size)
                
                # Формируем обогащенную позицию
                enriched_position = {
                    'symbol': symbol,
                    'position_side': position_side,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'entry_timestamp': entry_timestamp,
                    'entry_rsi': entry_rsi,
                    'entry_trend': entry_trend,
                    'current_price': current_price,
                    'current_rsi': current_rsi,
                    'current_trend': current_trend,
                    'pnl': position.get('pnl', 0),
                    'roi': position.get('roi', 0),
                    'max_profit': position.get('max_profit'),
                    'max_loss': position.get('max_loss'),
                    'size': position.get('size'),
                    'leverage': position.get('leverage', 1.0),
                    'position_category': position.get('position_category', category),
                    'high_roi': position.get('high_roi', False),
                    'high_loss': position.get('high_loss', False),
                    'last_update': position.get('last_update'),
                    'is_open': True,  # Маркер открытой позиции
                    'source': 'APP_POSITIONS'
                }
                
                enriched_positions.append(enriched_position)
            
            pass
            return enriched_positions
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки открытых позиций для ИИ: {e}")
            import traceback
            pass
            return []
    
    def analyze_patterns(self, 
                         symbol: Optional[str] = None,
                         rsi_range: Optional[Tuple[float, float]] = None,
                         min_trades: int = 10) -> List[Dict[str, Any]]:
        """
        Анализирует паттерны в сделках
        
        Args:
            symbol: Фильтр по символу
            rsi_range: Диапазон RSI (min, max)
            min_trades: Минимальное количество сделок для паттерна
        
        Returns:
            Список паттернов с метриками
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    symbol,
                    CASE 
                        WHEN entry_rsi <= 25 THEN '<=25'
                        WHEN entry_rsi <= 30 THEN '26-30'
                        WHEN entry_rsi <= 35 THEN '31-35'
                        WHEN entry_rsi >= 70 THEN '>=70'
                        WHEN entry_rsi >= 65 THEN '65-69'
                        ELSE 'OTHER'
                    END as rsi_range,
                    entry_trend as trend,
                    COUNT(*) as trade_count,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                    AVG(duration_candles) as avg_duration
                FROM simulated_trades
                WHERE entry_rsi IS NOT NULL
            """
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if rsi_range:
                query += " AND entry_rsi >= ? AND entry_rsi <= ?"
                params.extend(rsi_range)
            
            query += """
                GROUP BY symbol, rsi_range, trend
                HAVING trade_count >= ?
                ORDER BY win_rate DESC, avg_pnl DESC
            """
            params.append(min_trades)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_ai_decision_performance(self, 
                                    symbol: Optional[str] = None,
                                    min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Анализирует производительность решений AI
        
        Returns:
            Статистика по решениям AI
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_decisions,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN result_successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(result_pnl) as avg_pnl,
                    COUNT(DISTINCT symbol) as symbols_count
                FROM ai_decisions
                WHERE result_pnl IS NOT NULL
            """
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if min_confidence:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            
            cursor.execute(query, params)
            result = dict(cursor.fetchone())
            
            return result
    
    def get_training_statistics(self, session_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Получает статистику по сессиям обучения"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM training_sessions WHERE 1=1"
            params = []
            
            if session_type:
                query += " AND session_type = ?"
                params.append(session_type)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                session = dict(row)
                if session.get('metadata_json'):
                    session['metadata'] = json.loads(session['metadata_json'])
                result.append(session)
            
            return result
    
    def save_parameter_training_sample(self, sample: Dict[str, Any]) -> Optional[int]:
        """
        Сохраняет образец для обучения предсказателя качества параметров
        
        Args:
            sample: Словарь с данными образца:
                - rsi_params: Dict - параметры RSI
                - risk_params: Optional[Dict] - параметры риск-менеджмента
                - win_rate: float - Win Rate (0-100)
                - total_pnl: float - Total PnL
                - trades_count: int - Количество сделок
                - quality: float - Качество (вычисленное)
                - blocked: bool - Были ли входы заблокированы
                - rsi_entered_zones: int - Сколько раз RSI входил в зоны
                - filters_blocked: int - Сколько раз фильтры заблокировали вход
                - block_reasons: Optional[Dict] - Причины блокировок
                - symbol: Optional[str] - Символ монеты
        
        Returns:
            ID сохраненного образца или None при ошибке
        """
        try:
            now = datetime.now().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Извлекаем RSI параметры
                rsi_params = sample.get('rsi_params', {})
                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                
                # Собираем остальные RSI параметры
                extra_rsi = {}
                known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                for key, value in rsi_params.items():
                    if key not in known_rsi:
                        extra_rsi[key] = value
                extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                
                # Извлекаем Risk параметры
                risk_params = sample.get('risk_params', {})
                max_loss = risk_params.get('max_loss_percent')
                take_profit = risk_params.get('take_profit_percent')
                trailing_activation = risk_params.get('trailing_stop_activation')
                trailing_distance = risk_params.get('trailing_stop_distance')
                trailing_take = risk_params.get('trailing_take_distance')
                trailing_interval = risk_params.get('trailing_update_interval')
                break_even_trigger = risk_params.get('break_even_trigger')
                break_even_protection = risk_params.get('break_even_protection')
                max_hours = risk_params.get('max_position_hours')
                
                # Собираем остальные Risk параметры
                extra_risk = {}
                known_risk = {'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours'}
                for key, value in risk_params.items():
                    if key not in known_risk:
                        extra_risk[key] = value
                extra_risk_json = json.dumps(extra_risk, ensure_ascii=False) if extra_risk else None
                
                cursor.execute("""
                    INSERT INTO parameter_training_samples (
                        rsi_long_threshold, rsi_short_threshold,
                        rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                        rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                        max_loss_percent, take_profit_percent,
                        trailing_stop_activation, trailing_stop_distance,
                        trailing_take_distance, trailing_update_interval,
                        break_even_trigger, break_even_protection, max_position_hours,
                        win_rate, total_pnl, trades_count, quality, blocked,
                        rsi_entered_zones, filters_blocked,
                        block_reasons_json, extra_rsi_params_json, extra_risk_params_json,
                        symbol, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                    rsi_exit_short_with, rsi_exit_short_against,
                    max_loss, take_profit, trailing_activation, trailing_distance,
                    trailing_take, trailing_interval, break_even_trigger,
                    break_even_protection, max_hours,
                    sample.get('win_rate', 0.0),
                    sample.get('total_pnl', 0.0),
                    sample.get('trades_count', 0),
                    sample.get('quality', 0.0),
                    1 if sample.get('blocked', False) else 0,
                    sample.get('rsi_entered_zones', 0),
                    sample.get('filters_blocked', 0),
                    json.dumps(sample.get('block_reasons', {}), ensure_ascii=False) if sample.get('block_reasons') else None,
                    extra_rsi_json, extra_risk_json,
                    sample.get('symbol'),
                    now
                ))
                sample_id = cursor.lastrowid
                conn.commit()
                return sample_id
        except MemoryError:
            # КРИТИЧНО: Не логируем при MemoryError (это вызывает рекурсию)
            # Просто возвращаем None - graceful degradation
            return None
        except Exception as e:
            # Используем безопасное логирование
            try:
                logger.error(f"❌ Ошибка сохранения образца параметров: {e}")
            except MemoryError:
                # Не логируем при MemoryError
                pass
            return None
    
    def get_parameter_training_samples(self, limit: Optional[int] = None, 
                                       order_by: str = 'created_at DESC') -> List[Dict[str, Any]]:
        """
        Получает образцы для обучения предсказателя качества параметров
        
        Args:
            limit: Максимальное количество образцов (None = все)
            order_by: Поле для сортировки (по умолчанию: created_at DESC)
        
        Returns:
            Список словарей с данными образцов
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = f"SELECT * FROM parameter_training_samples ORDER BY {order_by}"
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                samples = []
                for row in rows:
                    # Преобразуем Row в dict для удобства
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных столбцов
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Загружаем extra_rsi_params_json если есть
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    # Восстанавливаем risk_params из нормализованных столбцов
                    risk_params = {}
                    if row_dict.get('max_loss_percent') is not None:
                        risk_params['max_loss_percent'] = row_dict['max_loss_percent']
                    if row_dict.get('take_profit_percent') is not None:
                        risk_params['take_profit_percent'] = row_dict['take_profit_percent']
                    if row_dict.get('trailing_stop_activation') is not None:
                        risk_params['trailing_stop_activation'] = row_dict['trailing_stop_activation']
                    if row_dict.get('trailing_stop_distance') is not None:
                        risk_params['trailing_stop_distance'] = row_dict['trailing_stop_distance']
                    if row_dict.get('trailing_take_distance') is not None:
                        risk_params['trailing_take_distance'] = row_dict['trailing_take_distance']
                    if row_dict.get('trailing_update_interval') is not None:
                        risk_params['trailing_update_interval'] = row_dict['trailing_update_interval']
                    if row_dict.get('break_even_trigger') is not None:
                        risk_params['break_even_trigger'] = row_dict['break_even_trigger']
                    if row_dict.get('break_even_protection') is not None:
                        risk_params['break_even_protection'] = row_dict['break_even_protection']
                    if row_dict.get('max_position_hours') is not None:
                        risk_params['max_position_hours'] = row_dict['max_position_hours']
                    
                    # Загружаем extra_risk_params_json если есть
                    if row_dict.get('extra_risk_params_json'):
                        try:
                            extra_risk = json.loads(row_dict['extra_risk_params_json'])
                            risk_params.update(extra_risk)
                        except:
                            pass
                    
                    sample = {
                        'id': row_dict['id'],
                        'rsi_params': rsi_params,
                        'risk_params': risk_params,
                        'win_rate': row_dict['win_rate'],
                        'total_pnl': row_dict['total_pnl'],
                        'trades_count': row_dict['trades_count'],
                        'quality': row_dict['quality'],
                        'blocked': bool(row_dict['blocked']),
                        'rsi_entered_zones': row_dict['rsi_entered_zones'],
                        'filters_blocked': row_dict['filters_blocked'],
                        'block_reasons': json.loads(row_dict['block_reasons_json']) if row_dict.get('block_reasons_json') else {},
                        'symbol': row_dict['symbol'],
                        'timestamp': row_dict['created_at']
                    }
                    samples.append(sample)
                
                return samples
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки образцов параметров: {e}")
            return []
    
    def count_parameter_training_samples(self) -> int:
        """Возвращает количество сохраненных образцов параметров"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM parameter_training_samples")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Ошибка подсчета образцов параметров: {e}")
            return 0
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ИСПОЛЬЗОВАННЫМИ ПАРАМЕТРАМИ ====================
    
    def save_used_training_parameter(self, param_hash: str, rsi_params: Dict, training_seed: int,
                                     win_rate: float = 0.0, total_pnl: float = 0.0,
                                     signal_accuracy: float = 0.0, trades_count: int = 0,
                                     rating: float = 0.0, symbol: Optional[str] = None) -> Optional[int]:
        """
        Сохраняет или обновляет использованные параметры обучения
        
        Returns:
            ID записи или None при ошибке
        """
        try:
            now = datetime.now().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Используем INSERT OR REPLACE для атомарной операции (быстрее чем SELECT + UPDATE)
                # Но сначала проверяем рейтинг, чтобы обновлять только если лучше
                cursor.execute("SELECT rating FROM used_training_parameters WHERE param_hash = ?", (param_hash,))
                existing = cursor.fetchone()
                
                if existing and rating <= existing['rating']:
                    # Не обновляем если рейтинг не лучше
                    cursor.execute("SELECT id FROM used_training_parameters WHERE param_hash = ?", (param_hash,))
                    return cursor.fetchone()['id']
                
                # Извлекаем RSI параметры
                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                
                # Собираем остальные RSI параметры
                extra_rsi = {}
                known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                for key, value in rsi_params.items():
                    if key not in known_rsi:
                        extra_rsi[key] = value
                extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                
                # Обновляем или вставляем
                cursor.execute("""
                    INSERT INTO used_training_parameters (
                        param_hash, rsi_long_threshold, rsi_short_threshold,
                        rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                        rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                        extra_rsi_params_json, training_seed, win_rate,
                        total_pnl, signal_accuracy, trades_count, rating, symbol, used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(param_hash) DO UPDATE SET
                        rsi_long_threshold = excluded.rsi_long_threshold,
                        rsi_short_threshold = excluded.rsi_short_threshold,
                        rsi_exit_long_with_trend = excluded.rsi_exit_long_with_trend,
                        rsi_exit_long_against_trend = excluded.rsi_exit_long_against_trend,
                        rsi_exit_short_with_trend = excluded.rsi_exit_short_with_trend,
                        rsi_exit_short_against_trend = excluded.rsi_exit_short_against_trend,
                        extra_rsi_params_json = excluded.extra_rsi_params_json,
                        training_seed = excluded.training_seed,
                        win_rate = excluded.win_rate,
                        total_pnl = excluded.total_pnl,
                        signal_accuracy = excluded.signal_accuracy,
                        trades_count = excluded.trades_count,
                        rating = excluded.rating,
                        symbol = excluded.symbol,
                        used_at = excluded.used_at,
                        update_count = update_count + 1
                    WHERE excluded.rating > used_training_parameters.rating
                """, (
                    param_hash, rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                    rsi_exit_short_with, rsi_exit_short_against, extra_rsi_json,
                    training_seed, win_rate, total_pnl, signal_accuracy, trades_count, rating, symbol, now
                ))
                param_id = cursor.lastrowid
                conn.commit()
                return param_id
        except Exception as e:
            pass
            return None
    
    def get_used_training_parameter(self, param_hash: str) -> Optional[Dict[str, Any]]:
        """Получает использованные параметры по хешу"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM used_training_parameters WHERE param_hash = ?", (param_hash,))
                row = cursor.fetchone()
                if row:
                    # Конвертируем sqlite3.Row в словарь для работы с .get()
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных столбцов
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Загружаем extra_rsi_params_json если есть
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    return {
                        'id': row_dict['id'],
                        'param_hash': row_dict['param_hash'],
                        'rsi_params': rsi_params,
                        'training_seed': row_dict['training_seed'],
                        'win_rate': row_dict['win_rate'],
                        'total_pnl': row_dict['total_pnl'],
                        'signal_accuracy': row_dict['signal_accuracy'],
                        'trades_count': row_dict['trades_count'],
                        'rating': row_dict['rating'],
                        'symbol': row_dict['symbol'],
                        'used_at': row_dict['used_at'],
                        'update_count': row_dict['update_count']
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки использованных параметров: {e}")
            return None
    
    def count_used_training_parameters(self) -> int:
        """Возвращает количество использованных параметров"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM used_training_parameters")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Ошибка подсчета использованных параметров: {e}")
            return 0
    
    def get_best_used_parameters(self, limit: int = 10, min_win_rate: float = 80.0) -> List[Dict[str, Any]]:
        """Получает лучшие использованные параметры"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM used_training_parameters
                    WHERE win_rate >= ?
                    ORDER BY rating DESC
                    LIMIT ?
                """, (min_win_rate, limit))
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    # Конвертируем sqlite3.Row в словарь для работы с .get()
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных столбцов
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Загружаем extra_rsi_params_json если есть
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    result.append({
                        'rsi_params': rsi_params,
                        'training_seed': row_dict['training_seed'],
                        'win_rate': row_dict['win_rate'],
                        'total_pnl': row_dict['total_pnl'],
                        'signal_accuracy': row_dict['signal_accuracy'],
                        'trades_count': row_dict['trades_count'],
                        'rating': row_dict['rating'],
                        'symbol': row_dict['symbol'],
                        'used_at': row_dict['used_at']
                    })
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки лучших параметров: {e}")
            return []
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ЛУЧШИМИ ПАРАМЕТРАМИ ДЛЯ МОНЕТ ====================
    
    def save_best_params_for_symbol(self, symbol: str, rsi_params: Dict, rating: float,
                                    win_rate: float, total_pnl: float) -> Optional[int]:
        """Сохраняет или обновляет лучшие параметры для монеты с нормализованными полями"""
        try:
            now = datetime.now().isoformat()
            with self.lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Извлекаем RSI параметры
                    rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                    rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                    rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                    rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                    rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                    rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                    
                    # Собираем остальные RSI параметры в extra_rsi_params_json
                    extra_rsi = {}
                    known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                                'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold',
                                'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                                'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                    for key, value in rsi_params.items():
                        if key not in known_rsi:
                            extra_rsi[key] = value
                    
                    extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                    
                    # Сохраняем полный JSON для обратной совместимости
                    rsi_params_json = json.dumps(rsi_params, ensure_ascii=False)
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO best_params_per_symbol (
                            symbol, rsi_long_threshold, rsi_short_threshold,
                            rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                            rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                            extra_rsi_params_json, rating, win_rate, total_pnl, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                        rsi_exit_short_with, rsi_exit_short_against, extra_rsi_json,
                        rating, win_rate, total_pnl, now
                    ))
                    param_id = cursor.lastrowid
                    conn.commit()
                    return param_id
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения лучших параметров для {symbol}: {e}")
            return None
    
    def get_best_params_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получает лучшие параметры для монеты, восстанавливая структуру rsi_params"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM best_params_per_symbol WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row:
                    # Преобразуем Row в dict для удобства
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных полей
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Добавляем extra_rsi_params
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    return {
                        'symbol': row_dict['symbol'],
                        'rsi_params': rsi_params,
                        'rating': row_dict['rating'],
                        'win_rate': row_dict['win_rate'],
                        'total_pnl': row_dict['total_pnl'],
                        'updated_at': row_dict['updated_at']
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки лучших параметров для {symbol}: {e}")
            return None
    
    def get_all_best_params_per_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Получает лучшие параметры для всех монет, восстанавливая структуру rsi_params"""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM best_params_per_symbol")
                rows = cursor.fetchall()
                result = {}
                for row in rows:
                    # Преобразуем Row в dict для удобства
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных полей
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Добавляем extra_rsi_params
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    result[row_dict['symbol']] = {
                        'rsi_params': rsi_params,
                        'rating': row_dict['rating'],
                        'win_rate': row_dict['win_rate'],
                        'total_pnl': row_dict['total_pnl'],
                        'updated_at': row_dict['updated_at']
                    }
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки лучших параметров: {e}")
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ЗАБЛОКИРОВАННЫМИ ПАРАМЕТРАМИ ====================
    
    def save_blocked_params(self, rsi_params: Dict, block_reasons: Optional[Dict] = None,
                           symbol: Optional[str] = None, blocked_attempts: int = 0,
                           blocked_long: int = 0, blocked_short: int = 0) -> Optional[int]:
        """Сохраняет заблокированные параметры"""
        try:
            now = datetime.now().isoformat()
            # Вычисляем hash параметров для уникальности
            import hashlib
            params_str = json.dumps(rsi_params, sort_keys=True, ensure_ascii=False)
            param_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Извлекаем RSI параметры
                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold')
                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold')
                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend')
                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend')
                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend')
                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend')
                
                # Собираем остальные RSI параметры
                extra_rsi = {}
                known_rsi = {'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend', 'rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend'}
                for key, value in rsi_params.items():
                    if key not in known_rsi:
                        extra_rsi[key] = value
                extra_rsi_json = json.dumps(extra_rsi, ensure_ascii=False) if extra_rsi else None
                
                # Используем INSERT OR IGNORE чтобы не дублировать одинаковые параметры
                cursor.execute("""
                    INSERT OR IGNORE INTO blocked_params (
                        param_hash, rsi_long_threshold, rsi_short_threshold,
                        rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                        rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                        extra_rsi_params_json, block_reasons_json, 
                        blocked_attempts, blocked_long, blocked_short,
                        symbol, blocked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    param_hash,
                    rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                    rsi_exit_short_with, rsi_exit_short_against, extra_rsi_json,
                    json.dumps(block_reasons, ensure_ascii=False) if block_reasons else None,
                    blocked_attempts,
                    blocked_long,
                    blocked_short,
                    symbol,
                    now
                ))
                param_id = cursor.lastrowid
                conn.commit()
                return param_id
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения заблокированных параметров: {e}")
            return None
    
    def get_blocked_params(self, limit: Optional[int] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получает заблокированные параметры"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM blocked_params WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY blocked_at DESC"
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    # Конвертируем sqlite3.Row в словарь для работы с .get()
                    row_dict = dict(row)
                    
                    # Восстанавливаем rsi_params из нормализованных столбцов
                    rsi_params = {}
                    if row_dict.get('rsi_long_threshold') is not None:
                        rsi_params['oversold'] = row_dict['rsi_long_threshold']
                        rsi_params['rsi_long_threshold'] = row_dict['rsi_long_threshold']
                    if row_dict.get('rsi_short_threshold') is not None:
                        rsi_params['overbought'] = row_dict['rsi_short_threshold']
                        rsi_params['rsi_short_threshold'] = row_dict['rsi_short_threshold']
                    if row_dict.get('rsi_exit_long_with_trend') is not None:
                        rsi_params['exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                        rsi_params['rsi_exit_long_with_trend'] = row_dict['rsi_exit_long_with_trend']
                    if row_dict.get('rsi_exit_long_against_trend') is not None:
                        rsi_params['exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                        rsi_params['rsi_exit_long_against_trend'] = row_dict['rsi_exit_long_against_trend']
                    if row_dict.get('rsi_exit_short_with_trend') is not None:
                        rsi_params['exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                        rsi_params['rsi_exit_short_with_trend'] = row_dict['rsi_exit_short_with_trend']
                    if row_dict.get('rsi_exit_short_against_trend') is not None:
                        rsi_params['exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                        rsi_params['rsi_exit_short_against_trend'] = row_dict['rsi_exit_short_against_trend']
                    
                    # Загружаем extra_rsi_params_json если есть
                    if row_dict.get('extra_rsi_params_json'):
                        try:
                            extra_rsi = json.loads(row_dict['extra_rsi_params_json'])
                            rsi_params.update(extra_rsi)
                        except:
                            pass
                    
                    result.append({
                        'rsi_params': rsi_params,
                        'block_reasons': json.loads(row_dict['block_reasons_json']) if row_dict.get('block_reasons_json') else {},
                        'blocked_at': row_dict['blocked_at'],
                        'blocked_attempts': row_dict.get('blocked_attempts', 0),
                        'blocked_long': row_dict.get('blocked_long', 0),
                        'blocked_short': row_dict.get('blocked_short', 0),
                        'symbol': row_dict.get('symbol'),
                        'timestamp': row_dict.get('blocked_at')  # Для совместимости
                    })
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки заблокированных параметров: {e}")
            return []
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ЦЕЛЕВЫМИ ЗНАЧЕНИЯМИ WIN RATE ====================
    
    def save_win_rate_target(self, symbol: str, target_win_rate: float,
                             current_win_rate: Optional[float] = None) -> Optional[int]:
        """Сохраняет или обновляет целевое значение win rate для монеты"""
        try:
            now = datetime.now().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO win_rate_targets (
                        symbol, target_win_rate, current_win_rate, updated_at
                    ) VALUES (?, ?, ?, ?)
                """, (symbol, target_win_rate, current_win_rate, now))
                target_id = cursor.lastrowid
                conn.commit()
                return target_id
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения целевого win rate для {symbol}: {e}")
            return None
    
    def get_win_rate_target(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получает целевое значение win rate для монеты"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM win_rate_targets WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row:
                    return {
                        'symbol': row['symbol'],
                        'target_win_rate': row['target_win_rate'],
                        'current_win_rate': row['current_win_rate'],
                        'updated_at': row['updated_at']
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки целевого win rate для {symbol}: {e}")
            return None
    
    def get_all_win_rate_targets(self) -> Dict[str, Dict[str, Any]]:
        """Получает все целевые значения win rate"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM win_rate_targets")
                rows = cursor.fetchall()
                result = {}
                for row in rows:
                    result[row['symbol']] = {
                        'target_win_rate': row['target_win_rate'],
                        'current_win_rate': row['current_win_rate'],
                        'updated_at': row['updated_at']
                    }
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки целевых win rate: {e}")
            return {}
    
    # ==================== МЕТОДЫ ДЛЯ КООРДИНАЦИИ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ====================
    
    def try_lock_symbol(self, symbol: str, process_id: str, hostname: str = None, 
                        lock_duration_minutes: int = 60) -> bool:
        """
        Пытается заблокировать символ для обработки (для параллельной работы на разных ПК)
        
        Args:
            symbol: Символ монеты
            process_id: Уникальный ID процесса (например, PID + timestamp)
            hostname: Имя хоста (опционально)
            lock_duration_minutes: Длительность блокировки в минутах
        
        Returns:
            True если удалось заблокировать, False если уже заблокирован
        """
        try:
            now = datetime.now()
            expires_at = now.replace(second=0, microsecond=0)
            from datetime import timedelta
            expires_at += timedelta(minutes=lock_duration_minutes)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Очищаем истекшие блокировки
                cursor.execute("""
                    DELETE FROM training_locks 
                    WHERE expires_at < ?
                """, (now.isoformat(),))
                
                # Пытаемся заблокировать
                try:
                    cursor.execute("""
                        INSERT INTO training_locks (
                            symbol, process_id, hostname, locked_at, expires_at, status
                        ) VALUES (?, ?, ?, ?, ?, 'PROCESSING')
                    """, (
                        symbol, process_id, hostname, now.isoformat(), expires_at.isoformat()
                    ))
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    # Символ уже заблокирован
                    return False
        except Exception as e:
            pass
            return False
    
    def release_lock(self, symbol: str, process_id: str) -> bool:
        """
        Освобождает блокировку символа
        
        Args:
            symbol: Символ монеты
            process_id: ID процесса, который блокировал
        
        Returns:
            True если удалось освободить
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM training_locks 
                    WHERE symbol = ? AND process_id = ?
                """, (symbol, process_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            pass
            return False
    
    def get_available_symbols(self, all_symbols: List[str], process_id: str, 
                             hostname: str = None) -> List[str]:
        """
        Получает список доступных символов (не заблокированных другими процессами)
        
        Args:
            all_symbols: Все символы для обработки
            process_id: ID текущего процесса
            hostname: Имя хоста (опционально)
        
        Returns:
            Список доступных символов
        """
        try:
            now = datetime.now()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Очищаем истекшие блокировки
                cursor.execute("""
                    DELETE FROM training_locks 
                    WHERE expires_at < ?
                """, (now.isoformat(),))
                conn.commit()
                
                # Получаем заблокированные символы
                cursor.execute("SELECT symbol FROM training_locks")
                locked_symbols = {row[0] for row in cursor.fetchall()}
                
                # Возвращаем только незаблокированные
                available = [s for s in all_symbols if s not in locked_symbols]
                return available
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения доступных символов: {e}")
            return all_symbols  # В случае ошибки возвращаем все
    
    def extend_lock(self, symbol: str, process_id: str, 
                   additional_minutes: int = 30) -> bool:
        """
        Продлевает блокировку символа
        
        Args:
            symbol: Символ монеты
            process_id: ID процесса
            additional_minutes: Сколько минут добавить
        
        Returns:
            True если удалось продлить
        """
        try:
            from datetime import timedelta
            now = datetime.now()
            new_expires_at = now + timedelta(minutes=additional_minutes)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE training_locks 
                    SET expires_at = ?
                    WHERE symbol = ? AND process_id = ?
                """, (new_expires_at.isoformat(), symbol, process_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            pass
            return False
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ИСТОРИЕЙ СВЕЧЕЙ ====================
    
    def save_candles(self, symbol: str, candles: List[Dict], timeframe: str = '6h') -> int:
        """
        Сохраняет свечи для символа в БД с ограничением количества
        
        Args:
            symbol: Символ монеты
            candles: Список свечей [{'time': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}, ...]
            timeframe: Таймфрейм (по умолчанию '6h')
        
        Returns:
            Количество сохраненных свечей
        """
        if not candles:
            return 0
        
        try:
            now = datetime.now().isoformat()
            saved_count = 0
            
            # ОГРАНИЧЕНИЕ: Максимум 1000 свечей на символ для предотвращения раздувания БД
            # 1000 свечей = ~250 дней истории - более чем достаточно (запрашивается только 30 дней)
            MAX_CANDLES_PER_SYMBOL = 1000
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # ⚠️ КРИТИЧНО: Кэш должен ПОЛНОСТЬЮ ПЕРЕЗАПИСЫВАТЬСЯ, а не накапливаться!
                # ВСЕГДА удаляем ВСЕ старые свечи для этого символа перед вставкой новых
                # Это гарантирует, что старые неиспользуемые данные всегда удаляются
                cursor.execute("""
                    DELETE FROM candles_history 
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                deleted_old_count = cursor.rowcount
                
                # Сортируем свечи по времени и берем только последние MAX_CANDLES_PER_SYMBOL
                candles_sorted = sorted(candles, key=lambda x: x.get('time', 0))
                candles_to_save = candles_sorted[-MAX_CANDLES_PER_SYMBOL:]
                
                if len(candles_sorted) > MAX_CANDLES_PER_SYMBOL:
                    pass
                
                # ⚡ ОПТИМИЗИРОВАННАЯ ВСТАВКА: используем executemany вместо цикла
                # Вставляем только новые свечи (старые уже удалены)
                if candles_to_save:
                    cursor.executemany("""
                        INSERT INTO candles_history (
                            symbol, timeframe, candle_time, open_price, high_price,
                            low_price, close_price, volume, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        (
                            symbol, timeframe,
                            int(candle['time']),
                            float(candle['open']),
                            float(candle['high']),
                            float(candle['low']),
                            float(candle['close']),
                            float(candle['volume']),
                            now
                        )
                        for candle in candles_to_save
                    ])
                    saved_count = cursor.rowcount
                else:
                    saved_count = 0
                
                conn.commit()
            return saved_count
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения свечей для {symbol}: {e}")
            return 0
    
    def save_candles_batch(self, candles_data: Dict[str, List[Dict]], timeframe: str = '6h') -> Dict[str, int]:
        """
        Сохраняет свечи для нескольких символов (батч операция)
        
        Args:
            candles_data: Словарь {symbol: [candles]}
            timeframe: Таймфрейм
        
        Returns:
            Словарь {symbol: saved_count}
        """
        # ⚡ ОПТИМИЗАЦИЯ: При батч-сохранении используем TRUNCATE-подход для всех символов в батче
        # Удаляем старые свечи для всех символов из батча одним запросом, затем вставляем новые
        if not candles_data:
            return {}
        
        try:
            now = datetime.now().isoformat()
            MAX_CANDLES_PER_SYMBOL = 1000
            saved_counts = {}
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # ⚠️ КРИТИЧНО: Удаляем ВСЕ старые свечи для символов из батча одним запросом
                # Используем DELETE, так как candles_history - это история, а не кэш, и там могут быть данные для других символов
                # Но для гарантированной очистки используем более агрессивный подход
                symbols_list = list(candles_data.keys())
                placeholders = ','.join(['?'] * len(symbols_list))
                
                # Проверяем количество перед удалением
                cursor.execute(f"SELECT COUNT(*) FROM candles_history WHERE symbol IN ({placeholders}) AND timeframe = ?", symbols_list + [timeframe])
                old_count = cursor.fetchone()[0]
                
                # Удаляем старые свечи для символов из батча
                cursor.execute(f"""
                    DELETE FROM candles_history 
                    WHERE symbol IN ({placeholders}) AND timeframe = ?
                """, symbols_list + [timeframe])
                deleted_total = cursor.rowcount
                
                # ⚠️ КРИТИЧНО: Проверяем, что DELETE действительно удалил все записи
                cursor.execute(f"SELECT COUNT(*) FROM candles_history WHERE symbol IN ({placeholders}) AND timeframe = ?", symbols_list + [timeframe])
                count_after_delete = cursor.fetchone()[0]
                
                if count_after_delete > 0:
                    logger.warning(f"⚠️ DELETE не удалил все записи! Осталось {count_after_delete:,} записей. Пытаемся удалить еще раз...")
                    # Пытаемся удалить еще раз
                    cursor.execute(f"DELETE FROM candles_history WHERE symbol IN ({placeholders}) AND timeframe = ?", symbols_list + [timeframe])
                    cursor.execute(f"SELECT COUNT(*) FROM candles_history WHERE symbol IN ({placeholders}) AND timeframe = ?", symbols_list + [timeframe])
                    final_count = cursor.fetchone()[0]
                    if final_count > 0:
                        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА! После повторного DELETE осталось {final_count:,} записей для символов из батча!")
                
                if old_count > 0:
                    pass
                
                # Собираем все свечи для пакетной вставки
                all_candles_to_insert = []
                
                for symbol, candles in candles_data.items():
                    if not candles:
                        saved_counts[symbol] = 0
                        continue
                    
                    # Сортируем свечи по времени и берем только последние MAX_CANDLES_PER_SYMBOL
                    candles_sorted = sorted(candles, key=lambda x: x.get('time', 0))
                    candles_to_save = candles_sorted[-MAX_CANDLES_PER_SYMBOL:]
                    
                    if len(candles_sorted) > MAX_CANDLES_PER_SYMBOL:
                        pass
                    
                    # Добавляем свечи в общий список для пакетной вставки
                    for candle in candles_to_save:
                        all_candles_to_insert.append((
                            symbol, timeframe,
                            int(candle.get('time', 0)),
                            float(candle.get('open', 0)),
                            float(candle.get('high', 0)),
                            float(candle.get('low', 0)),
                            float(candle.get('close', 0)),
                            float(candle.get('volume', 0)),
                            now
                        ))
                    
                    saved_counts[symbol] = len(candles_to_save)
                
                # ⚡ ОПТИМИЗИРОВАННАЯ ПАКЕТНАЯ ВСТАВКА: вставляем все свечи одним запросом
                if all_candles_to_insert:
                    cursor.executemany("""
                        INSERT INTO candles_history (
                            symbol, timeframe, candle_time, open_price, high_price,
                            low_price, close_price, volume, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, all_candles_to_insert)
                    inserted_total = cursor.rowcount
                    pass
                
                conn.commit()
            
            return saved_counts
            
        except Exception as e:
            logger.error(f"❌ Ошибка батч-сохранения свечей: {e}")
            import traceback
            pass
            return {}
    
    def get_candles(self, symbol: str, timeframe: str = '6h', 
                    limit: Optional[int] = None,
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> List[Dict]:
        """
        Получает свечи для символа
        
        Args:
            symbol: Символ монеты
            timeframe: Таймфрейм
            limit: Максимальное количество свечей
            start_time: Начальное время (timestamp)
            end_time: Конечное время (timestamp)
        
        Returns:
            Список свечей [{'time': int, 'open': float, ...}, ...]
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT candle_time, open_price, high_price, low_price, close_price, volume
                    FROM candles_history
                    WHERE symbol = ? AND timeframe = ?
                """
                params = [symbol, timeframe]
                
                if start_time:
                    query += " AND candle_time >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND candle_time <= ?"
                    params.append(end_time)
                
                query += " ORDER BY candle_time ASC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                candles = []
                for row in rows:
                    candles.append({
                        'time': row['candle_time'],
                        'open': row['open_price'],
                        'high': row['high_price'],
                        'low': row['low_price'],
                        'close': row['close_price'],
                        'volume': row['volume']
                    })
                
                return candles
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки свечей для {symbol}: {e}")
            return []
    
    def get_all_candles_dict(self, timeframe: str = '6h', max_symbols: int = 50, max_candles_per_symbol: int = 1000) -> Dict[str, List[Dict]]:
        """
        Получает свечи для символов из БД (таблица candles_history)
        
        ВАЖНО: Ограничения по умолчанию для предотвращения переполнения памяти!
        
        Args:
            timeframe: Таймфрейм
            max_symbols: Максимальное количество символов (по умолчанию 50 для экономии памяти)
            max_candles_per_symbol: Максимальное количество свечей на символ (по умолчанию 1000)
        
        Returns:
            Словарь {symbol: [candles]} (только последние свечи для каждого символа)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Получаем список символов с ограничением (если max_symbols > 0)
                if max_symbols > 0:
                    cursor.execute("""
                        SELECT DISTINCT symbol
                        FROM candles_history
                        WHERE timeframe = ?
                        ORDER BY symbol
                        LIMIT ?
                    """, (timeframe, max_symbols))
                else:
                    # Без ограничения - загружаем все символы
                    cursor.execute("""
                        SELECT DISTINCT symbol
                        FROM candles_history
                        WHERE timeframe = ?
                        ORDER BY symbol
                    """, (timeframe,))
                symbols = [row[0] for row in cursor.fetchall()]
                
                if not symbols:
                    pass
                    return {}
                
                result = {}
                for symbol in symbols:
                    # Получаем только последние N свечей для каждого символа
                    cursor.execute("""
                        SELECT candle_time, open_price, high_price, low_price, close_price, volume
                        FROM candles_history
                        WHERE timeframe = ? AND symbol = ?
                        ORDER BY candle_time DESC
                        LIMIT ?
                    """, (timeframe, symbol, max_candles_per_symbol))
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Разворачиваем обратно (от старых к новым)
                        candles = []
                        for row in reversed(rows):
                            if hasattr(row, 'keys'):
                                row_dict = dict(row)
                            else:
                                row_dict = {
                                    'candle_time': row[0],
                                    'open_price': row[1],
                                    'high_price': row[2],
                                    'low_price': row[3],
                                    'close_price': row[4],
                                    'volume': row[5]
                                }
                            
                            candles.append({
                                'time': row_dict['candle_time'],
                                'open': row_dict['open_price'],
                                'high': row_dict['high_price'],
                                'low': row_dict['low_price'],
                                'close': row_dict['close_price'],
                                'volume': row_dict['volume']
                            })
                        
                        result[symbol] = candles
                
                total_candles = sum(len(c) for c in result.values())
                pass
                return result
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки всех свечей: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def count_candles(self, symbol: Optional[str] = None, timeframe: str = '6h') -> int:
        """Подсчитывает количество свечей"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if symbol:
                    cursor.execute("SELECT COUNT(*) FROM candles_history WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
                else:
                    cursor.execute("SELECT COUNT(*) FROM candles_history WHERE timeframe = ?", (timeframe,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Ошибка подсчета свечей: {e}")
            return 0
    
    def count_symbols_with_candles(self, timeframe: str = '6h') -> int:
        """Подсчитывает количество уникальных символов со свечами"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM candles_history WHERE timeframe = ?", (timeframe,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Ошибка подсчета символов: {e}")
            return 0
    
    def get_candles_last_time(self, symbol: str, timeframe: str = '6h') -> Optional[int]:
        """Получает время последней свечи для символа"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(candle_time) as last_time
                    FROM candles_history
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                row = cursor.fetchone()
                return row['last_time'] if row and row['last_time'] else None
        except Exception as e:
            logger.error(f"❌ Ошибка получения последнего времени для {symbol}: {e}")
            return None
    
    # ==================== МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ БОТОВ ====================
    
    def save_bots_data_snapshot(self, bots_data: Dict) -> int:
        """
        ВАЖНО: Метод больше не сохраняет снапшоты!
        
        Снапшоты - это избыточное дублирование данных.
        Все данные ботов уже есть в нормализованных таблицах:
        - bots_data.db → bots (текущее состояние ботов)
        - bots_data.db → rsi_cache_coins (RSI данные)
        
        Args:
            bots_data: Словарь с данными ботов (игнорируется)
        
        Returns:
            0 (не сохраняем)
        """
        # Не сохраняем снапшоты - данные уже в нормализованных таблицах
        pass
        return 0
    
    def get_bots_data_snapshots(self, limit: int = 1000, 
                                start_time: Optional[str] = None,
                                end_time: Optional[str] = None) -> List[Dict]:
        """
        ВАЖНО: Метод больше не загружает снапшоты из старой таблицы!
        
        Вместо снапшотов используйте напрямую нормализованные таблицы:
        - bots_data.db → bots (текущее состояние ботов)
        - bots_data.db → rsi_cache_coins (RSI данные)
        
        Args:
            limit: Максимальное количество записей (игнорируется)
            start_time: Начальное время (игнорируется)
            end_time: Конечное время (игнорируется)
        
        Returns:
            Пустой список (снапшоты больше не используются)
        """
        # Не загружаем снапшоты - используйте напрямую bots_data.db
        pass
        return []
    
    def get_latest_bots_data(self) -> Optional[Dict]:
        """
        ВАЖНО: Метод больше не использует снапшоты!
        
        Используйте напрямую bots_data.db:
        - bots_data.db → bots (текущее состояние ботов)
        - bots_data.db → rsi_cache_coins (RSI данные)
        
        Returns:
            None (снапшоты больше не используются)
        """
        pass
        return None
    
    def count_bots_data_snapshots(self) -> int:
        """Подсчитывает количество снимков данных ботов (всегда 0 - снапшоты больше не используются)"""
        return 0
    
    def cleanup_old_bots_data_snapshots(self, keep_count: int = 1000) -> int:
        """
        ВАЖНО: Метод больше не удаляет снапшоты!
        
        Таблица bots_data_snapshots будет удалена при миграции.
        Снапшоты больше не используются - данные в нормализованных таблицах.
        
        Args:
            keep_count: Количество снимков для сохранения (игнорируется)
        
        Returns:
            0 (нечего удалять)
        """
        pass
        return 0
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Список доступных резервных копий БД из data/backups.
        
        Returns:
            Список словарей с информацией о резервных копиях
        """
        backups = []
        try:
            backup_dir = _get_project_root() / 'data' / 'backups'
            if not backup_dir.exists():
                return backups
            
            for filename in os.listdir(backup_dir):
                if not filename.startswith("ai_data_") or not filename.endswith(".db"):
                    continue
                if filename.count(".db") != 1 or "-wal" in filename or "-shm" in filename:
                    continue
                backup_path = os.path.join(backup_dir, filename)
                try:
                    file_size = os.path.getsize(backup_path)
                    # ai_data_20260127_020021.db -> 20260127_020021
                    timestamp_str = filename.replace("ai_data_", "").replace(".db", "")
                    try:
                        backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except Exception:
                        backup_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
                    
                    backups.append({
                        'path': backup_path,
                        'filename': filename,
                        'size_mb': file_size / 1024 / 1024,
                        'created_at': backup_time.isoformat(),
                        'timestamp': timestamp_str
                    })
                except Exception as e:
                    pass
            
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
            import shutil
            
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
            
            # Создаем резервную копию текущей БД (если она существует)
            if os.path.exists(self.db_path):
                current_backup = self._backup_database()
                if current_backup:
                    logger.info(f"💾 Текущая БД сохранена как: {current_backup}")
            
            # Восстанавливаем БД
            shutil.copy2(backup_path, self.db_path)
            
            # Восстанавливаем WAL и SHM файлы если есть
            wal_backup = f"{backup_path}-wal"
            shm_backup = f"{backup_path}-shm"
            wal_file = self.db_path + '-wal'
            shm_file = self.db_path + '-shm'
            
            if os.path.exists(wal_backup):
                shutil.copy2(wal_backup, wal_file)
            elif os.path.exists(wal_file):
                os.remove(wal_file)
            
            if os.path.exists(shm_backup):
                shutil.copy2(shm_backup, shm_file)
            elif os.path.exists(shm_file):
                os.remove(shm_file)
            
            logger.info(f"✅ БД восстановлена из резервной копии: {backup_path}")
            
            # Проверяем, что БД работает
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    logger.info("✅ Восстановленная БД проверена и работает")
                    return True
            except Exception as e:
                logger.error(f"❌ Восстановленная БД не работает: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления БД из резервной копии: {e}")
            import traceback
            pass
            return False
    
    # ==================== МЕТОДЫ ДЛЯ ИСТОРИИ ОБУЧЕНИЯ (training_history) ====================
    
    def add_training_history_record(self, training_data: Dict) -> int:
        """Добавляет запись в историю обучения"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Используем training_sessions для хранения истории
                event_type = training_data.get('event_type', 'TRAINING')
                status = training_data.get('status', 'COMPLETED')
                
                cursor.execute("""
                    INSERT INTO training_sessions (
                        session_type, started_at, completed_at, status, metadata_json
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    event_type,
                    training_data.get('timestamp', now),
                    now if status in ('COMPLETED', 'FAILED') else None,
                    status,
                    json.dumps(training_data, ensure_ascii=False)
                ))
                
                return cursor.lastrowid
    
    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """Получает историю обучения"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_sessions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            result = []
            for row in rows:
                record = dict(row)
                if record.get('metadata_json'):
                    metadata = json.loads(record['metadata_json'])
                    record.update(metadata)
                result.append(record)
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ МЕТРИК ПРОИЗВОДИТЕЛЬНОСТИ ====================
    
    def save_performance_metrics(self, metrics: Dict, symbol: Optional[str] = None):
        """Сохраняет метрики производительности"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Сохраняем общие метрики
                if 'overall' in metrics:
                    for name, value in metrics['overall'].items():
                        if isinstance(value, (int, float)):
                            cursor.execute("""
                                INSERT INTO performance_metrics (
                                    symbol, metric_type, metric_name, metric_value, recorded_at
                                ) VALUES (?, 'overall', ?, ?, ?)
                            """, (symbol, name, float(value), now))
                
                # Сохраняем метрики по символам
                if 'by_symbol' in metrics:
                    for sym, sym_metrics in metrics['by_symbol'].items():
                        for name, value in sym_metrics.items():
                            if isinstance(value, (int, float)):
                                cursor.execute("""
                                    INSERT INTO performance_metrics (
                                        symbol, metric_type, metric_name, metric_value, recorded_at
                                    ) VALUES (?, 'by_symbol', ?, ?, ?)
                                """, (sym, name, float(value), now))
    
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict:
        """Получает метрики производительности"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT metric_type, metric_name, metric_value, symbol
                FROM performance_metrics
                WHERE 1=1
            """
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY recorded_at DESC LIMIT 1000"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = {
                'overall': {},
                'by_symbol': {}
            }
            
            for row in rows:
                metric_type = row['metric_type']
                metric_name = row['metric_name']
                metric_value = row['metric_value']
                sym = row['symbol']
                
                if metric_type == 'overall':
                    result['overall'][metric_name] = metric_value
                elif metric_type == 'by_symbol' and sym:
                    if sym not in result['by_symbol']:
                        result['by_symbol'][sym] = {}
                    result['by_symbol'][sym][metric_name] = metric_value
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ ВЕРСИЙ МОДЕЛЕЙ ====================
    
    def save_model_version(self, version_data: Dict) -> int:
        """Сохраняет версию модели"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                model_id = version_data.get('id', f"model_{int(datetime.now().timestamp())}")
                
                cursor.execute("""
                    INSERT OR REPLACE INTO model_versions (
                        model_id, model_type, version_number, model_path,
                        accuracy, mse, win_rate, total_pnl, training_samples,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    version_data.get('model_type', 'UNKNOWN'),
                    version_data.get('version_number'),
                    version_data.get('model_path'),
                    version_data.get('accuracy'),
                    version_data.get('mse'),
                    version_data.get('win_rate'),
                    version_data.get('total_pnl'),
                    version_data.get('training_samples'),
                    json.dumps(version_data, ensure_ascii=False),
                    now
                ))
                
                return cursor.lastrowid
    
    def get_model_versions(self, limit: int = 10) -> List[Dict]:
        """Получает версии моделей"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_versions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            result = []
            for row in rows:
                version = dict(row)
                if version.get('metadata_json'):
                    metadata = json.loads(version['metadata_json'])
                    version.update(metadata)
                result.append(version)
            
            return result
    
    def get_latest_model_version(self, model_type: Optional[str] = None) -> Optional[Dict]:
        """Получает последнюю версию модели"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM model_versions WHERE 1=1"
            params = []
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type)
            
            query += " ORDER BY created_at DESC LIMIT 1"
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                version = dict(row)
                if version.get('metadata_json'):
                    metadata = json.loads(version['metadata_json'])
                    version.update(metadata)
                return version
            
            return None
    
    # ==================== МЕТОДЫ ДЛЯ АНАЛИЗА СТРАТЕГИЙ ====================
    
    def save_strategy_analysis(self, analysis_type: str, results: Dict, symbol: Optional[str] = None) -> int:
        """Сохраняет анализ стратегии"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT INTO strategy_analysis (
                        analysis_type, symbol, results_json, created_at
                    ) VALUES (?, ?, ?, ?)
                """, (
                    analysis_type,
                    symbol,
                    json.dumps(results, ensure_ascii=False),
                    now
                ))
                
                return cursor.lastrowid
    
    def get_strategy_analysis(self, analysis_type: Optional[str] = None, symbol: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Получает анализ стратегии"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM strategy_analysis WHERE 1=1"
            params = []
            
            if analysis_type:
                query += " AND analysis_type = ?"
                params.append(analysis_type)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                analysis = dict(row)
                if analysis.get('results_json'):
                    analysis['results'] = json.loads(analysis['results_json'])
                result.append(analysis)
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ ОПТИМИЗИРОВАННЫХ ПАРАМЕТРОВ ====================
    
    def save_optimized_params(self, symbol: Optional[str], params: Dict, optimization_type: Optional[str] = None) -> int:
        """Сохраняет оптимизированные параметры с нормализованными полями"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Извлекаем RSI параметры
                rsi_params = params.get('rsi_params', {}) if isinstance(params.get('rsi_params'), dict) else {}
                if not rsi_params:
                    # Пытаемся извлечь напрямую из params
                    rsi_params = {k: v for k, v in params.items() if 'rsi' in k.lower() or k in ['oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend']}
                
                rsi_long = rsi_params.get('oversold') or rsi_params.get('rsi_long_threshold') or params.get('rsi_long_threshold')
                rsi_short = rsi_params.get('overbought') or rsi_params.get('rsi_short_threshold') or params.get('rsi_short_threshold')
                rsi_exit_long_with = rsi_params.get('exit_long_with_trend') or rsi_params.get('rsi_exit_long_with_trend') or params.get('rsi_exit_long_with_trend')
                rsi_exit_long_against = rsi_params.get('exit_long_against_trend') or rsi_params.get('rsi_exit_long_against_trend') or params.get('rsi_exit_long_against_trend')
                rsi_exit_short_with = rsi_params.get('exit_short_with_trend') or rsi_params.get('rsi_exit_short_with_trend') or params.get('rsi_exit_short_with_trend')
                rsi_exit_short_against = rsi_params.get('exit_short_against_trend') or rsi_params.get('rsi_exit_short_against_trend') or params.get('rsi_exit_short_against_trend')
                
                # Извлекаем Risk параметры
                risk_params = params.get('risk_params', {}) if isinstance(params.get('risk_params'), dict) else {}
                if not risk_params:
                    # Пытаемся извлечь напрямую из params
                    risk_params = {k: v for k, v in params.items() if k in ['max_loss_percent', 'take_profit_percent', 'trailing_stop_activation', 'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval', 'break_even_trigger', 'break_even_protection', 'max_position_hours']}
                
                max_loss = risk_params.get('max_loss_percent') or params.get('max_loss_percent')
                take_profit = risk_params.get('take_profit_percent') or params.get('take_profit_percent')
                trailing_activation = risk_params.get('trailing_stop_activation') or params.get('trailing_stop_activation')
                trailing_distance = risk_params.get('trailing_stop_distance') or params.get('trailing_stop_distance')
                trailing_take = risk_params.get('trailing_take_distance') or params.get('trailing_take_distance')
                trailing_interval = risk_params.get('trailing_update_interval') or params.get('trailing_update_interval')
                break_even_trigger = risk_params.get('break_even_trigger') or params.get('break_even_trigger')
                break_even_protection = risk_params.get('break_even_protection') or params.get('break_even_protection')
                max_hours = risk_params.get('max_position_hours') or params.get('max_position_hours')
                
                # Собираем остальные параметры в extra_params_json
                extra_params = {}
                known_fields = {
                    'rsi_params', 'risk_params', 'rsi_long_threshold', 'rsi_short_threshold',
                    'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                    'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                    'max_loss_percent', 'take_profit_percent', 'trailing_stop_activation',
                    'trailing_stop_distance', 'trailing_take_distance', 'trailing_update_interval',
                    'break_even_trigger', 'break_even_protection', 'max_position_hours',
                    'oversold', 'overbought', 'exit_long_with_trend', 'exit_long_against_trend',
                    'exit_short_with_trend', 'exit_short_against_trend', 'win_rate', 'total_pnl'
                }
                for key, value in params.items():
                    if key not in known_fields:
                        extra_params[key] = value
                
                extra_params_json = json.dumps(extra_params, ensure_ascii=False) if extra_params else None
                
                # Сохраняем полный JSON для обратной совместимости
                params_json = json.dumps(params, ensure_ascii=False)
                
                # Проверяем существующие параметры
                cursor.execute("""
                    SELECT id FROM optimized_params WHERE symbol = ? AND optimization_type = ?
                """, (symbol, optimization_type))
                existing = cursor.fetchone()
                
                if existing:
                    # Обновляем существующие
                    cursor.execute("""
                        UPDATE optimized_params SET
                            rsi_long_threshold = ?, rsi_short_threshold = ?,
                            rsi_exit_long_with_trend = ?, rsi_exit_long_against_trend = ?,
                            rsi_exit_short_with_trend = ?, rsi_exit_short_against_trend = ?,
                            max_loss_percent = ?, take_profit_percent = ?,
                            trailing_stop_activation = ?, trailing_stop_distance = ?,
                            trailing_take_distance = ?, trailing_update_interval = ?,
                            break_even_trigger = ?, break_even_protection = ?,
                            max_position_hours = ?, win_rate = ?, total_pnl = ?,
                            params_json = ?, extra_params_json = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                        rsi_exit_short_with, rsi_exit_short_against,
                        max_loss, take_profit, trailing_activation, trailing_distance,
                        trailing_take, trailing_interval, break_even_trigger,
                        break_even_protection, max_hours,
                        params.get('win_rate'), params.get('total_pnl'),
                        params_json, extra_params_json, now,
                        existing['id']
                    ))
                    return existing['id']
                else:
                    # Создаем новые
                    cursor.execute("""
                        INSERT INTO optimized_params (
                            symbol, rsi_long_threshold, rsi_short_threshold,
                            rsi_exit_long_with_trend, rsi_exit_long_against_trend,
                            rsi_exit_short_with_trend, rsi_exit_short_against_trend,
                            max_loss_percent, take_profit_percent,
                            trailing_stop_activation, trailing_stop_distance,
                            trailing_take_distance, trailing_update_interval,
                            break_even_trigger, break_even_protection,
                            max_position_hours, optimization_type,
                            win_rate, total_pnl, params_json, extra_params_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        rsi_long, rsi_short, rsi_exit_long_with, rsi_exit_long_against,
                        rsi_exit_short_with, rsi_exit_short_against,
                        max_loss, take_profit, trailing_activation, trailing_distance,
                        trailing_take, trailing_interval, break_even_trigger,
                        break_even_protection, max_hours,
                        optimization_type,
                        params.get('win_rate'), params.get('total_pnl'),
                        params_json, extra_params_json,
                        now, now
                    ))
                    return cursor.lastrowid
    
    def get_optimized_params(self, symbol: Optional[str] = None, optimization_type: Optional[str] = None) -> Optional[Dict]:
        """Получает оптимизированные параметры, восстанавливая структуру params"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM optimized_params WHERE 1=1"
            query_params = []
            
            if symbol:
                query += " AND symbol = ?"
                query_params.append(symbol)
            
            if optimization_type:
                query += " AND optimization_type = ?"
                query_params.append(optimization_type)
            
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            cursor.execute(query, query_params)
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                
                # Восстанавливаем params из нормализованных полей или JSON
                params = {}
                if result.get('params_json'):
                    try:
                        params = json.loads(result['params_json'])
                    except:
                        params = {}
                
                # Добавляем нормализованные RSI параметры
                if result.get('rsi_long_threshold') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_long_threshold'] = result['rsi_long_threshold']
                if result.get('rsi_short_threshold') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_short_threshold'] = result['rsi_short_threshold']
                if result.get('rsi_exit_long_with_trend') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_exit_long_with_trend'] = result['rsi_exit_long_with_trend']
                if result.get('rsi_exit_long_against_trend') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_exit_long_against_trend'] = result['rsi_exit_long_against_trend']
                if result.get('rsi_exit_short_with_trend') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_exit_short_with_trend'] = result['rsi_exit_short_with_trend']
                if result.get('rsi_exit_short_against_trend') is not None:
                    if 'rsi_params' not in params:
                        params['rsi_params'] = {}
                    params['rsi_params']['rsi_exit_short_against_trend'] = result['rsi_exit_short_against_trend']
                
                # Добавляем нормализованные Risk параметры
                if result.get('max_loss_percent') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['max_loss_percent'] = result['max_loss_percent']
                if result.get('take_profit_percent') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['take_profit_percent'] = result['take_profit_percent']
                if result.get('trailing_stop_activation') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['trailing_stop_activation'] = result['trailing_stop_activation']
                if result.get('trailing_stop_distance') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['trailing_stop_distance'] = result['trailing_stop_distance']
                if result.get('trailing_take_distance') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['trailing_take_distance'] = result['trailing_take_distance']
                if result.get('trailing_update_interval') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['trailing_update_interval'] = result['trailing_update_interval']
                if result.get('break_even_trigger') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['break_even_trigger'] = result['break_even_trigger']
                if result.get('break_even_protection') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['break_even_protection'] = result['break_even_protection']
                if result.get('max_position_hours') is not None:
                    if 'risk_params' not in params:
                        params['risk_params'] = {}
                    params['risk_params']['max_position_hours'] = result['max_position_hours']
                
                # Добавляем extra_params
                if result.get('extra_params_json'):
                    try:
                        extra_params = json.loads(result['extra_params_json'])
                        params.update(extra_params)
                    except:
                        pass
                
                result['params'] = params
                return result
            
            return None
    
    # ==================== МЕТОДЫ ДЛЯ ПАТТЕРНОВ ТОРГОВЛИ ====================
    
    def save_trade_patterns(self, patterns: List[Dict]) -> int:
        """Сохраняет паттерны торговли"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                saved_count = 0
                
                for pattern in patterns:
                    cursor.execute("""
                        INSERT OR REPLACE INTO trading_patterns (
                            pattern_type, symbol, rsi_range, trend_condition, volatility_range,
                            success_count, failure_count, avg_pnl, avg_duration,
                            pattern_data_json, discovered_at, last_seen_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.get('pattern_type'),
                        pattern.get('symbol'),
                        pattern.get('rsi_range'),
                        pattern.get('trend_condition'),
                        pattern.get('volatility_range'),
                        pattern.get('success_count', 0),
                        pattern.get('failure_count', 0),
                        pattern.get('avg_pnl'),
                        pattern.get('avg_duration'),
                        json.dumps(pattern.get('pattern_data', {}), ensure_ascii=False),
                        pattern.get('discovered_at', now),
                        now
                    ))
                    saved_count += 1
                
                return saved_count
    
    def get_trade_patterns(self, pattern_type: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Получает паттерны торговли"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM trading_patterns WHERE 1=1"
            params = []
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY last_seen_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                pattern = dict(row)
                if pattern.get('pattern_data_json'):
                    pattern['pattern_data'] = json.loads(pattern['pattern_data_json'])
                result.append(pattern)
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ СТАТУСА СЕРВИСА ДАННЫХ ====================
    
    def save_data_service_status(self, service_name: str, status: Dict) -> int:
        """Сохраняет статус сервиса данных в нормализованные столбцы"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Извлекаем основные поля
                last_collection = status.get('last_collection')
                trades_count = status.get('trades', 0)
                candles_count = status.get('candles', 0)
                ready = 1 if status.get('ready', False) else 0
                history_loaded = 1 if status.get('history_loaded', False) else 0
                timestamp = status.get('timestamp', now)
                
                # Собираем остальные поля в extra_status_json
                extra_status = {}
                known_fields = {
                    'last_collection', 'trades', 'candles', 'ready', 
                    'history_loaded', 'timestamp'
                }
                for key, value in status.items():
                    if key not in known_fields:
                        extra_status[key] = value
                
                extra_status_json = json.dumps(extra_status, ensure_ascii=False) if extra_status else None
                
                # ВСЕГДА используем нормализованную структуру (старая колонка status_json должна быть удалена миграцией)
                cursor.execute("""
                    INSERT OR REPLACE INTO data_service_status (
                        service_name, last_collection, trades_count, candles_count,
                        ready, history_loaded, timestamp, extra_status_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    service_name, last_collection, trades_count, candles_count,
                    ready, history_loaded, timestamp, extra_status_json, now
                ))
                
                conn.commit()
                return cursor.lastrowid
    
    def get_data_service_status(self, service_name: str) -> Optional[Dict]:
        """Получает статус сервиса данных из нормализованных столбцов"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Проверяем, есть ли старая структура с status_json
            try:
                cursor.execute("SELECT status_json FROM data_service_status LIMIT 1")
                # Старая структура - используем её для обратной совместимости
                cursor.execute("""
                    SELECT * FROM data_service_status WHERE service_name = ?
                """, (service_name,))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    if result.get('status_json'):
                        result['status'] = json.loads(result['status_json'])
                    return result
            except sqlite3.OperationalError:
                # Новая нормализованная структура
                cursor.execute("""
                    SELECT service_name, last_collection, trades_count, candles_count,
                           ready, history_loaded, timestamp, extra_status_json, updated_at
                    FROM data_service_status WHERE service_name = ?
                """, (service_name,))
                
                row = cursor.fetchone()
                if row:
                    # Восстанавливаем структуру status из нормализованных столбцов
                    status = {
                        'last_collection': row['last_collection'],
                        'trades': row['trades_count'],
                        'candles': row['candles_count'],
                        'ready': bool(row['ready']),
                        'history_loaded': bool(row['history_loaded']),
                        'timestamp': row['timestamp'] or row['updated_at']
                    }
                    
                    # Добавляем дополнительные поля из extra_status_json
                    if row['extra_status_json']:
                        try:
                            extra_status = json.loads(row['extra_status_json'])
                            status.update(extra_status)
                        except:
                            pass
                    
                    return {
                        'service_name': row['service_name'],
                        'status': status,
                        'updated_at': row['updated_at']
                    }
            
            return None
    
    # ==================== МЕТОДЫ ДЛЯ РЕЗУЛЬТАТОВ БЭКТЕСТОВ ====================
    
    def save_backtest_result(self, results: Dict, backtest_name: str = None, symbol: str = None) -> int:
        """Сохраняет результат бэктеста в БД с нормализованными полями"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Извлекаем основные поля
                period_days = results.get('period_days')
                initial_balance = results.get('initial_balance')
                final_balance = results.get('final_balance')
                total_return = results.get('total_return')
                total_pnl = results.get('total_pnl')
                total_trades = results.get('total_trades')
                winning_trades = results.get('winning_trades')
                losing_trades = results.get('losing_trades')
                win_rate = results.get('win_rate')
                avg_win = results.get('avg_win')
                avg_loss = results.get('avg_loss')
                profit_factor = results.get('profit_factor')
                
                # Собираем остальные поля в extra_results_json
                extra_results = {}
                known_fields = {
                    'period_days', 'initial_balance', 'final_balance', 'total_return',
                    'total_pnl', 'total_trades', 'winning_trades', 'losing_trades',
                    'win_rate', 'avg_win', 'avg_loss', 'profit_factor', 'timestamp'
                }
                for key, value in results.items():
                    if key not in known_fields:
                        extra_results[key] = value
                
                extra_results_json = json.dumps(extra_results, ensure_ascii=False) if extra_results else None
                
                # Сохраняем полный JSON для обратной совместимости
                results_json = json.dumps(results, ensure_ascii=False)
                
                cursor.execute("""
                    INSERT INTO backtest_results (
                        backtest_name, symbol, period_days, initial_balance, final_balance,
                        total_return, total_pnl, total_trades, winning_trades, losing_trades,
                        win_rate, avg_win, avg_loss, profit_factor,
                        results_json, extra_results_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_name, symbol, period_days, initial_balance, final_balance,
                    total_return, total_pnl, total_trades, winning_trades, losing_trades,
                    win_rate, avg_win, avg_loss, profit_factor,
                    results_json, extra_results_json, now
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_backtest_results(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Получает результаты бэктестов из БД, восстанавливая структуру results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM backtest_results WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                backtest = dict(row)
                
                # Восстанавливаем results из нормализованных полей или JSON
                results = {}
                if backtest.get('results_json'):
                    try:
                        results = json.loads(backtest['results_json'])
                    except:
                        results = {}
                
                # Добавляем нормализованные поля в results
                if backtest.get('period_days') is not None:
                    results['period_days'] = backtest['period_days']
                if backtest.get('initial_balance') is not None:
                    results['initial_balance'] = backtest['initial_balance']
                if backtest.get('final_balance') is not None:
                    results['final_balance'] = backtest['final_balance']
                if backtest.get('total_return') is not None:
                    results['total_return'] = backtest['total_return']
                if backtest.get('total_pnl') is not None:
                    results['total_pnl'] = backtest['total_pnl']
                if backtest.get('total_trades') is not None:
                    results['total_trades'] = backtest['total_trades']
                if backtest.get('winning_trades') is not None:
                    results['winning_trades'] = backtest['winning_trades']
                if backtest.get('losing_trades') is not None:
                    results['losing_trades'] = backtest['losing_trades']
                if backtest.get('win_rate') is not None:
                    results['win_rate'] = backtest['win_rate']
                if backtest.get('avg_win') is not None:
                    results['avg_win'] = backtest['avg_win']
                if backtest.get('avg_loss') is not None:
                    results['avg_loss'] = backtest['avg_loss']
                if backtest.get('profit_factor') is not None:
                    results['profit_factor'] = backtest['profit_factor']
                
                # Добавляем extra_results
                if backtest.get('extra_results_json'):
                    try:
                        extra_results = json.loads(backtest['extra_results_json'])
                        results.update(extra_results)
                    except:
                        pass
                
                backtest['results'] = results
                result.append(backtest)
            
            return result
    
    # ==================== МЕТОДЫ ДЛЯ БАЗЫ ЗНАНИЙ ====================
    
    def save_knowledge_base(self, knowledge_type: str, knowledge_data: Dict) -> int:
        """Сохраняет базу знаний в БД"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge_base (
                        knowledge_type, knowledge_data_json, last_update
                    ) VALUES (?, ?, ?)
                """, (
                    knowledge_type,
                    json.dumps(knowledge_data, ensure_ascii=False),
                    now
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_knowledge_base(self, knowledge_type: str) -> Optional[Dict]:
        """Получает базу знаний из БД"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM knowledge_base WHERE knowledge_type = ?
            """, (knowledge_type,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('knowledge_data_json'):
                    result['knowledge_data'] = json.loads(result['knowledge_data_json'])
                return result
            
            return None
    
    # ==================== МЕТОДЫ ДЛЯ ДАННЫХ ОБУЧЕНИЯ ====================
    
    def save_training_data(self, data_type: str, data: Dict, symbol: str = None) -> int:
        """Сохраняет данные обучения в БД"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Проверяем, есть ли уже данные этого типа для символа
                if symbol:
                    cursor.execute("""
                        SELECT id FROM training_data WHERE data_type = ? AND symbol = ?
                    """, (data_type, symbol))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Обновляем существующую запись
                        cursor.execute("""
                            UPDATE training_data 
                            SET data_json = ?, updated_at = ?
                            WHERE data_type = ? AND symbol = ?
                        """, (json.dumps(data, ensure_ascii=False), now, data_type, symbol))
                        conn.commit()
                        return existing['id']
                
                # Создаем новую запись
                cursor.execute("""
                    INSERT INTO training_data (
                        data_type, symbol, data_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    data_type,
                    symbol,
                    json.dumps(data, ensure_ascii=False),
                    now,
                    now
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_training_data(self, data_type: str, symbol: str = None) -> Optional[Dict]:
        """Получает данные обучения из БД"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM training_data WHERE data_type = ?"
            params = [data_type]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                if result.get('data_json'):
                    result['data'] = json.loads(result['data_json'])
                return result
            
            return None
    
    # ==================== МЕТОДЫ ДЛЯ КОНФИГОВ БОТОВ ====================
    
    def save_bot_config(self, symbol: str, config: Dict) -> int:
        """Сохраняет конфиг бота в БД с нормализованными полями"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Извлекаем основные поля
                rsi_long_threshold = config.get('rsi_long_threshold')
                rsi_short_threshold = config.get('rsi_short_threshold')
                rsi_exit_long_with_trend = config.get('rsi_exit_long_with_trend')
                rsi_exit_long_against_trend = config.get('rsi_exit_long_against_trend')
                rsi_exit_short_with_trend = config.get('rsi_exit_short_with_trend')
                rsi_exit_short_against_trend = config.get('rsi_exit_short_against_trend')
                max_loss_percent = config.get('max_loss_percent')
                take_profit_percent = config.get('take_profit_percent')
                trailing_stop_activation = config.get('trailing_stop_activation')
                trailing_stop_distance = config.get('trailing_stop_distance')
                trailing_take_distance = config.get('trailing_take_distance')
                trailing_update_interval = config.get('trailing_update_interval')
                break_even_trigger = config.get('break_even_trigger')
                break_even_protection = config.get('break_even_protection')
                max_position_hours = config.get('max_position_hours')
                rsi_time_filter_enabled = 1 if config.get('rsi_time_filter_enabled') else 0
                rsi_time_filter_candles = config.get('rsi_time_filter_candles')
                rsi_time_filter_upper = config.get('rsi_time_filter_upper')
                rsi_time_filter_lower = config.get('rsi_time_filter_lower')
                avoid_down_trend = 1 if config.get('avoid_down_trend') else 0
                
                # Собираем остальные поля в extra_config_json
                extra_config = {}
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
                for key, value in config.items():
                    if key not in known_fields:
                        extra_config[key] = value
                
                extra_config_json = json.dumps(extra_config, ensure_ascii=False) if extra_config else None
                
                # Сохраняем полный JSON для обратной совместимости
                config_json = json.dumps(config, ensure_ascii=False)
                
                # Получаем created_at из существующей записи
                cursor.execute("SELECT created_at FROM bot_configs WHERE symbol = ?", (symbol,))
                existing = cursor.fetchone()
                final_created_at = existing[0] if existing else now
                
                cursor.execute("""
                    INSERT OR REPLACE INTO bot_configs (
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
                        config_json, extra_config_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
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
                    config_json, extra_config_json,
                    final_created_at, now
                ))
                conn.commit()
                return cursor.lastrowid
    
    def get_bot_config(self, symbol: str) -> Optional[Dict]:
        """Получает конфиг бота из БД, восстанавливая структуру config"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM bot_configs WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                
                # Восстанавливаем config из нормализованных полей или JSON
                config = {}
                if result.get('config_json'):
                    try:
                        config = json.loads(result['config_json'])
                    except:
                        config = {}
                
                # Добавляем нормализованные поля в config
                if result.get('rsi_long_threshold') is not None:
                    config['rsi_long_threshold'] = result['rsi_long_threshold']
                if result.get('rsi_short_threshold') is not None:
                    config['rsi_short_threshold'] = result['rsi_short_threshold']
                if result.get('rsi_exit_long_with_trend') is not None:
                    config['rsi_exit_long_with_trend'] = result['rsi_exit_long_with_trend']
                if result.get('rsi_exit_long_against_trend') is not None:
                    config['rsi_exit_long_against_trend'] = result['rsi_exit_long_against_trend']
                if result.get('rsi_exit_short_with_trend') is not None:
                    config['rsi_exit_short_with_trend'] = result['rsi_exit_short_with_trend']
                if result.get('rsi_exit_short_against_trend') is not None:
                    config['rsi_exit_short_against_trend'] = result['rsi_exit_short_against_trend']
                if result.get('max_loss_percent') is not None:
                    config['max_loss_percent'] = result['max_loss_percent']
                if result.get('take_profit_percent') is not None:
                    config['take_profit_percent'] = result['take_profit_percent']
                if result.get('trailing_stop_activation') is not None:
                    config['trailing_stop_activation'] = result['trailing_stop_activation']
                if result.get('trailing_stop_distance') is not None:
                    config['trailing_stop_distance'] = result['trailing_stop_distance']
                if result.get('trailing_take_distance') is not None:
                    config['trailing_take_distance'] = result['trailing_take_distance']
                if result.get('trailing_update_interval') is not None:
                    config['trailing_update_interval'] = result['trailing_update_interval']
                if result.get('break_even_trigger') is not None:
                    config['break_even_trigger'] = result['break_even_trigger']
                if result.get('break_even_protection') is not None:
                    config['break_even_protection'] = result['break_even_protection']
                if result.get('max_position_hours') is not None:
                    config['max_position_hours'] = result['max_position_hours']
                if result.get('rsi_time_filter_enabled') is not None:
                    config['rsi_time_filter_enabled'] = bool(result['rsi_time_filter_enabled'])
                if result.get('rsi_time_filter_candles') is not None:
                    config['rsi_time_filter_candles'] = result['rsi_time_filter_candles']
                if result.get('rsi_time_filter_upper') is not None:
                    config['rsi_time_filter_upper'] = result['rsi_time_filter_upper']
                if result.get('rsi_time_filter_lower') is not None:
                    config['rsi_time_filter_lower'] = result['rsi_time_filter_lower']
                if result.get('avoid_down_trend') is not None:
                    config['avoid_down_trend'] = bool(result['avoid_down_trend'])
                
                # Добавляем extra_config
                if result.get('extra_config_json'):
                    try:
                        extra_config = json.loads(result['extra_config_json'])
                        config.update(extra_config)
                    except:
                        pass
                
                result['config'] = config
                return result
            
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Получает общую статистику базы данных"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Подсчеты по таблицам
            tables = ['simulated_trades', 'bot_trades', 'exchange_trades', 'ai_decisions', 
                     'training_sessions', 'parameter_training_samples', 'used_training_parameters',
                     'best_params_per_symbol', 'blocked_params', 'win_rate_targets', 'training_locks',
                     'candles_history', 'model_versions', 'performance_metrics',
                     'strategy_analysis', 'optimized_params', 'trading_patterns', 'data_service_status']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Размер базы данных
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            stats['database_size_mb'] = db_size / 1024 / 1024
            
            # Статистика по символам
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM simulated_trades")
            stats['unique_symbols_simulated'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM bot_trades WHERE is_simulated = 0")
            stats['unique_symbols_real'] = cursor.fetchone()[0]
            
            return stats


# Глобальный экземпляр базы данных
_ai_database_instance = None
_ai_database_lock = threading.Lock()


def get_ai_database(db_path: str = None) -> AIDatabase:
    """Получает глобальный экземпляр базы данных AI"""
    global _ai_database_instance
    
    with _ai_database_lock:
        if _ai_database_instance is None:
            _ai_database_instance = AIDatabase(db_path)
        
        return _ai_database_instance

