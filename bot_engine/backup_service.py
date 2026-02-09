#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервис для бэкапа баз данных AI и Bots

Предоставляет централизованное управление резервными копиями:
- Создание бэкапов обеих БД
- Управление бэкапами (список, удаление, восстановление)
- Автоматическая очистка старых бэкапов
- Проверка целостности бэкапов
"""

import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger('BackupService')


def _get_project_root() -> Path:
    """
    Определяет корень проекта относительно текущего файла.
    Корень проекта - директория, где лежит app.py/bots.py и bot_engine/
    """
    current = Path(__file__).resolve()
    # Поднимаемся от bot_engine/backup_service.py до корня проекта
    # bot_engine/ -> корень
    for parent in [current.parent.parent] + list(current.parents):
        if parent and ((parent / 'app.py').exists() or (parent / 'bots.py').exists()) and (parent / 'bot_engine').exists():
            return parent
    # Фолбек: поднимаемся на 1 уровень
    try:
        return current.parents[1]
    except IndexError:
        return current.parent


class DatabaseBackupService:
    """
    Сервис для управления бэкапами баз данных AI и Bots
    """
    
    def __init__(self, backup_dir: str = None):
        """
        Инициализация сервиса бэкапа
        
        Args:
            backup_dir: Директория для хранения бэкапов (по умолчанию: data/backups/)
        """
        if backup_dir is None:
            # ✅ ПУТЬ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
            project_root = _get_project_root()
            backup_dir = project_root / 'data' / 'backups'
            backup_dir = str(backup_dir.resolve())
        
        self.backup_dir = os.path.normpath(backup_dir)
        self.lock = threading.RLock()
        
        # Создаем директорию для бэкапов если её нет
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            logger.info(f"✅ Директория бэкапов: {self.backup_dir}")
        except OSError as e:
            logger.error(f"❌ Ошибка создания директории бэкапов: {e}")
            raise
    
    def create_backup(self, include_ai: bool = True, include_bots: bool = True,
                     include_app: bool = False, max_retries: int = 3,
                     keep_last_n: int = 5) -> Dict[str, Any]:
        """
        Создает резервные копии указанных баз данных.
        После создания оставляет только последние keep_last_n бэкапов для каждой системы.
        
        Args:
            include_ai: Создавать бэкап AI БД (ai_data.db)
            include_bots: Создавать бэкап Bots БД (bots_data.db)
            include_app: Создавать бэкап App БД (app_data.db)
            max_retries: Максимальное количество попыток при блокировке файла
            keep_last_n: Сколько последних бэкапов хранить для каждой БД (остальные удаляются)
        
        Returns:
            Словарь с результатами бэкапа (backups: ai, bots, app).
        """
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = {
                'success': True,
                'timestamp': timestamp,
                'backups': {
                    'ai': None,
                    'bots': None,
                    'app': None
                },
                'errors': []
            }
            
            # ✅ ПУТИ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
            project_root = _get_project_root()
            ai_db_path = str((project_root / 'data' / 'ai_data.db').resolve())
            bots_db_path = str((project_root / 'data' / 'bots_data.db').resolve())
            app_db_path = str((project_root / 'data' / 'app_data.db').resolve())
            
            # Бэкап App БД (только из app.py)
            if include_app:
                try:
                    app_backup = self._backup_database(
                        db_path=app_db_path,
                        db_name='app_data',
                        timestamp=timestamp,
                        max_retries=max_retries
                    )
                    if app_backup:
                        result['backups']['app'] = app_backup
                        logger.info(f"✅ Создан бэкап App БД: {app_backup['path']}")
                    else:
                        if not os.path.exists(app_db_path):
                            result['errors'].append(f"App БД не найдена: {app_db_path}")
                            logger.warning(f"⚠️ App БД не найдена: {app_db_path}")
                        else:
                            result['success'] = False
                            result['errors'].append("Не удалось создать бэкап App БД")
                except Exception as e:
                    result['success'] = False
                    result['errors'].append(f"Ошибка создания бэкапа App БД: {e}")
                    logger.error(f"❌ Ошибка создания бэкапа App БД: {e}")
            
            # Бэкап AI БД
            if include_ai:
                try:
                    ai_backup = self._backup_database(
                        db_path=ai_db_path,
                        db_name='ai_data',
                        timestamp=timestamp,
                        max_retries=max_retries
                    )
                    if ai_backup:
                        result['backups']['ai'] = ai_backup
                        logger.info(f"✅ Создан бэкап AI БД: {ai_backup['path']}")
                    else:
                        # Если БД не найдена, это не критическая ошибка
                        if not os.path.exists(ai_db_path):
                            result['errors'].append(f"AI БД не найдена: {ai_db_path}")
                            logger.warning(f"⚠️ AI БД не найдена: {ai_db_path}")
                        else:
                            result['success'] = False
                            result['errors'].append("Не удалось создать бэкап AI БД")
                except Exception as e:
                    result['success'] = False
                    error_msg = f"Ошибка создания бэкапа AI БД: {e}"
                    result['errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            # Бэкап Bots БД
            if include_bots:
                try:
                    bots_backup = self._backup_database(
                        db_path=bots_db_path,
                        db_name='bots_data',
                        timestamp=timestamp,
                        max_retries=max_retries
                    )
                    if bots_backup:
                        result['backups']['bots'] = bots_backup
                        logger.info(f"✅ Создан бэкап Bots БД: {bots_backup['path']}")
                    else:
                        # Если БД не найдена, это не критическая ошибка
                        if not os.path.exists(bots_db_path):
                            result['errors'].append(f"Bots БД не найдена: {bots_db_path}")
                            logger.warning(f"⚠️ Bots БД не найдена: {bots_db_path}")
                        else:
                            result['success'] = False
                            result['errors'].append("Не удалось создать бэкап Bots БД")
                except Exception as e:
                    result['success'] = False
                    error_msg = f"Ошибка создания бэкапа Bots БД: {e}"
                    result['errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            # Считаем успешным, если создан хотя бы один бэкап
            has_backups = (
                result['backups']['app'] is not None
                or result['backups']['ai'] is not None
                or result['backups']['bots'] is not None
            )
            if has_backups:
                if result['errors']:
                    logger.warning(f"⚠️ Бэкап создан с предупреждениями: {timestamp}")
                else:
                    logger.info(f"✅ Бэкап успешно создан: {timestamp}")
            else:
                result['success'] = False
                logger.warning(f"⚠️ Бэкап не создан: {timestamp}")

            # Оставляем только последние keep_last_n бэкапов для каждой системы.
            # Небольшая пауза перед удалением, чтобы снизить WinError 32 (файл занят) на Windows.
            if keep_last_n > 0:
                try:
                    time.sleep(2)
                    self.cleanup_excess_backups(keep_count=keep_last_n)
                except Exception as e:
                    logger.warning(f"⚠️ Очистка лишних бэкапов не выполнена: {e}")
            
            return result
    
    def _backup_database(self, db_path: str, db_name: str, timestamp: str,
                        max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Создаёт резервную копию одной БД в виде SQL-дампа (.sql).
        Через запрос/iterdump — не копируем файл, нет блокировок.
        """
        if not os.path.exists(db_path):
            logger.warning(f"⚠️ БД не найдена: {db_path}")
            return None

        backup_filename = f"{db_name}_{timestamp}.sql"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        backup_path_tmp = backup_path + '.tmp'

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(1.0 * attempt)
                conn = sqlite3.connect(db_path, timeout=30.0)
                try:
                    with open(backup_path_tmp, 'w', encoding='utf-8') as f:
                        for line in conn.iterdump():
                            f.write(line + '\n')
                finally:
                    conn.close()
                if os.path.getsize(backup_path_tmp) == 0:
                    try:
                        os.remove(backup_path_tmp)
                    except OSError:
                        pass
                    return None
                os.replace(backup_path_tmp, backup_path)
                file_size = os.path.getsize(backup_path)
                size_mb = file_size / (1024 * 1024)
                return {
                    'path': backup_path,
                    'size_mb': size_mb,
                    'size_bytes': file_size,
                    'valid': True,
                    'created_at': datetime.now().isoformat()
                }
            except sqlite3.Error as e:
                if os.path.exists(backup_path_tmp):
                    try:
                        os.remove(backup_path_tmp)
                    except OSError:
                        pass
                logger.warning(f"⚠️ Ошибка дампа БД (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                if os.path.exists(backup_path_tmp):
                    try:
                        os.remove(backup_path_tmp)
                    except OSError:
                        pass
                logger.error(f"❌ Ошибка создания SQL-бэкапа: {e}")
                return None
        return None
    
    def _check_backup_integrity(self, backup_path: str) -> Tuple[bool, Optional[str]]:
        """
        Проверяет целостность бэкапа: для .sql — файл непустой; для .db — PRAGMA integrity_check.
        """
        if not os.path.exists(backup_path):
            return False, "Файл бэкапа не найден"
        if backup_path.endswith('.sql'):
            return (os.path.getsize(backup_path) > 0, None)
        try:
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()
            if result and result[0] == "ok":
                return True, None
            return False, result[0] if result else "Неизвестная ошибка"
        except Exception as e:
            return False, str(e)
    
    def list_backups(self, db_name: str = None) -> List[Dict[str, Any]]:
        """
        Получает список всех бэкапов
        
        Args:
            db_name: Фильтр по имени БД ('ai_data', 'bots_data', 'app_data'), None для всех
        
        Returns:
            Список словарей с информацией о бэкапах
        """
        backups = []
        
        try:
            if not os.path.exists(self.backup_dir):
                return backups
            
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('-wal') or filename.endswith('-shm'):
                    continue
                is_sql = filename.endswith('.sql')
                if not is_sql and not filename.endswith('.db'):
                    continue
                if db_name and not filename.startswith(db_name):
                    continue
                backup_path = os.path.join(self.backup_dir, filename)
                try:
                    name_without_ext = filename[:-4] if is_sql else filename[:-3]
                    parts = name_without_ext.split('_')
                    timestamp_str = None
                    db_name_from_file = None
                    for i in range(len(parts) - 1):
                        potential_timestamp = '_'.join(parts[i:])
                        if len(potential_timestamp) == 15 and potential_timestamp.replace('_', '').isdigit():
                            timestamp_str = potential_timestamp
                            db_name_from_file = '_'.join(parts[:i])
                            break
                    if not timestamp_str:
                        timestamp_str = datetime.fromtimestamp(os.path.getmtime(backup_path)).strftime("%Y%m%d_%H%M%S")
                        db_name_from_file = name_without_ext.rsplit('_', 2)[0] if '_' in name_without_ext else name_without_ext
                    try:
                        backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except ValueError:
                        backup_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
                    file_size = os.path.getsize(backup_path)
                    size_mb = file_size / (1024 * 1024)
                    is_valid, error_msg = self._check_backup_integrity(backup_path)
                    backups.append({
                        'path': backup_path,
                        'filename': filename,
                        'db_name': db_name_from_file,
                        'size_mb': size_mb,
                        'size_bytes': file_size,
                        'created_at': backup_time.isoformat(),
                        'timestamp': timestamp_str,
                        'valid': is_valid,
                        'error': error_msg if not is_valid else None
                    })
                except Exception:
                    pass
            
            # Сортируем по дате создания (новые первыми)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения списка бэкапов: {e}")
        
        return backups
    
    def restore_backup(self, backup_path: str, db_name: str = None) -> bool:
        """
        Восстанавливает БД из бэкапа
        
        Args:
            backup_path: Путь к файлу бэкапа
            db_name: Имя БД ('ai_data', 'bots_data', 'app_data'), если None определяется автоматически
        
        Returns:
            True если восстановление успешно, False в противном случае
        """
        if not os.path.exists(backup_path):
            logger.error(f"❌ Бэкап не найден: {backup_path}")
            return False
        
        # Определяем имя БД если не указано
        if db_name is None:
            filename = os.path.basename(backup_path)
            if filename.startswith('ai_data'):
                db_name = 'ai_data'
            elif filename.startswith('bots_data'):
                db_name = 'bots_data'
            elif filename.startswith('app_data'):
                db_name = 'app_data'
            else:
                logger.error(f"❌ Не удалось определить имя БД из имени файла: {filename}")
                return False
        
        # ✅ ПУТИ ОТНОСИТЕЛЬНО КОРНЯ ПРОЕКТА, А НЕ РАБОЧЕЙ ДИРЕКТОРИИ
        project_root = _get_project_root()
        if db_name == 'ai_data':
            target_db_path = str((project_root / 'data' / 'ai_data.db').resolve())
        elif db_name == 'bots_data':
            target_db_path = str((project_root / 'data' / 'bots_data.db').resolve())
        elif db_name == 'app_data':
            target_db_path = str((project_root / 'data' / 'app_data.db').resolve())
        else:
            logger.error(f"❌ Неизвестное имя БД: {db_name}")
            return False
        
        try:
            logger.info(f"📦 Восстановление {db_name} из бэкапа: {backup_path}")
            # Удаляем старую БД (и -wal, -shm), создаём новую и загружаем дамп
            for path in [target_db_path, target_db_path + '-wal', target_db_path + '-shm']:
                self._remove_file_safe(path)
            if backup_path.endswith('.sql'):
                with open(backup_path, 'r', encoding='utf-8') as f:
                    sql_dump = f.read()
                conn = sqlite3.connect(target_db_path)
                conn.executescript(sql_dump)
                conn.close()
            else:
                shutil.copy2(backup_path, target_db_path)
                self._remove_file_safe(target_db_path + '-wal')
                self._remove_file_safe(target_db_path + '-shm')
            is_valid, error_msg = self._check_backup_integrity(target_db_path)
            if is_valid:
                logger.info(f"✅ БД {db_name} успешно восстановлена из бэкапа")
                return True
            logger.error(f"❌ Восстановленная БД повреждена: {error_msg}")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления БД из бэкапа: {e}")
            return False
    
    def _remove_file_safe(self, path: str, max_retries: int = 3) -> bool:
        """
        Удаляет файл с повторами при WinError 32 / EBUSY (файл занят другим процессом).
        Returns True если удалён или файла нет, False если не удалось.
        """
        for attempt in range(max_retries):
            try:
                if not os.path.exists(path):
                    return True
                os.remove(path)
                return True
            except (PermissionError, OSError) as e:
                # Windows: 32 = ERROR_SHARING_VIOLATION (файл занят)
                # Unix: 13 EACCES, 16 EBUSY
                is_busy = getattr(e, 'winerror', None) == 32 or getattr(e, 'errno', None) in (13, 16)
                if is_busy and attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if is_busy:
                    logger.warning(
                        f"⚠️ Файл занят другим процессом, пропуск удаления (будет повтор при следующем бэкапе): {path}"
                    )
                else:
                    logger.warning(f"⚠️ Не удалось удалить файл: {path}: {e}")
                return False
        return False

    def delete_backup(self, backup_path: str) -> bool:
        """
        Удаляет бэкап (основной файл и -wal/-shm при наличии).
        При «файл занят» выполняет несколько попыток с паузой, затем пропускает без падения.
        """
        if not os.path.exists(backup_path):
            logger.warning(f"⚠️ Бэкап не найден: {backup_path}")
            return False

        ok = self._remove_file_safe(backup_path)
        if not ok:
            return False

        wal_file = backup_path + '-wal'
        shm_file = backup_path + '-shm'
        self._remove_file_safe(wal_file)
        self._remove_file_safe(shm_file)

        logger.info(f"🗑️ Бэкап удален: {backup_path}")
        return True
    
    def cleanup_excess_backups(self, keep_count: int = 5) -> Dict[str, int]:
        """
        Оставляет только последние keep_count бэкапов для каждой системы (AI, Bots).
        Все остальные бэкапы удаляются независимо от возраста.
        
        Args:
            keep_count: Сколько последних бэкапов сохранять для каждой БД (по умолчанию 5)
        
        Returns:
            Словарь с количеством удаленных бэкапов по типам
        """
        result = {
            'ai_data': 0,
            'bots_data': 0,
            'app_data': 0,
            'total': 0
        }
        try:
            backups = self.list_backups()
            backups_by_type = {}
            for backup in backups:
                db_name = backup.get('db_name', 'unknown')
                if db_name not in backups_by_type:
                    backups_by_type[db_name] = []
                backups_by_type[db_name].append(backup)

            for db_name, db_backups in backups_by_type.items():
                if db_name not in result:
                    result[db_name] = 0
                to_keep = db_backups[:keep_count]
                to_delete = db_backups[keep_count:]
                for backup in to_delete:
                    if self.delete_backup(backup['path']):
                        result[db_name] = result.get(db_name, 0) + 1
                        result['total'] += 1

            if result['total'] > 0:
                logger.info(
                    f"🗑️ Удалено лишних бэкапов (оставлено по {keep_count} на систему): "
                    f"{result['total']} (ai: {result.get('ai_data', 0)}, bots: {result.get('bots_data', 0)}, app: {result.get('app_data', 0)})"
                )
        except Exception as e:
            logger.error(f"❌ Ошибка очистки лишних бэкапов: {e}")
        return result

    def cleanup_old_backups(self, days: int = 30, keep_count: int = 10) -> Dict[str, int]:
        """
        Удаляет старые бэкапы
        
        Args:
            days: Удалять бэкапы старше указанного количества дней
            keep_count: Минимальное количество бэкапов каждого типа для сохранения
        
        Returns:
            Словарь с количеством удаленных бэкапов по типам
        """
        result = {
            'ai_data': 0,
            'bots_data': 0,
            'app_data': 0,
            'total': 0
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            backups = self.list_backups()
            
            backups_by_type = {}
            for backup in backups:
                db_name = backup.get('db_name', 'unknown')
                if db_name not in backups_by_type:
                    backups_by_type[db_name] = []
                backups_by_type[db_name].append(backup)
            
            for db_name, db_backups in backups_by_type.items():
                if db_name not in result:
                    result[db_name] = 0
                # Сортируем по дате (старые первыми)
                db_backups.sort(key=lambda x: x['created_at'])
                
                # Оставляем последние keep_count бэкапов
                to_keep = db_backups[-keep_count:] if len(db_backups) > keep_count else []
                to_delete = []
                
                for backup in db_backups:
                    if backup in to_keep:
                        continue
                    
                    backup_date = datetime.fromisoformat(backup['created_at'])
                    if backup_date < cutoff_date:
                        to_delete.append(backup)
                
                # Удаляем старые бэкапы
                for backup in to_delete:
                    if self.delete_backup(backup['path']):
                        result[db_name] = result.get(db_name, 0) + 1
                        result['total'] += 1
            
            if result['total'] > 0:
                logger.info(
                    f"🗑️ Удалено старых бэкапов: {result['total']} "
                    f"(ai: {result.get('ai_data', 0)}, bots: {result.get('bots_data', 0)}, app: {result.get('app_data', 0)})"
                )
            else:
                logger.info("ℹ️ Старые бэкапы не найдены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки старых бэкапов: {e}")
        
        return result
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """
        Получает статистику по бэкапам
        
        Returns:
            Словарь со статистикой
        """
        backups = self.list_backups()
        
        stats = {
            'total_backups': len(backups),
            'total_size_mb': 0,
            'ai_data_backups': 0,
            'bots_data_backups': 0,
            'app_data_backups': 0,
            'ai_data_size_mb': 0,
            'bots_data_size_mb': 0,
            'app_data_size_mb': 0,
            'oldest_backup': None,
            'newest_backup': None,
            'invalid_backups': 0
        }
        
        for backup in backups:
            stats['total_size_mb'] += backup['size_mb']
            db_name = backup.get('db_name', '')
            if db_name == 'ai_data':
                stats['ai_data_backups'] += 1
                stats['ai_data_size_mb'] += backup['size_mb']
            elif db_name == 'bots_data':
                stats['bots_data_backups'] += 1
                stats['bots_data_size_mb'] += backup['size_mb']
            elif db_name == 'app_data':
                stats['app_data_backups'] += 1
                stats['app_data_size_mb'] += backup['size_mb']
            
            if not backup.get('valid', True):
                stats['invalid_backups'] += 1
            
            if stats['oldest_backup'] is None or backup['created_at'] < stats['oldest_backup']:
                stats['oldest_backup'] = backup['created_at']
            
            if stats['newest_backup'] is None or backup['created_at'] > stats['newest_backup']:
                stats['newest_backup'] = backup['created_at']
        
        return stats


def _run_backup_job(backup_service: 'DatabaseBackupService', backup_config: dict) -> None:
    """Запускает единичный цикл резервного копирования (только выбранные БД: app / ai / bots)."""
    backup_logger = logging.getLogger('BackupScheduler')
    include_app = backup_config.get('APP_ENABLED', False)
    include_ai = backup_config.get('AI_ENABLED', False)
    include_bots = backup_config.get('BOTS_ENABLED', False)

    if not include_app and not include_ai and not include_bots:
        backup_logger.info("[Backup] Нет активных БД для резервного копирования, задание пропущено")
        return

    max_retries = backup_config.get('MAX_RETRIES', 3)
    keep_last_n = backup_config.get('KEEP_LAST_N', 5)
    try:
        result = backup_service.create_backup(
            include_app=include_app,
            include_ai=include_ai,
            include_bots=include_bots,
            max_retries=max_retries,
            keep_last_n=keep_last_n
        )
    except Exception as exc:
        backup_logger.exception(f"[Backup] Ошибка выполнения резервного копирования: {exc}")
        return

    timestamp = result.get('timestamp', 'unknown')
    if result.get('success'):
        backup_logger.info(f"[Backup] Резервное копирование завершено успешно (timestamp={timestamp})")
    else:
        backup_logger.warning(f"[Backup] Резервное копирование завершено с ошибками (timestamp={timestamp})")

    for db_key in ('app', 'ai', 'bots'):
        backup_info = result.get('backups', {}).get(db_key)
        if backup_info:
            backup_logger.info(
                "[Backup] %s: файл %s (%.2f MB, valid=%s)",
                db_key.upper(),
                backup_info.get('path', ''),
                backup_info.get('size_mb', 0),
                'yes' if backup_info.get('valid', True) else 'no'
            )

    for warning_msg in result.get('errors', []):
        backup_logger.warning(f"[Backup] {warning_msg}")


def run_backup_scheduler_loop(
    backup_config: dict,
    stop_event: Optional[threading.Event] = None
) -> None:
    """
    Фоновый планировщик регулярных бэкапов. Каждый процесс бэкапит только свою БД:
    app.py → app_data, bots.py → bots_data, ai.py → ai_data.
    """
    backup_logger = logging.getLogger('BackupScheduler')
    backup_config = backup_config or {}

    if not backup_config.get('ENABLED', True):
        backup_logger.info("[Backup] Автоматическое резервное копирование выключено настройками")
        return

    has_any = (
        backup_config.get('APP_ENABLED', False)
        or backup_config.get('AI_ENABLED', False)
        or backup_config.get('BOTS_ENABLED', False)
    )
    if not has_any:
        backup_logger.info("[Backup] Ни одна база не выбрана для резервного копирования, поток остановлен")
        return

    backup_dir = backup_config.get('BACKUP_DIR')
    try:
        backup_svc = get_backup_service(backup_dir)
    except Exception as exc:
        backup_logger.exception(f"[Backup] Не удалось инициализировать сервис бэкапов: {exc}")
        return

    interval_minutes = backup_config.get('INTERVAL_MINUTES', 180)
    try:
        interval_minutes = float(interval_minutes)
    except (TypeError, ValueError):
        backup_logger.warning("[Backup] Некорректное значение INTERVAL_MINUTES, используется 180 минут (3 часа)")
        interval_minutes = 180

    interval_seconds = max(60, int(interval_minutes * 60))
    backup_logger.info(
        f"[Backup] Планировщик запущен: каждые {interval_minutes} мин ({interval_seconds} сек). Директория: {backup_dir or 'data/backups'}"
    )

    if backup_config.get('RUN_ON_START', True):
        _run_backup_job(backup_svc, backup_config)

    ev = stop_event if stop_event is not None else threading.Event()
    while not ev.wait(interval_seconds):
        _run_backup_job(backup_svc, backup_config)


# Глобальный экземпляр сервиса
_backup_service_instance = None
_backup_service_lock = threading.Lock()


def get_backup_service(backup_dir: str = None) -> DatabaseBackupService:
    """
    Получает глобальный экземпляр сервиса бэкапа
    
    Args:
        backup_dir: Директория для хранения бэкапов (по умолчанию: data/backups/)
    
    Returns:
        Экземпляр DatabaseBackupService
    """
    global _backup_service_instance
    
    with _backup_service_lock:
        if _backup_service_instance is None:
            _backup_service_instance = DatabaseBackupService(backup_dir)
        
        return _backup_service_instance

