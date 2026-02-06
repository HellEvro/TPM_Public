#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Удаление записей из candles_cache_data, у которых нет соответствующего cache_id в candles_cache
"""

import sys
import os
from pathlib import Path
import logging
import sqlite3
import time

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.color_logger import setup_color_logging

# Настройка логирования
setup_color_logging(console_log_levels=['+INFO', '+WARNING', '+ERROR'])
logger = logging.getLogger('RemoveOrphanedCandles')

def remove_orphaned_candles(db_path: str):
    """Удаление записей из candles_cache_data без соответствующего cache_id в candles_cache"""
    logger.info("=" * 80)
    logger.info(f"Удаление orphaned записей из candles_cache_data в: {db_path}")
    logger.info("=" * 80)
    
    if not Path(db_path).exists():
        logger.error(f"Файл базы данных не найден: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        cursor = conn.cursor()
        
        # Подсчитываем количество записей до удаления
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_before = cursor.fetchone()[0]
        logger.info(f"Всего записей в candles_cache_data до удаления: {total_before:,}")
        
        # Подсчитываем количество уникальных cache_id в candles_cache
        cursor.execute("SELECT COUNT(*) FROM candles_cache")
        valid_cache_ids_count = cursor.fetchone()[0]
        logger.info(f"Уникальных cache_id в candles_cache: {valid_cache_ids_count:,}")
        
        # Подсчитываем количество orphaned записей
        cursor.execute("""
            SELECT COUNT(*) FROM candles_cache_data ccd
            WHERE NOT EXISTS (
                SELECT 1 FROM candles_cache cc WHERE cc.id = ccd.cache_id
            )
        """)
        orphaned_count = cursor.fetchone()[0]
        logger.info(f"Orphaned записей (без соответствующего cache_id): {orphaned_count:,}")
        
        if orphaned_count == 0:
            logger.info("Orphaned записей не найдено, удаление не требуется.")
            conn.close()
            return True
        
        # Удаляем orphaned записи
        logger.info("Начинаю удаление orphaned записей...")
        start_time = time.time()
        
        cursor.execute("""
            DELETE FROM candles_cache_data
            WHERE NOT EXISTS (
                SELECT 1 FROM candles_cache cc WHERE cc.id = candles_cache_data.cache_id
            )
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        elapsed_time = time.time() - start_time
        
        # Подсчитываем количество записей после удаления
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_after = cursor.fetchone()[0]
        
        logger.info(f"Удалено orphaned записей: {deleted_count:,}")
        logger.info(f"Записей после удаления: {total_after:,}")
        logger.info(f"Время выполнения: {elapsed_time:.2f} секунд")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при удалении orphaned записей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    db_path = os.environ.get('BOTS_DB_PATH')
    if not db_path:
        db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    logger.info(f"База данных: {db_path}")
    logger.info("=" * 80)
    
    success = remove_orphaned_candles(db_path)
    
    if success:
        logger.info("=" * 80)
        logger.info("Удаление orphaned записей завершено успешно")
        logger.info("=" * 80)
    else:
        logger.error("=" * 80)
        logger.error("Ошибка при удалении orphaned записей")
        logger.error("=" * 80)
        sys.exit(1)

