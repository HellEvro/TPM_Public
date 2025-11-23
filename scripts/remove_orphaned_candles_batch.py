#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для удаления orphaned свечей из candles_cache_data (пакетная обработка)
Удаляет свечи, у которых нет соответствующего cache_id в candles_cache
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

def remove_orphaned_candles_batch(db_path: str, batch_size: int = 100000):
    """Удаление orphaned свечей пакетами"""
    logger.info("=" * 80)
    logger.info(f"🧹 Удаление orphaned свечей из: {db_path}")
    logger.info("=" * 80)
    
    if not Path(db_path).exists():
        logger.error(f"❌ Файл базы данных не найден: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем список всех cache_id из candles_cache
        logger.info("📊 Получение списка валидных cache_id...")
        cursor.execute("SELECT id FROM candles_cache")
        valid_cache_ids = {row[0] for row in cursor.fetchall()}
        logger.info(f"✅ Найдено {len(valid_cache_ids)} валидных cache_id")
        
        if not valid_cache_ids:
            logger.warning("⚠️ Нет валидных cache_id, удаляем все свечи")
            cursor.execute("DELETE FROM candles_cache_data")
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"✅ Удалено {deleted_count:,} orphaned свечей")
            conn.close()
            return True
        
        # Получаем общее количество orphaned свечей
        logger.info("📊 Подсчет orphaned свечей...")
        placeholders = ','.join('?' * len(valid_cache_ids))
        cursor.execute(f"""
            SELECT COUNT(*) FROM candles_cache_data 
            WHERE cache_id NOT IN ({placeholders})
        """, list(valid_cache_ids))
        total_orphaned = cursor.fetchone()[0]
        logger.info(f"📊 Найдено {total_orphaned:,} orphaned свечей для удаления")
        
        if total_orphaned == 0:
            logger.info("✅ Orphaned свечей не найдено")
            conn.close()
            return True
        
        # Удаляем пакетами
        deleted_total = 0
        start_time = time.time()
        
        while True:
            cursor.execute(f"""
                DELETE FROM candles_cache_data 
                WHERE cache_id NOT IN ({placeholders})
                LIMIT ?
            """, list(valid_cache_ids) + [batch_size])
            
            deleted_count = cursor.rowcount
            if deleted_count == 0:
                break
            
            deleted_total += deleted_count
            conn.commit()
            
            elapsed = time.time() - start_time
            progress = (deleted_total / total_orphaned * 100) if total_orphaned > 0 else 0
            logger.info(f"🗑️ Удалено {deleted_total:,} / {total_orphaned:,} orphaned свечей ({progress:.1f}%) - {elapsed:.1f}s")
        
        elapsed_total = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ Удаление завершено: {deleted_total:,} orphaned свечей за {elapsed_total:.1f}s")
        logger.info("=" * 80)
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка удаления orphaned свечей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    db_path = PROJECT_ROOT / 'data' / 'bots_data.db'
    remove_orphaned_candles_batch(str(db_path), batch_size=100000)

