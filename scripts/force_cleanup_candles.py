#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Принудительная очистка свечей - удаляет ВСЕ лишние свечи агрессивно
"""

import sys
import os
from pathlib import Path
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
import logging

setup_color_logging(console_log_levels=['+INFO', '+WARNING', '+ERROR'])
logger = logging.getLogger('ForceCleanupCandles')

MAX_CANDLES_PER_SYMBOL = 1000

def force_cleanup(db_path: str):
    """Принудительная очистка свечей"""
    logger.info("=" * 80)
    logger.info(f"ПРИНУДИТЕЛЬНАЯ ОЧИСТКА СВЕЧЕЙ: {db_path}")
    logger.info("=" * 80)
    
    if not Path(db_path).exists():
        logger.error(f"❌ Файл не найден: {db_path}")
        return False
    
    db_size_before = Path(db_path).stat().st_size
    logger.info(f"📊 Размер БД до очистки: {db_size_before / (1024**3):.2f} GB")
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=300.0)  # 5 минут timeout
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем общее количество свечей
        logger.info("⏳ Подсчет общего количества свечей...")
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_before = cursor.fetchone()[0]
        logger.info(f"📊 Всего свечей в БД: {total_before:,}")
        
        # Получаем все символы
        cursor.execute("SELECT id, symbol FROM candles_cache")
        symbols = cursor.fetchall()
        logger.info(f"📊 Символов в кэше: {len(symbols)}")
        
        total_deleted = 0
        symbols_processed = 0
        symbols_with_excess = 0
        
        for cache_row in symbols:
            cache_id = cache_row['id']
            symbol = cache_row['symbol']
            symbols_processed += 1
            
            if symbols_processed % 50 == 0:
                logger.info(f"⏳ Обработано символов: {symbols_processed}/{len(symbols)}")
            
            # Получаем количество свечей
            cursor.execute("SELECT COUNT(*) FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
            count = cursor.fetchone()[0]
            
            if count <= MAX_CANDLES_PER_SYMBOL:
                continue
            
            symbols_with_excess += 1
            excess = count - MAX_CANDLES_PER_SYMBOL
            
            # АГРЕССИВНОЕ УДАЛЕНИЕ: используем простой запрос без подзапросов
            # Удаляем все свечи, кроме последних MAX_CANDLES_PER_SYMBOL
            # Используем более простой подход: удаляем по времени напрямую
            try:
                # Получаем время последней свечи, которую нужно оставить
                cursor.execute(f"""
                    SELECT time FROM candles_cache_data
                    WHERE cache_id = ?
                    ORDER BY time DESC
                    LIMIT 1 OFFSET {MAX_CANDLES_PER_SYMBOL - 1}
                """, (cache_id,))
                
                result = cursor.fetchone()
                if result:
                    min_time = result[0]
                    # Удаляем все свечи старше этого времени
                    cursor.execute("""
                        DELETE FROM candles_cache_data
                        WHERE cache_id = ? AND time < ?
                    """, (cache_id, min_time))
                    
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    
                    if symbols_with_excess <= 10 or deleted > 10000:
                        logger.info(f"   🗑️ {symbol}: удалено {deleted:,} свечей (было {count:,}, осталось {count - deleted:,})")
                    
                    conn.commit()  # Коммитим после каждого символа
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка при очистке {symbol}: {e}")
                continue
        
        logger.info(f"\n✅ Очистка завершена:")
        logger.info(f"   Обработано символов: {symbols_processed}")
        logger.info(f"   Символов с превышением: {symbols_with_excess}")
        logger.info(f"   Удалено свечей: {total_deleted:,}")
        
        # Проверяем результат
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_after = cursor.fetchone()[0]
        logger.info(f"   Осталось свечей: {total_after:,}")
        logger.info(f"   Уменьшение: {total_before - total_after:,} свечей")
        
        conn.close()
        
        # Проверяем размер после очистки
        db_size_after = Path(db_path).stat().st_size
        freed_gb = (db_size_before - db_size_after) / (1024**3)
        logger.info(f"\n📊 Размер БД после очистки: {db_size_after / (1024**3):.2f} GB")
        logger.info(f"💾 Освобождено места: {freed_gb:.2f} GB")
        
        if freed_gb < 0.1:
            logger.warning("⚠️ Размер БД не уменьшился значительно!")
            logger.warning("💡 Нужно выполнить VACUUM для освобождения места")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Принудительная очистка свечей')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    force_cleanup(db_path)

