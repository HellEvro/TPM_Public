#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для очистки старых свечей из ВСЕХ БД:
- bots_data.db -> candles_cache_data
- ai_data.db -> candles_history

Оставляет только последние N свечей для каждого символа и выполняет VACUUM для освобождения места.
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
logger = logging.getLogger('CleanupAllCandles')

DEFAULT_MAX_CANDLES_PER_SYMBOL = 5000  # Оставляем 5000 последних свечей

def cleanup_bots_db_candles(db_path: str, max_candles_per_symbol: int = DEFAULT_MAX_CANDLES_PER_SYMBOL):
    """Очистка candles_cache_data в bots_data.db"""
    logger.info("=" * 80)
    logger.info(f"🧹 Очистка candles_cache_data в: {db_path}")
    logger.info("=" * 80)
    
    if not Path(db_path).exists():
        logger.error(f"❌ Файл базы данных не найден: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем все символы из candles_cache
        cursor.execute("SELECT id, symbol FROM candles_cache")
        symbols_in_cache = cursor.fetchall()
        
        if not symbols_in_cache:
            logger.info("ℹ️ В кэше свечей нет символов для очистки.")
            conn.close()
            return True
        
        total_deleted_candles = 0
        total_symbols_processed = 0
        
        for cache_row in symbols_in_cache:
            cache_id = cache_row['id']
            symbol = cache_row['symbol']
            total_symbols_processed += 1
            logger.info(f"⚙️ Обработка символа: {symbol} (ID кэша: {cache_id})")
            
            # Получаем количество свечей для текущего символа
            cursor.execute("SELECT COUNT(*) FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
            current_candle_count = cursor.fetchone()[0]
            
            if current_candle_count <= max_candles_per_symbol:
                logger.info(f"   ℹ️ Для {symbol} всего {current_candle_count} свечей, очистка не требуется.")
                continue
            
            # Определяем, какие свечи нужно удалить
            # Оставляем последние N свечей, удаляя более старые
            cursor.execute(f"""
                DELETE FROM candles_cache_data
                WHERE id IN (
                    SELECT id FROM candles_cache_data
                    WHERE cache_id = ?
                    ORDER BY time DESC
                    LIMIT -1 OFFSET {max_candles_per_symbol}
                )
            """, (cache_id,))
            
            deleted_count = cursor.rowcount
            total_deleted_candles += deleted_count
            logger.info(f"   🗑️ Удалено {deleted_count} старых свечей для символа {symbol}.")
            
            # Обновляем candles_count в candles_cache
            cursor.execute("""
                UPDATE candles_cache
                SET candles_count = (SELECT COUNT(*) FROM candles_cache_data WHERE cache_id = ?)
                WHERE id = ?
            """, (cache_id, cache_id))
            
            conn.commit()  # Коммитим после каждого символа
        
        logger.info(f"✅ Очистка bots_data.db завершена. Обработано символов: {total_symbols_processed}, удалено свечей: {total_deleted_candles}.")
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке bots_data.db: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def cleanup_ai_db_candles(db_path: str, max_candles_per_symbol: int = DEFAULT_MAX_CANDLES_PER_SYMBOL):
    """Очистка candles_history в ai_data.db"""
    logger.info("=" * 80)
    logger.info(f"🧹 Очистка candles_history в: {db_path}")
    logger.info("=" * 80)
    
    if not Path(db_path).exists():
        logger.error(f"❌ Файл базы данных не найден: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем все уникальные символы и таймфреймы
        cursor.execute("SELECT DISTINCT symbol, timeframe FROM candles_history")
        symbol_timeframes = cursor.fetchall()
        
        if not symbol_timeframes:
            logger.info("ℹ️ В candles_history нет данных для очистки.")
            conn.close()
            return True
        
        total_deleted_candles = 0
        total_symbols_processed = 0
        
        for row in symbol_timeframes:
            symbol = row['symbol']
            timeframe = row['timeframe']
            total_symbols_processed += 1
            logger.info(f"⚙️ Обработка: {symbol} ({timeframe})")
            
            # Получаем количество свечей для текущего символа и таймфрейма
            cursor.execute("""
                SELECT COUNT(*) FROM candles_history 
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            current_candle_count = cursor.fetchone()[0]
            
            if current_candle_count <= max_candles_per_symbol:
                logger.info(f"   ℹ️ Для {symbol} ({timeframe}) всего {current_candle_count} свечей, очистка не требуется.")
                continue
            
            # Удаляем самые старые свечи, оставляя только последние MAX_CANDLES_PER_SYMBOL
            cursor.execute("""
                DELETE FROM candles_history
                WHERE id IN (
                    SELECT id FROM candles_history
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY candle_time ASC
                    LIMIT ?
                )
            """, (symbol, timeframe, current_candle_count - max_candles_per_symbol))
            
            deleted_count = cursor.rowcount
            total_deleted_candles += deleted_count
            logger.info(f"   🗑️ Удалено {deleted_count} старых свечей для {symbol} ({timeframe}).")
            
            conn.commit()  # Коммитим после каждого символа
        
        logger.info(f"✅ Очистка ai_data.db завершена. Обработано символов: {total_symbols_processed}, удалено свечей: {total_deleted_candles}.")
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке ai_data.db: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def vacuum_database(db_path: str, db_name: str):
    """Выполняет VACUUM для освобождения места"""
    logger.info("=" * 80)
    logger.info(f"⏳ Выполнение VACUUM для {db_name} (может занять много времени)...")
    logger.info("=" * 80)
    
    try:
        start_vacuum_time = time.time()
        conn = sqlite3.connect(str(db_path), timeout=300.0)  # Увеличенный timeout для VACUUM
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()
        end_vacuum_time = time.time()
        logger.info(f"✅ VACUUM для {db_name} завершен за {end_vacuum_time - start_vacuum_time:.2f} секунд.")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении VACUUM для {db_name}: {e}")
        return False

def main():
    logger.info("=" * 80)
    logger.info("🧹 ЗАПУСК ОЧИСТКИ ВСЕХ СВЕЧЕЙ ИЗ ВСЕХ БД")
    logger.info("=" * 80)
    
    # Пути к БД
    bots_db_path = os.environ.get('BOTS_DB_PATH')
    if not bots_db_path:
        bots_db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    ai_db_path = os.environ.get('AI_DB_PATH')
    if not ai_db_path:
        ai_db_path = str(PROJECT_ROOT / 'data' / 'ai_data.db')
    
    logger.info(f"📊 bots_data.db: {bots_db_path}")
    logger.info(f"📊 ai_data.db: {ai_db_path}")
    logger.info(f"📊 Максимум свечей на символ: {DEFAULT_MAX_CANDLES_PER_SYMBOL}")
    logger.info("=" * 80)
    
    # Очистка bots_data.db
    bots_success = cleanup_bots_db_candles(bots_db_path, DEFAULT_MAX_CANDLES_PER_SYMBOL)
    
    # Очистка ai_data.db
    ai_success = cleanup_ai_db_candles(ai_db_path, DEFAULT_MAX_CANDLES_PER_SYMBOL)
    
    # VACUUM для обеих БД
    if bots_success:
        vacuum_database(bots_db_path, "bots_data.db")
    
    if ai_success:
        vacuum_database(ai_db_path, "ai_data.db")
    
    logger.info("=" * 80)
    logger.info("🧹 ОЧИСТКА ЗАВЕРШЕНА")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

