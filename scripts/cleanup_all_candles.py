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

DEFAULT_MAX_CANDLES_PER_SYMBOL = 1000  # Оставляем 1000 последних свечей (~250 дней для 6h свечей)

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
            # Оптимизированный запрос: сначала находим минимальное время для сохранения
            cursor.execute(f"""
                SELECT MIN(time) FROM (
                    SELECT time FROM candles_cache_data
                    WHERE cache_id = ?
                    ORDER BY time DESC
                    LIMIT {max_candles_per_symbol}
                )
            """, (cache_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                min_time_to_keep = result[0]
                # Удаляем все свечи старше минимального времени
                cursor.execute("""
                    DELETE FROM candles_cache_data
                    WHERE cache_id = ? AND time < ?
                """, (cache_id, min_time_to_keep))
                deleted_count = cursor.rowcount
            else:
                deleted_count = 0
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

def vacuum_database(db_path: str, db_name: str, skip_vacuum: bool = False):
    """Выполняет VACUUM для освобождения места или альтернативные операции"""
    if skip_vacuum:
        logger.info("=" * 80)
        logger.info(f"⏭️ Пропуск VACUUM для {db_name} (опция --skip-vacuum)")
        logger.info("=" * 80)
        return True
    
    logger.info("=" * 80)
    logger.info(f"⏳ Выполнение операций оптимизации для {db_name}...")
    logger.info("=" * 80)
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        cursor = conn.cursor()
        
        # Сначала делаем checkpoint для WAL файлов (быстрее и безопаснее)
        logger.info(f"   [1/3] Выполнение PRAGMA wal_checkpoint(TRUNCATE)...")
        try:
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info(f"   ✅ Checkpoint выполнен")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка checkpoint: {e}")
        
        # Проверяем размер БД перед VACUUM
        db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        logger.info(f"   [2/3] Размер БД перед оптимизацией: {db_size_mb:.2f} MB")
        
        # Если БД очень большая (>5 GB), предлагаем пропустить VACUUM
        if db_size_mb > 5000:
            logger.warning(f"   ⚠️ БД очень большая ({db_size_mb:.2f} MB), VACUUM может занять много времени!")
            logger.warning(f"   💡 Рекомендуется запустить VACUUM отдельно или использовать --skip-vacuum")
            logger.info(f"   [3/3] Пропуск VACUUM для {db_name} (БД слишком большая)")
            conn.close()
            return True
        
        # Выполняем VACUUM только для небольших БД
        logger.info(f"   [3/3] Выполнение VACUUM (может занять время)...")
        start_vacuum_time = time.time()
        
        # Устанавливаем увеличенный timeout
        conn.close()
        conn = sqlite3.connect(str(db_path), timeout=600.0)  # 10 минут для VACUUM
        cursor = conn.cursor()
        
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()
        
        end_vacuum_time = time.time()
        elapsed_minutes = (end_vacuum_time - start_vacuum_time) / 60
        
        # Проверяем размер после VACUUM
        new_db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        freed_mb = db_size_mb - new_db_size_mb
        
        logger.info(f"   ✅ VACUUM для {db_name} завершен за {elapsed_minutes:.1f} минут")
        logger.info(f"   📊 Размер БД после оптимизации: {new_db_size_mb:.2f} MB")
        if freed_mb > 0:
            logger.info(f"   💾 Освобождено места: {freed_mb:.2f} MB")
        
        return True
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            logger.warning(f"   ⚠️ БД заблокирована другим процессом, пропускаем VACUUM")
            return False
        logger.error(f"   ❌ Ошибка при выполнении VACUUM для {db_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"   ❌ Ошибка при выполнении VACUUM для {db_name}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Очистка старых свечей из всех БД')
    parser.add_argument('--skip-vacuum', action='store_true', 
                       help='Пропустить VACUUM (рекомендуется для больших БД)')
    args = parser.parse_args()
    
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
    if args.skip_vacuum:
        logger.info(f"⏭️ VACUUM будет пропущен (--skip-vacuum)")
    logger.info("=" * 80)
    
    # Очистка bots_data.db
    bots_success = cleanup_bots_db_candles(bots_db_path, DEFAULT_MAX_CANDLES_PER_SYMBOL)
    
    # Очистка ai_data.db
    ai_success = cleanup_ai_db_candles(ai_db_path, DEFAULT_MAX_CANDLES_PER_SYMBOL)
    
    # VACUUM для обеих БД (или пропуск)
    if bots_success:
        vacuum_database(bots_db_path, "bots_data.db", skip_vacuum=args.skip_vacuum)
    
    if ai_success:
        vacuum_database(ai_db_path, "ai_data.db", skip_vacuum=args.skip_vacuum)
    
    logger.info("=" * 80)
    logger.info("🧹 ОЧИСТКА ЗАВЕРШЕНА")
    if args.skip_vacuum:
        logger.info("💡 Для полной оптимизации запустите VACUUM отдельно после закрытия всех соединений с БД")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

