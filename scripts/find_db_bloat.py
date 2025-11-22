#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Поиск реальной причины раздувания БД
"""

import sys
import os
from pathlib import Path
import sqlite3

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def find_bloat(db_path: str):
    """Поиск причин раздувания БД"""
    print("=" * 80)
    print(f"ПОИСК ПРИЧИН РАЗДУВАНИЯ БД: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    db_size = Path(db_path).stat().st_size
    print(f"\n📊 Размер БД: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=120.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Параметры БД
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]
        
        used_pages = page_count - freelist_count
        used_size = used_pages * page_size
        free_size = freelist_count * page_size
        
        print(f"\n📄 СТРУКТУРА БД:")
        print(f"   Размер страницы: {page_size} байт")
        print(f"   Всего страниц: {page_count:,}")
        print(f"   Используется страниц: {used_pages:,} ({used_size / (1024**3):.2f} GB)")
        print(f"   Свободных страниц: {freelist_count:,} ({free_size / (1024**3):.2f} GB)")
        print(f"   Процент свободного места: {(freelist_count / page_count * 100) if page_count > 0 else 0:.1f}%")
        
        # Анализ всех таблиц
        print(f"\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ ТАБЛИЦ:")
        print("=" * 80)
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        total_data_size = 0
        table_details = []
        
        for table in tables:
            try:
                # Количество записей
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                if row_count == 0:
                    continue
                
                # Пробуем получить реальный размер через dbstat
                table_size = 0
                try:
                    cursor.execute(f"""
                        SELECT SUM(pgsize) as size 
                        FROM dbstat 
                        WHERE name = ? AND aggregate = 1
                    """, (table,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        table_size = result[0]
                except:
                    # Если dbstat не работает, оцениваем
                    if 'candles_cache_data' in table:
                        table_size = row_count * 70
                    elif 'candles' in table:
                        table_size = row_count * 50
                    elif 'trades' in table or 'history' in table:
                        table_size = row_count * 500
                    else:
                        table_size = row_count * 200
                
                total_data_size += table_size
                
                # Получаем информацию об индексах для этой таблицы
                cursor.execute(f"""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='index' AND tbl_name = ?
                """, (table,))
                index_count = cursor.fetchone()[0]
                
                table_details.append({
                    'name': table,
                    'rows': row_count,
                    'size': table_size,
                    'indexes': index_count
                })
                
            except Exception as e:
                print(f"   ⚠️ Ошибка анализа {table}: {e}")
        
        # Сортируем по размеру
        table_details.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"{'Таблица':<35} {'Записей':>15} {'Размер (GB)':>12} {'Индексов':>10}")
        print("-" * 80)
        
        for detail in table_details:
            size_gb = detail['size'] / (1024**3)
            if size_gb > 0.01 or detail['rows'] > 1000:
                print(f"{detail['name']:<35} {detail['rows']:>15,} {size_gb:>12.2f} {detail['indexes']:>10}")
        
        print("-" * 80)
        print(f"{'ИТОГО (данные)':<35} {'':>15} {total_data_size / (1024**3):>12.2f}")
        print(f"{'Разница (индексы+свободное)':<35} {'':>15} {(used_size - total_data_size) / (1024**3):>12.2f}")
        
        # Анализ индексов
        print(f"\n📋 АНАЛИЗ ИНДЕКСОВ:")
        print("=" * 80)
        
        try:
            cursor.execute("""
                SELECT 
                    tbl_name,
                    name,
                    sql
                FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name
            """)
            indexes = cursor.fetchall()
            
            if indexes:
                index_sizes = {}
                for idx in indexes:
                    tbl = idx[0]
                    idx_name = idx[1]
                    try:
                        # Пробуем получить размер индекса
                        cursor.execute(f"""
                            SELECT SUM(pgsize) as size 
                            FROM dbstat 
                            WHERE name = ? AND aggregate = 1
                        """, (idx_name,))
                        result = cursor.fetchone()
                        if result and result[0]:
                            if tbl not in index_sizes:
                                index_sizes[tbl] = 0
                            index_sizes[tbl] += result[0]
                    except:
                        pass
                
                print(f"   Всего индексов: {len(indexes)}")
                if index_sizes:
                    print(f"   Размеры индексов по таблицам:")
                    for tbl, size in sorted(index_sizes.items(), key=lambda x: x[1], reverse=True):
                        print(f"      {tbl}: {size / (1024**3):.2f} GB")
            else:
                print(f"   Нет пользовательских индексов")
        except Exception as e:
            print(f"   ⚠️ Ошибка анализа индексов: {e}")
        
        # Проверяем candles_cache_data детально
        if 'candles_cache_data' in [t['name'] for t in table_details]:
            print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ candles_cache_data:")
            print("=" * 80)
            try:
                cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
                total_candles = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
                unique_symbols = cursor.fetchone()[0]
                
                print(f"   Всего свечей: {total_candles:,}")
                print(f"   Уникальных символов: {unique_symbols}")
                
                if unique_symbols > 0:
                    avg = total_candles / unique_symbols
                    print(f"   Среднее на символ: {avg:,.0f}")
                    
                    # Проверяем распределение
                    cursor.execute("""
                        SELECT 
                            cc.symbol,
                            COUNT(ccd.id) as count
                        FROM candles_cache_data ccd
                        JOIN candles_cache cc ON ccd.cache_id = cc.id
                        GROUP BY cc.symbol
                        ORDER BY count DESC
                        LIMIT 20
                    """)
                    top = cursor.fetchall()
                    
                    print(f"\n   ТОП-20 символов по количеству свечей:")
                    excess_total = 0
                    for row in top:
                        symbol = row[0]
                        count = row[1]
                        if count > 5000:
                            excess = count - 5000
                            excess_total += excess
                            print(f"      {symbol:20} {count:>10,} свечей ⚠️ (+{excess:,} лишних)")
                        else:
                            print(f"      {symbol:20} {count:>10,} свечей")
                    
                    if excess_total > 0:
                        excess_gb = (excess_total * 70) / (1024**3)
                        print(f"\n   ⚠️ ВСЕГО ЛИШНИХ СВЕЧЕЙ: {excess_total:,} ({excess_gb:.2f} GB)")
            except Exception as e:
                print(f"   ⚠️ Ошибка: {e}")
                import traceback
                traceback.print_exc()
        
        conn.close()
        
        # Выводы
        print(f"\n" + "=" * 80)
        print("ВЫВОДЫ:")
        print("=" * 80)
        
        if used_size - total_data_size > db_size * 0.3:  # Больше 30% разницы
            print(f"⚠️ Большая разница между размером данных и используемым местом!")
            print(f"   Это может указывать на:")
            print(f"   - Большие индексы")
            print(f"   - Сильную фрагментацию")
            print(f"   - Много свободного места в страницах")
            print(f"\n💡 Рекомендация: Выполните VACUUM для дефрагментации")
        
        if freelist_count > page_count * 0.1:  # Больше 10% свободных страниц
            print(f"⚠️ Много свободных страниц: {freelist_count:,} ({free_size / (1024**3):.2f} GB)")
            print(f"💡 Рекомендация: VACUUM освободит это место")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Поиск причин раздувания БД')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    find_bloat(db_path)

