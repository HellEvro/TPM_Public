#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка размера БД на удаленном ПК
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

# Путь к удаленному ПК
REMOTE_PATH = Path(r'\\Evromini\projects\InfoBot')
db_path = REMOTE_PATH / 'data' / 'bots_data.db'

print("=" * 80)
print(f"ПРОВЕРКА БД НА УДАЛЕННОМ ПК: {REMOTE_PATH}")
print("=" * 80)

if not REMOTE_PATH.exists():
    print(f"❌ Не удалось подключиться к {REMOTE_PATH}")
    print("💡 Проверьте, что удаленный ПК доступен и путь правильный")
    sys.exit(1)

if not db_path.exists():
    print(f"❌ БД не найдена: {db_path}")
    sys.exit(1)

# Размер файла
file_size_mb = db_path.stat().st_size / (1024 * 1024)
file_size_gb = file_size_mb / 1024
print(f"\n📊 Общий размер файла: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

# Проверяем WAL файлы
wal_path = Path(str(db_path) + '-wal')
shm_path = Path(str(db_path) + '-shm')

if wal_path.exists():
    wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
    wal_size_gb = wal_size_mb / 1024
    print(f"📊 Размер WAL файла: {wal_size_mb:.2f} MB ({wal_size_gb:.2f} GB)")
    total_size_mb = file_size_mb + wal_size_mb
    total_size_gb = total_size_mb / 1024
    print(f"📊 Общий размер (БД + WAL): {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
    
    if wal_size_mb > 100:
        print(f"\n⚠️ WAL файл очень большой ({wal_size_gb:.2f} GB)!")
        print("💡 Это может быть причиной зависания PRAGMA quick_check")
        print("💡 Рекомендуется выполнить PRAGMA wal_checkpoint(TRUNCATE)")

if shm_path.exists():
    shm_size_mb = shm_path.stat().st_size / (1024 * 1024)
    print(f"📊 Размер SHM файла: {shm_size_mb:.2f} MB")

print("\n" + "=" * 80)
print("ПОДКЛЮЧЕНИЕ К БД ДЛЯ АНАЛИЗА ТАБЛИЦ:")
print("=" * 80)

try:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    cursor = conn.cursor()
    
    # Получаем список всех таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\n📋 Найдено таблиц: {len(tables)}")
    print("\n" + "=" * 80)
    print("РАЗМЕР КАЖДОЙ ТАБЛИЦЫ:")
    print("=" * 80)
    
    total_rows = 0
    table_info_list = []
    
    for table in tables:
        try:
            # Количество строк
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Получаем информацию о колонках
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            table_info_list.append({
                'name': table,
                'rows': row_count,
                'columns': len(columns)
            })
            
            total_rows += row_count
            
            print(f"\n📊 {table}:")
            print(f"   Записей: {row_count:,}")
            print(f"   Колонок: {len(columns)}")
            
            if row_count > 100000:
                print(f"   ⚠️ ОЧЕНЬ БОЛЬШАЯ ТАБЛИЦА!")
            
        except Exception as e:
            print(f"\n❌ Ошибка при анализе таблицы {table}: {e}")
    
    # Сортируем по количеству записей
    table_info_list.sort(key=lambda x: x['rows'], reverse=True)
    
    print("\n" + "=" * 80)
    print("ТАБЛИЦЫ ПО КОЛИЧЕСТВУ ЗАПИСЕЙ (топ-10):")
    print("=" * 80)
    for i, table_info in enumerate(table_info_list[:10], 1):
        print(f"{i}. {table_info['name']}: {table_info['rows']:,} записей")
    
    print(f"\n📊 Всего записей во всех таблицах: {total_rows:,}")
    
    # Детальный анализ больших таблиц
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ БОЛЬШИХ ТАБЛИЦ:")
    print("=" * 80)
    
    large_tables = ['candles_cache_data', 'bot_trades_history', 'rsi_cache_coins']
    for table in large_tables:
        if table in [t['name'] for t in table_info_list]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                if table == 'candles_cache_data':
                    print(f"\n📊 {table}:")
                    print(f"   Всего свечей: {count:,}")
                    if count > 0:
                        # Примерный размер одной свечи (8 колонок: id, cache_id, time, open, high, low, close, volume)
                        # примерно 8 * 8 байт = 64 байта + overhead
                        estimated_size_mb = (count * 100) / (1024 * 1024)  # ~100 байт на свечу
                        print(f"   Примерный размер данных: ~{estimated_size_mb:.2f} MB")
                
                elif table == 'bot_trades_history':
                    print(f"\n📊 {table}:")
                    print(f"   Всего сделок: {count:,}")
                    if count > 0:
                        # Примерный размер одной сделки (32 колонки)
                        estimated_size_mb = (count * 500) / (1024 * 1024)  # ~500 байт на сделку
                        print(f"   Примерный размер данных: ~{estimated_size_mb:.2f} MB")
                
            except Exception as e:
                print(f"\n❌ Ошибка при анализе {table}: {e}")
    
    conn.close()
    
except Exception as e:
    print(f"\n❌ Ошибка подключения к БД: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ:")
print("=" * 80)
if wal_path.exists() and wal_path.stat().st_size > 100 * 1024 * 1024:
    wal_size_gb = wal_path.stat().st_size / (1024 * 1024 * 1024)
    print(f"⚠️ WAL файл очень большой ({wal_size_gb:.2f} GB)!")
    print("💡 Это основная причина зависания PRAGMA quick_check")
    print("💡 Решение: выполнить PRAGMA wal_checkpoint(TRUNCATE) для сброса WAL")
    print("💡 Или пропустить проверку целостности для больших БД (уже реализовано)")

print("=" * 80)

