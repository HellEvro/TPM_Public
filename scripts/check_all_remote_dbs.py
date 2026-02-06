#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка всех БД на удаленном ПК
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

print("=" * 80)
print(f"ПРОВЕРКА ВСЕХ БД НА УДАЛЕННОМ ПК: {REMOTE_PATH}")
print("=" * 80)

if not REMOTE_PATH.exists():
    print(f"❌ Не удалось подключиться к {REMOTE_PATH}")
    sys.exit(1)

# Список всех БД для проверки
db_files = [
    'data/bots_data.db',
    'data/ai_data.db',
    'data/app_data.db',
    'license_generator/licenses.db',
]

print("\n" + "=" * 80)
print("РАЗМЕРЫ БАЗ ДАННЫХ:")
print("=" * 80)

total_size_mb = 0
total_size_gb = 0

for db_file in db_files:
    db_path = REMOTE_PATH / db_file
    
    if not db_path.exists():
        print(f"\n❌ {db_file}: не найден")
        continue
    
    # Размер основного файла
    file_size_mb = db_path.stat().st_size / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    total_size_mb += file_size_mb
    total_size_gb += file_size_gb
    
    # Проверяем WAL и SHM файлы
    wal_path = Path(str(db_path) + '-wal')
    shm_path = Path(str(db_path) + '-shm')
    
    wal_size_mb = 0
    shm_size_mb = 0
    
    if wal_path.exists():
        wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
        wal_size_gb = wal_size_mb / 1024
    
    if shm_path.exists():
        shm_size_mb = shm_path.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 {db_file}:")
    print(f"   Основной файл: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    
    if wal_path.exists():
        print(f"   WAL файл: {wal_size_mb:.2f} MB ({wal_size_gb:.2f} GB)")
        if wal_size_mb > 100:
            print(f"   ⚠️ WAL файл очень большой!")
        total_size_mb += wal_size_mb
        total_size_gb += wal_size_mb / 1024
    
    if shm_path.exists():
        print(f"   SHM файл: {shm_size_mb:.2f} MB")
    
    total_with_wal = file_size_mb + wal_size_mb
    total_with_wal_gb = total_with_wal / 1024
    print(f"   ИТОГО (БД + WAL): {total_with_wal:.2f} MB ({total_with_wal_gb:.2f} GB)")
    
    # Анализ таблиц для больших БД
    if file_size_mb > 100 or (wal_path.exists() and wal_size_mb > 100):
        print(f"\n   📋 Анализ таблиц:")
        try:
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            large_tables = []
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 10000:
                        large_tables.append((table, count))
                except:
                    pass
            
            if large_tables:
                large_tables.sort(key=lambda x: x[1], reverse=True)
                print(f"   Большие таблицы (>10k записей):")
                for table, count in large_tables[:10]:
                    print(f"      - {table}: {count:,} записей")
            
            conn.close()
        except Exception as e:
            print(f"   ⚠️ Ошибка анализа: {e}")

print("\n" + "=" * 80)
print("ИТОГО:")
print("=" * 80)
print(f"Общий размер всех БД: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")

print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ:")
print("=" * 80)

# Проверяем bots_data.db отдельно
bots_db_path = REMOTE_PATH / 'data' / 'bots_data.db'
if bots_db_path.exists():
    bots_wal = Path(str(bots_db_path) + '-wal')
    
    if bots_wal.exists():
        wal_size_gb = bots_wal.stat().st_size / (1024 * 1024 * 1024)
        if wal_size_gb > 1:
            print(f"⚠️ bots_data.db-wal очень большой ({wal_size_gb:.2f} GB)!")
            print("💡 Выполните PRAGMA wal_checkpoint(TRUNCATE) для сброса WAL")
            print("💡 Или запустите скрипт очистки свечей")
    
    # Проверяем candles_cache_data
    try:
        conn = sqlite3.connect(str(bots_db_path), timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        candles_count = cursor.fetchone()[0]
        conn.close()
        
        if candles_count > 100000:
            print(f"\n⚠️ candles_cache_data содержит {candles_count:,} свечей!")
            print("💡 Запустите: python scripts/cleanup_old_candles.py")
            print("💡 Или установите BOTS_DB_PATH=\\Evromini\projects\InfoBot\data\bots_data.db")
    except Exception as e:
        print(f"⚠️ Не удалось проверить candles_cache_data: {e}")

print("=" * 80)

