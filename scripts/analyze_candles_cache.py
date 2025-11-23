#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ таблицы candles_cache_data"""

import sqlite3
import sys
from pathlib import Path

def analyze_candles_cache(db_path):
    """Анализ candles_cache_data"""
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    try:
        # Общее количество записей
        cursor.execute('SELECT COUNT(*) FROM candles_cache_data')
        total = cursor.fetchone()[0]
        print(f'Всего записей в candles_cache_data: {total:,}')
        
        # Количество уникальных cache_id
        cursor.execute('SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data')
        unique_cache_ids = cursor.fetchone()[0]
        print(f'Уникальных cache_id: {unique_cache_ids:,}')
        
        # Топ-10 cache_id по количеству записей
        cursor.execute('''
            SELECT cache_id, COUNT(*) as cnt 
            FROM candles_cache_data 
            GROUP BY cache_id 
            ORDER BY cnt DESC 
            LIMIT 10
        ''')
        print('\nТоп-10 cache_id по количеству записей:')
        for row in cursor.fetchall():
            print(f'  cache_id={row[0]}: {row[1]:,} записей')
        
        # Среднее количество записей на cache_id
        if unique_cache_ids > 0:
            avg = total / unique_cache_ids
            print(f'\nСреднее количество записей на cache_id: {avg:,.0f}')
        
        # Проверка структуры таблицы
        cursor.execute('PRAGMA table_info(candles_cache_data)')
        columns = cursor.fetchall()
        print('\nСтруктура таблицы candles_cache_data:')
        for col in columns:
            print(f'  {col[1]} ({col[2]})')
            
    finally:
        conn.close()

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/bots_data.db'
    if not Path(db_path).exists():
        print(f'ОШИБКА: Файл {db_path} не найден!')
        sys.exit(1)
    
    analyze_candles_cache(db_path)

