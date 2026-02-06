#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка размера и содержимого history_data.json"""

import os
import json
from pathlib import Path

def main():
    history_file = Path('data/ai/history_data.json')
    
    if not history_file.exists():
        print("Файл history_data.json не найден")
        return
    
    size = history_file.stat().st_size
    print(f"Размер файла: {size / 1024 / 1024:.2f} MB ({size:,} байт)")
    print()
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        history = data.get('history', [])
        latest = data.get('latest', {})
        
        print(f"Записей в history: {len(history)}")
        print(f"Сделок в latest: {len(latest.get('trades', []))}")
        
        total_trades = 0
        total_actions = 0
        for entry in history:
            trades = entry.get('trades', [])
            actions = entry.get('actions', [])
            total_trades += len(trades)
            total_actions += len(actions)
        
        print(f"Всего сделок в history: {total_trades:,}")
        print(f"Всего действий в history: {total_actions:,}")
        print()
        
        # Показываем размер каждой записи
        if history:
            print("Размеры записей в history (первые 10):")
            for i, entry in enumerate(history[:10]):
                entry_size = len(json.dumps(entry))
                trades_count = len(entry.get('trades', []))
                actions_count = len(entry.get('actions', []))
                print(f"   Запись {i+1}: {entry_size / 1024:.2f} KB ({trades_count} сделок, {actions_count} действий)")
        
        if len(history) > 10:
            print(f"   ... и еще {len(history) - 10} записей")
        print()
        
        # Проблема: каждая запись содержит ВСЕ сделки, что приводит к дублированию
        print("ПРОБЛЕМА:")
        print("   Каждая запись в history содержит ВСЕ сделки из bot_history.json")
        print("   Это приводит к огромному дублированию данных!")
        print("   При 1000 записей и 1000 сделок = 1,000,000 записей сделок!")
        print()
        
        # Рекомендации
        print("РЕКОМЕНДАЦИИ:")
        print("   1. Ограничить количество записей в history (сейчас 1000)")
        print("   2. Сохранять только новые сделки, а не все каждый раз")
        print("   3. Использовать БД вместо JSON для хранения истории")
        print("   4. Очистить старые записи из history")
        
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

