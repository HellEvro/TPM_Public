#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Очистка history_data.json от дубликатов и уменьшение размера"""

import os
import json
from pathlib import Path
from datetime import datetime

def main():
    history_file = Path('data/ai/history_data.json')
    
    if not history_file.exists():
        print("Файл history_data.json не найден")
        return
    
    # Создаем резервную копию
    backup_file = history_file.with_suffix('.json.backup')
    print(f"Создаю резервную копию: {backup_file}")
    import shutil
    shutil.copy2(history_file, backup_file)
    
    # Загружаем данные
    print("Загружаю данные...")
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    old_size = history_file.stat().st_size
    print(f"Старый размер: {old_size / 1024 / 1024:.2f} MB")
    
    # Собираем все уникальные сделки
    all_trade_ids = set()
    unique_trades = []
    
    history = data.get('history', [])
    print(f"Обрабатываю {len(history)} записей в history...")
    
    # Проходим по всем записям и собираем уникальные сделки
    for entry in history:
        for trade in entry.get('trades', []):
            trade_id = trade.get('id') or trade.get('timestamp')
            if trade_id and trade_id not in all_trade_ids:
                all_trade_ids.add(trade_id)
                unique_trades.append(trade)
    
    print(f"Найдено уникальных сделок: {len(unique_trades)}")
    
    # Создаем новую структуру с только уникальными сделками
    # Берем последние 100 записей и оставляем только новые сделки в каждой
    new_history = []
    processed_trade_ids = set()
    
    # Обрабатываем записи в обратном порядке (от новых к старым)
    for entry in reversed(history[-100:]):  # Берем последние 100 записей
        new_trades = []
        for trade in entry.get('trades', []):
            trade_id = trade.get('id') or trade.get('timestamp')
            if trade_id and trade_id not in processed_trade_ids:
                processed_trade_ids.add(trade_id)
                new_trades.append(trade)
        
        if new_trades:  # Добавляем только если есть новые сделки
            new_history.append({
                'timestamp': entry.get('timestamp'),
                'trades': new_trades,
                'statistics': entry.get('statistics', {}),
                'actions': entry.get('actions', [])
            })
    
    # Разворачиваем обратно (от старых к новым)
    new_history.reverse()
    
    # Обновляем данные
    data['history'] = new_history
    data['last_update'] = datetime.now().isoformat()
    
    # Обновляем latest - только последние 100 сделок
    latest_trades = unique_trades[-100:] if unique_trades else []
    data['latest'] = {
        'timestamp': datetime.now().isoformat(),
        'trades': latest_trades,
        'statistics': data.get('latest', {}).get('statistics', {})
    }
    
    # Сохраняем
    print("Сохраняю очищенные данные...")
    temp_file = history_file.with_suffix('.json.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Атомарная замена
    temp_file.replace(history_file)
    
    new_size = history_file.stat().st_size
    print(f"Новый размер: {new_size / 1024 / 1024:.2f} MB")
    print(f"Сжато: {(old_size - new_size) / 1024 / 1024:.2f} MB ({(1 - new_size/old_size)*100:.1f}%)")
    print()
    print(f"Записей в history: {len(new_history)} (было {len(history)})")
    print(f"Уникальных сделок: {len(unique_trades)}")
    print()
    print("Готово! Резервная копия сохранена в:", backup_file)

if __name__ == "__main__":
    main()

