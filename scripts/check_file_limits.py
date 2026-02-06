#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки лимитов файлов данных
Показывает:
1. Лимиты для simulated_trades.json
2. Лимиты для exchange_trades_history.json
3. Размеры файлов и сколько записей они содержат
"""

import json
import os
import sys
from pathlib import Path

def get_file_size_mb(file_path):
    """Возвращает размер файла в MB"""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path) / 1024 / 1024

def check_simulated_trades():
    """Проверяет simulated_trades.json"""
    file_path = Path('data/ai/simulated_trades.json')
    
    print("="*70)
    print("SIMULATED_TRADES.JSON")
    print("="*70)
    
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        print("Лимит: 50000 сделок (~11 MB)")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_trades = data.get('total_trades', 0)
        trades = data.get('trades', [])
        file_size_mb = get_file_size_mb(file_path)
        
        print(f"Файл: {file_path}")
        print(f"Размер: {file_size_mb:.2f} MB")
        print(f"Всего сделок: {total_trades}")
        print(f"Лимит: 50000 сделок (~11 MB)")
        
        if total_trades >= 50000:
            print("ВНИМАНИЕ: Достигнут лимит! Старые сделки будут удаляться.")
        else:
            remaining = 50000 - total_trades
            print(f"Осталось места: {remaining} сделок")
        
        # Оценка размера одной сделки
        if trades:
            sample_size = len(json.dumps(trades[0], ensure_ascii=False))
            estimated_total_size = sample_size * 50000 / 1024 / 1024
            print(f"Оценка размера при лимите: ~{estimated_total_size:.2f} MB")
        
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")

def check_exchange_trades():
    """Проверяет exchange_trades_history.json"""
    file_path = Path('data/ai/exchange_trades_history.json')
    
    print("\n" + "="*70)
    print("EXCHANGE_TRADES_HISTORY.JSON")
    print("="*70)
    
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        print("Лимит: 100000 сделок (~22 MB)")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trades = data.get('trades', [])
        total_trades = len(trades)
        file_size_mb = get_file_size_mb(file_path)
        
        print(f"Файл: {file_path}")
        print(f"Размер: {file_size_mb:.2f} MB")
        print(f"Всего сделок: {total_trades}")
        print(f"Лимит: 100000 сделок (~22 MB)")
        
        if total_trades >= 100000:
            print("ВНИМАНИЕ: Достигнут лимит! Старые сделки будут удаляться.")
        else:
            remaining = 100000 - total_trades
            print(f"Осталось места: {remaining} сделок")
        
        # Оценка размера одной сделки
        if trades:
            sample_size = len(json.dumps(trades[0], ensure_ascii=False))
            estimated_total_size = sample_size * 100000 / 1024 / 1024
            print(f"Оценка размера при лимите: ~{estimated_total_size:.2f} MB")
        
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")

def check_log_files():
    """Проверяет размеры лог-файлов"""
    log_files = [
        'logs/ai.log',
        'logs/bots.log',
        'logs/app.log',
        'logs/ai_trace.log'
    ]
    
    print("\n" + "="*70)
    print("ЛОГ-ФАЙЛЫ (лимит: 10 MB каждый)")
    print("="*70)
    
    for log_file in log_files:
        file_path = Path(log_file)
        if file_path.exists():
            size_mb = get_file_size_mb(file_path)
            status = "ПРЕВЫШЕН!" if size_mb > 10 else "OK"
            print(f"{log_file}: {size_mb:.2f} MB [{status}]")
        else:
            print(f"{log_file}: не найден")

def main():
    check_simulated_trades()
    check_exchange_trades()
    check_log_files()
    
    print("\n" + "="*70)
    print("ПРИМЕЧАНИЯ:")
    print("="*70)
    print("1. simulated_trades.json: лимит 50000 сделок (~11 MB)")
    print("   - Старые сделки автоматически удаляются при превышении")
    print("2. exchange_trades_history.json: лимит 100000 сделок (~22 MB)")
    print("   - Старые сделки автоматически удаляются при превышении")
    print("3. Лог-файлы: лимит 10 MB каждый")
    print("   - Автоматически перезаписываются при превышении")
    print("   - Ротация настроена через RotatingFileHandlerWithSizeLimit")
    return 0

if __name__ == '__main__':
    sys.exit(main())

