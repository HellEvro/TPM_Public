#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка дубликатов в bot_history.json"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_history(file_path):
    """Загружает историю из файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

def find_duplicates(data):
    """Находит дубликаты сделок"""
    trades = data.get('trades', [])
    history = data.get('history', [])
    
    # Группируем сделки по символу, цене входа и направлению
    trade_groups = defaultdict(list)
    for trade in trades:
        symbol = trade.get('symbol')
        entry_price = trade.get('entry_price')
        direction = trade.get('direction')
        status = trade.get('status')
        decision_source = trade.get('decision_source')
        
        if symbol and entry_price:
            key = (symbol, entry_price, direction)
            trade_groups[key].append({
                'trade': trade,
                'id': trade.get('id'),
                'status': status,
                'source': decision_source,
                'timestamp': trade.get('timestamp')
            })
    
    # Находим дубликаты (более одной сделки с одинаковыми параметрами)
    duplicates = {}
    for key, group in trade_groups.items():
        if len(group) > 1:
            symbol, entry_price, direction = key
            duplicates[key] = group
    
    return duplicates

def main():
    file_path = Path('data/bot_history.json')
    if not file_path.exists():
        print(f"Файл {file_path} не найден")
        return
    
    data = load_history(file_path)
    if not data:
        print("Не удалось загрузить данные")
        return
    
    trades = data.get('trades', [])
    history = data.get('history', [])
    
    print(f"Всего сделок: {len(trades)}")
    print(f"Всего записей истории: {len(history)}")
    print("="*70)
    
    # Ищем дубликаты
    duplicates = find_duplicates(data)
    
    if duplicates:
        print(f"\n[!] НАЙДЕНО ДУБЛИКАТОВ: {len(duplicates)} групп")
        print("="*70)
        
        for (symbol, entry_price, direction), group in list(duplicates.items())[:20]:
            print(f"\n[!] {symbol} | {direction} | entry={entry_price}")
            print(f"   Найдено {len(group)} сделок с одинаковыми параметрами:")
            
            for item in group:
                trade = item['trade']
                print(f"   - ID: {item['id']}")
                print(f"     Source: {item['source']}")
                print(f"     Status: {item['status']}")
                print(f"     Timestamp: {item['timestamp']}")
                print(f"     Exit: {trade.get('exit_price', 'N/A')}")
                print(f"     PnL: {trade.get('pnl', 'N/A')}")
                print()
    else:
        print("\n[OK] Дубликатов не найдено")
    
    # Проверяем открытые позиции, которые могут дублироваться
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    print(f"\n[INFO] Открытых позиций: {len(open_trades)}")
    
    # Группируем открытые позиции по символу
    open_by_symbol = defaultdict(list)
    for trade in open_trades:
        symbol = trade.get('symbol')
        open_by_symbol[symbol].append(trade)
    
    multiple_open = {s: ts for s, ts in open_by_symbol.items() if len(ts) > 1}
    if multiple_open:
        print(f"\n[!] СИМВОЛЫ С НЕСКОЛЬКИМИ ОТКРЫТЫМИ ПОЗИЦИЯМИ: {len(multiple_open)}")
        for symbol, trades_list in list(multiple_open.items())[:10]:
            print(f"   {symbol}: {len(trades_list)} открытых позиций")
            for t in trades_list:
                print(f"     - {t.get('id')} | source={t.get('decision_source')} | entry={t.get('entry_price')}")

if __name__ == '__main__':
    main()

