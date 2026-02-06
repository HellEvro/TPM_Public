#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение двух файлов bot_history.json для поиска различий"""

import json
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime

# Исправляем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description='Сравнение двух файлов bot_history.json')
    parser.add_argument('--file1', '-f1', type=str, required=True,
                       help='Первый файл (исходный)')
    parser.add_argument('--file2', '-f2', type=str, required=True,
                       help='Второй файл (измененный)')
    return parser.parse_args()

def load_file(file_path):
    """Загружает файл"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_files(file1_path, file2_path):
    """Сравнивает два файла и находит различия"""
    print("="*70)
    print("СРАВНЕНИЕ ФАЙЛОВ bot_history.json")
    print("="*70)
    print(f"\nФайл 1 (исходный): {file1_path}")
    print(f"Файл 2 (измененный): {file2_path}")
    
    data1 = load_file(file1_path)
    data2 = load_file(file2_path)
    
    history1 = {e.get('id'): e for e in data1.get('history', [])}
    history2 = {e.get('id'): e for e in data2.get('history', [])}
    
    trades1 = {t.get('id'): t for t in data1.get('trades', [])}
    trades2 = {t.get('id'): t for t in data2.get('trades', [])}
    
    # Находим новые записи
    new_history_ids = set(history2.keys()) - set(history1.keys())
    new_trade_ids = set(trades2.keys()) - set(trades1.keys())
    
    print(f"\nИСХОДНЫЙ ФАЙЛ:")
    print(f"  История: {len(history1)} записей")
    print(f"  Сделки: {len(trades1)} сделок")
    
    print(f"\nИЗМЕНЕННЫЙ ФАЙЛ:")
    print(f"  История: {len(history2)} записей")
    print(f"  Сделки: {len(trades2)} сделок")
    
    print(f"\nНОВЫЕ ЗАПИСИ:")
    print(f"  История: {len(new_history_ids)} новых записей")
    print(f"  Сделки: {len(new_trade_ids)} новых сделок")
    
    if new_history_ids:
        print(f"\n" + "="*70)
        print("НОВЫЕ ЗАПИСИ ИСТОРИИ:")
        print("="*70)
        for entry_id in sorted(new_history_ids):
            entry = history2[entry_id]
            bot_id = entry.get('bot_id', 'N/A')
            decision_source = entry.get('decision_source', 'NONE')
            is_simulated = entry.get('is_simulated', 'NONE')
            action_type = entry.get('action_type', 'N/A')
            symbol = entry.get('symbol', 'N/A')
            timestamp = entry.get('timestamp', 'N/A')
            
            print(f"\nID: {entry_id}")
            print(f"  bot_id: {bot_id}")
            print(f"  symbol: {symbol}")
            print(f"  action_type: {action_type}")
            print(f"  decision_source: {decision_source}")
            print(f"  is_simulated: {is_simulated}")
            print(f"  timestamp: {timestamp}")
            
            # Проверяем, проблемная ли это запись
            if decision_source == 'AI' and is_simulated != True:
                print(f"  ⚠️ ПРОБЛЕМА: decision_source=AI но is_simulated={is_simulated}!")
            if bot_id and len(str(bot_id)) > 10 and '_' in str(bot_id) and is_simulated != True:
                print(f"  ⚠️ ПРОБЛЕМА: длинный bot_id ({bot_id}) но is_simulated={is_simulated}!")
    
    if new_trade_ids:
        print(f"\n" + "="*70)
        print("НОВЫЕ СДЕЛКИ:")
        print("="*70)
        for trade_id in sorted(new_trade_ids):
            trade = trades2[trade_id]
            bot_id = trade.get('bot_id', 'N/A')
            decision_source = trade.get('decision_source', 'NONE')
            is_simulated = trade.get('is_simulated', 'NONE')
            symbol = trade.get('symbol', 'N/A')
            status = trade.get('status', 'N/A')
            pnl = trade.get('pnl', 'N/A')
            timestamp = trade.get('timestamp', 'N/A')
            
            print(f"\nID: {trade_id}")
            print(f"  bot_id: {bot_id}")
            print(f"  symbol: {symbol}")
            print(f"  status: {status}")
            print(f"  decision_source: {decision_source}")
            print(f"  is_simulated: {is_simulated}")
            print(f"  pnl: {pnl}")
            print(f"  timestamp: {timestamp}")
            
            # Проверяем, проблемная ли это запись
            if decision_source == 'AI' and is_simulated != True:
                print(f"  ⚠️ ПРОБЛЕМА: decision_source=AI но is_simulated={is_simulated}!")
            if bot_id and len(str(bot_id)) > 10 and '_' in str(bot_id) and is_simulated != True:
                print(f"  ⚠️ ПРОБЛЕМА: длинный bot_id ({bot_id}) но is_simulated={is_simulated}!")
    
    # Статистика по новым записям
    if new_history_ids or new_trade_ids:
        print(f"\n" + "="*70)
        print("СТАТИСТИКА НОВЫХ ЗАПИСЕЙ:")
        print("="*70)
        
        new_history_entries = [history2[eid] for eid in new_history_ids]
        new_trade_entries = [trades2[tid] for tid in new_trade_ids]
        
        # По decision_source
        history_sources = {}
        for e in new_history_entries:
            src = e.get('decision_source', 'NONE')
            history_sources[src] = history_sources.get(src, 0) + 1
        
        trade_sources = {}
        for t in new_trade_entries:
            src = t.get('decision_source', 'NONE')
            trade_sources[src] = trade_sources.get(src, 0) + 1
        
        print(f"\nИстория по decision_source:")
        for src, count in sorted(history_sources.items()):
            print(f"  {src}: {count}")
        
        print(f"\nСделки по decision_source:")
        for src, count in sorted(trade_sources.items()):
            print(f"  {src}: {count}")
        
        # По is_simulated
        history_sim = {}
        for e in new_history_entries:
            sim = str(e.get('is_simulated', 'NONE'))
            history_sim[sim] = history_sim.get(sim, 0) + 1
        
        trade_sim = {}
        for t in new_trade_entries:
            sim = str(t.get('is_simulated', 'NONE'))
            trade_sim[sim] = trade_sim.get(sim, 0) + 1
        
        print(f"\nИстория по is_simulated:")
        for sim, count in sorted(history_sim.items()):
            print(f"  is_simulated={sim}: {count}")
        
        print(f"\nСделки по is_simulated:")
        for sim, count in sorted(trade_sim.items()):
            print(f"  is_simulated={sim}: {count}")
        
        # По bot_id (топ 10)
        history_bot_ids = {}
        for e in new_history_entries:
            bid = e.get('bot_id', 'NONE')
            history_bot_ids[bid] = history_bot_ids.get(bid, 0) + 1
        
        trade_bot_ids = {}
        for t in new_trade_entries:
            bid = t.get('bot_id', 'NONE')
            trade_bot_ids[bid] = trade_bot_ids.get(bid, 0) + 1
        
        print(f"\nИстория по bot_id (топ 10):")
        for bid, count in sorted(history_bot_ids.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {bid}: {count}")
        
        print(f"\nСделки по bot_id (топ 10):")
        for bid, count in sorted(trade_bot_ids.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {bid}: {count}")
    
    print(f"\n" + "="*70)
    if new_history_ids or new_trade_ids:
        problematic = [e for e in new_history_entries 
                      if e.get('decision_source') == 'AI' and e.get('is_simulated') != True]
        problematic += [t for t in new_trade_entries 
                       if t.get('decision_source') == 'AI' and t.get('is_simulated') != True]
        
        if problematic:
            print(f"❌ ОБНАРУЖЕНЫ ПРОБЛЕМНЫЕ ЗАПИСИ: {len(problematic)}")
            print("   Записи с decision_source=AI но is_simulated!=True")
        else:
            print("✅ Все новые записи выглядят корректно!")
    else:
        print("Файлы идентичны - новых записей нет")
    print("="*70)

def main():
    args = parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.is_absolute() and not str(file1).startswith('\\\\'):
        root_dir = Path(__file__).parent.parent
        file1 = root_dir / file1
    
    if not file2.is_absolute() and not str(file2).startswith('\\\\'):
        root_dir = Path(__file__).parent.parent
        file2 = root_dir / file2
    
    compare_files(file1, file2)

if __name__ == '__main__':
    main()

