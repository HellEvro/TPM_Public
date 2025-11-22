#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Детальный анализ bot_history.json"""

import json
import sys
import io
import argparse
from pathlib import Path
from collections import Counter

# Исправляем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description='Детальный анализ bot_history.json')
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='Путь к файлу (по умолчанию: data/bot_history.json)')
    return parser.parse_args()

def analyze_file(file_path):
    """Анализирует файл bot_history.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    history = data.get('history', [])
    trades = data.get('trades', [])
    
    print("="*70)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ bot_history.json")
    print("="*70)
    print(f"\nФайл: {file_path}")
    print(f"Всего записей истории: {len(history)}")
    print(f"Всего сделок: {len(trades)}")
    
    # Анализ по decision_source
    print("\n" + "="*70)
    print("АНАЛИЗ ПО decision_source:")
    print("="*70)
    
    history_sources = Counter([e.get('decision_source', 'NONE') for e in history])
    trades_sources = Counter([t.get('decision_source', 'NONE') for t in trades])
    
    print("\nИстория (history):")
    for source, count in history_sources.most_common():
        print(f"  {source}: {count}")
    
    print("\nСделки (trades):")
    for source, count in trades_sources.most_common():
        print(f"  {source}: {count}")
    
    # Анализ по is_simulated
    print("\n" + "="*70)
    print("АНАЛИЗ ПО is_simulated:")
    print("="*70)
    
    history_sim = Counter([str(e.get('is_simulated', 'NONE')) for e in history])
    trades_sim = Counter([str(t.get('is_simulated', 'NONE')) for t in trades])
    
    print("\nИстория (history):")
    for sim, count in history_sim.most_common():
        print(f"  is_simulated={sim}: {count}")
    
    print("\nСделки (trades):")
    for sim, count in trades_sim.most_common():
        print(f"  is_simulated={sim}: {count}")
    
    # Анализ по bot_id (первые 20 уникальных)
    print("\n" + "="*70)
    print("АНАЛИЗ ПО bot_id (топ 20):")
    print("="*70)
    
    history_bot_ids = Counter([e.get('bot_id', 'NONE') for e in history])
    trades_bot_ids = Counter([t.get('bot_id', 'NONE') for t in trades])
    
    print("\nИстория (history) - топ 20:")
    for bot_id, count in history_bot_ids.most_common(20):
        # Проверяем, есть ли записи с этим bot_id и decision_source=AI
        ai_count = len([e for e in history if e.get('bot_id') == bot_id and e.get('decision_source') == 'AI'])
        sim_count = len([e for e in history if e.get('bot_id') == bot_id and e.get('is_simulated') == True])
        print(f"  {bot_id}: {count} (AI: {ai_count}, simulated: {sim_count})")
    
    print("\nСделки (trades) - топ 20:")
    for bot_id, count in trades_bot_ids.most_common(20):
        # Проверяем, есть ли записи с этим bot_id и decision_source=AI
        ai_count = len([t for t in trades if t.get('bot_id') == bot_id and t.get('decision_source') == 'AI'])
        sim_count = len([t for t in trades if t.get('bot_id') == bot_id and t.get('is_simulated') == True])
        print(f"  {bot_id}: {count} (AI: {ai_count}, simulated: {sim_count})")
    
    # Проверка проблемных записей
    print("\n" + "="*70)
    print("ПРОВЕРКА ПРОБЛЕМНЫХ ЗАПИСЕЙ:")
    print("="*70)
    
    # Записи с decision_source=AI но без is_simulated или is_simulated=False
    problematic_history = [e for e in history 
                         if e.get('decision_source') == 'AI' 
                         and (e.get('is_simulated') == False or 'is_simulated' not in e)]
    
    problematic_trades = [t for t in trades 
                        if t.get('decision_source') == 'AI' 
                        and (t.get('is_simulated') == False or 'is_simulated' not in t)]
    
    print(f"\nИстория с decision_source=AI и is_simulated=False/None: {len(problematic_history)}")
    if problematic_history:
        print("  Примеры:")
        for e in problematic_history[:5]:
            print(f"    bot_id={e.get('bot_id')}, is_simulated={e.get('is_simulated')}")
    
    print(f"\nСделки с decision_source=AI и is_simulated=False/None: {len(problematic_trades)}")
    if problematic_trades:
        print("  Примеры:")
        for t in problematic_trades[:5]:
            print(f"    bot_id={t.get('bot_id')}, is_simulated={t.get('is_simulated')}")
    
    # Записи с длинным bot_id (потенциально AI симуляции)
    long_bot_ids_history = [e for e in history 
                           if e.get('bot_id') and len(str(e.get('bot_id'))) > 10 
                           and e.get('is_simulated') != True]
    
    long_bot_ids_trades = [t for t in trades 
                          if t.get('bot_id') and len(str(t.get('bot_id'))) > 10 
                          and t.get('is_simulated') != True]
    
    print(f"\nИстория с длинным bot_id (>10 символов) и is_simulated!=True: {len(long_bot_ids_history)}")
    if long_bot_ids_history:
        print("  Примеры:")
        for e in long_bot_ids_history[:5]:
            print(f"    bot_id={e.get('bot_id')}, is_simulated={e.get('is_simulated')}, source={e.get('decision_source')}")
    
    print(f"\nСделки с длинным bot_id (>10 символов) и is_simulated!=True: {len(long_bot_ids_trades)}")
    if long_bot_ids_trades:
        print("  Примеры:")
        for t in long_bot_ids_trades[:5]:
            print(f"    bot_id={t.get('bot_id')}, is_simulated={t.get('is_simulated')}, source={t.get('decision_source')}")
    
    print("\n" + "="*70)
    if problematic_history or problematic_trades or long_bot_ids_history or long_bot_ids_trades:
        print("ВНИМАНИЕ: Обнаружены потенциально проблемные записи!")
    else:
        print("Все записи выглядят корректно!")
    print("="*70)

def main():
    args = parse_args()
    
    # Определяем путь к файлу
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute() and not str(file_path).startswith('\\\\'):
            root_dir = Path(__file__).parent.parent
            file_path = root_dir / file_path
    else:
        root_dir = Path(__file__).parent.parent
        file_path = root_dir / 'data' / 'bot_history.json'
    
    analyze_file(file_path)

if __name__ == '__main__':
    main()

