#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки AI симуляций
Проверяет наличие файла simulated_trades.json и его содержимое
"""

import json
import os
import sys
from pathlib import Path

def main():
    simulated_file = Path('data/ai/simulated_trades.json')
    
    print("="*70)
    print("ПРОВЕРКА AI СИМУЛЯЦИЙ")
    print("="*70)
    
    if not simulated_file.exists():
        print(f"\nФайл не найден: {simulated_file}")
        print("\nВОЗМОЖНЫЕ ПРИЧИНЫ:")
        print("1. AI обучение на исторических данных еще не запускалось")
        print("2. train_on_historical_data() не вызывается в ai.py")
        print("3. Симуляции не генерируются (нет сделок в симуляции)")
        print("\nРЕШЕНИЕ:")
        print("- Запустите ai.py и дождитесь завершения обучения на исторических данных")
        print("- Проверьте логи ai.py на наличие сообщений о сохранении симуляций")
        print("- Убедитесь, что train_on_historical_data() вызывается")
        return 1
    
    try:
        with open(simulated_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_trades = data.get('total_trades', 0)
        trades = data.get('trades', [])
        last_update = data.get('last_update', 'N/A')
        
        print(f"\nФайл найден: {simulated_file}")
        print(f"Всего симулированных сделок: {total_trades}")
        print(f"Последнее обновление: {last_update}")
        
        if total_trades == 0:
            print("\nВНИМАНИЕ: Файл существует, но пуст!")
            print("Это означает, что симуляции запускались, но не генерировали сделок")
            return 1
        
        # Анализ симуляций
        successful = sum(1 for t in trades if t.get('is_successful', False))
        failed = total_trades - successful
        win_rate = (successful / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\nСТАТИСТИКА СИМУЛЯЦИЙ:")
        print(f"  Успешных сделок: {successful} ({win_rate:.1f}%)")
        print(f"  Неуспешных сделок: {failed} ({100-win_rate:.1f}%)")
        
        # Проверяем наличие необходимых полей
        required_fields = ['symbol', 'entry_price', 'exit_price', 'pnl', 'is_simulated']
        missing_fields = []
        for trade in trades[:10]:  # Проверяем первые 10
            for field in required_fields:
                if field not in trade:
                    if field not in missing_fields:
                        missing_fields.append(field)
        
        if missing_fields:
            print(f"\nВНИМАНИЕ: Отсутствуют поля: {', '.join(missing_fields)}")
        else:
            print("\nВсе необходимые поля присутствуют")
        
        print("\nПРИМЕР СИМУЛЯЦИИ (первая сделка):")
        if trades:
            sample = trades[0]
            print(f"  Symbol: {sample.get('symbol')}")
            print(f"  Direction: {sample.get('direction')}")
            print(f"  Entry Price: {sample.get('entry_price')}")
            print(f"  Exit Price: {sample.get('exit_price')}")
            print(f"  PnL: {sample.get('pnl')}")
            print(f"  Is Successful: {sample.get('is_successful')}")
            print(f"  Is Simulated: {sample.get('is_simulated')}")
        
        print("\nВсе проверки пройдены!")
        return 0
        
    except Exception as e:
        print(f"\nОшибка чтения файла: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

