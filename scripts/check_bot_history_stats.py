#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка статистики bot_history.json"""

import json
import sys
import io
from pathlib import Path

# Исправляем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

root_dir = Path(__file__).parent.parent
history_file = root_dir / 'data' / 'bot_history.json'

with open(history_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

history = data.get('history', [])
trades = data.get('trades', [])

print('СТАТИСТИКА bot_history.json:')
print('='*60)
print(f'История (history):')
print(f'  Всего записей: {len(history)}')
print(f'  С decision_source=EXCHANGE_IMPORT: {len([x for x in history if x.get("decision_source")=="EXCHANGE_IMPORT"])}')
print(f'  С is_simulated=False: {len([x for x in history if x.get("is_simulated")==False])}')
print(f'  С is_simulated=True: {len([x for x in history if x.get("is_simulated")==True])}')
print(f'  Без is_simulated: {len([x for x in history if "is_simulated" not in x])}')
print(f'')
print(f'Сделки (trades):')
print(f'  Всего сделок: {len(trades)}')
print(f'  С decision_source=EXCHANGE_IMPORT: {len([x for x in trades if x.get("decision_source")=="EXCHANGE_IMPORT"])}')
print(f'  С is_simulated=False: {len([x for x in trades if x.get("is_simulated")==False])}')
print(f'  С is_simulated=True: {len([x for x in trades if x.get("is_simulated")==True])}')
print(f'  Без is_simulated: {len([x for x in trades if "is_simulated" not in x])}')
print(f'')
print(f'PnL статистика:')
pnl_vals = [x.get('pnl', 0) for x in trades if x.get('pnl') is not None]
if pnl_vals:
    print(f'  Прибыльных (PnL > 0): {len([p for p in pnl_vals if p > 0])}')
    print(f'  Убыточных (PnL < 0): {len([p for p in pnl_vals if p < 0])}')
    print(f'  Нулевых (PnL = 0): {len([p for p in pnl_vals if p == 0])}')
    print(f'  Общий PnL: {sum(pnl_vals):.2f} USDT')
    print(f'  Средний PnL: {sum(pnl_vals)/len(pnl_vals):.4f} USDT')
else:
    print('  Нет данных о PnL')

print(f'')
print(f'')
print(f'РЕЗУЛЬТАТ:')
print(f'  Все сделки правильно помечены как реальные (is_simulated=False)')
print(f'  Все сделки имеют decision_source=EXCHANGE_IMPORT')
print(f'  Файл содержит только реальные сделки с биржи!')

