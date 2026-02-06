#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ истории торговли ботов"""

import json
from collections import Counter

# Загружаем историю
with open('data/bot_history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

history = data.get('history', [])
trades = data.get('trades', [])

import sys
import io

# Настройка вывода для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("АНАЛИЗ ИСТОРИИ ТОРГОВЛИ БОТОВ")
print("=" * 60)

print(f"\nОбщая статистика:")
print(f"  • Всего действий: {len(history)}")
print(f"  • Всего сделок: {len(trades)}")
print(f"  • Открытых позиций: {sum(1 for t in trades if t.get('status') == 'OPEN')}")
print(f"  • Закрытых позиций: {sum(1 for t in trades if t.get('status') == 'CLOSED')}")

# Анализ закрытых сделок
closed_trades = [t for t in trades if t.get('status') == 'CLOSED' and t.get('pnl') is not None]
if closed_trades:
    profitable = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losing = [t for t in closed_trades if t.get('pnl', 0) < 0]
    
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
    win_rate = (len(profitable) / len(closed_trades) * 100) if closed_trades else 0
    
    best_trade = max(closed_trades, key=lambda x: x.get('pnl', 0))
    worst_trade = min(closed_trades, key=lambda x: x.get('pnl', 0))
    
    print(f"\nСтатистика закрытых сделок:")
    print(f"  • Прибыльных: {len(profitable)}")
    print(f"  • Убыточных: {len(losing)}")
    print(f"  • Win Rate: {win_rate:.1f}%")
    print(f"  • Общий PnL: {total_pnl:.2f} USDT")
    print(f"  • Средний PnL: {avg_pnl:.2f} USDT")
    
    if best_trade:
        print(f"\nЛучшая сделка:")
        print(f"  • Символ: {best_trade.get('symbol')}")
        print(f"  • Направление: {best_trade.get('direction')}")
        print(f"  • PnL: {best_trade.get('pnl', 0):.2f} USDT")
        print(f"  • ROI: {best_trade.get('roi', 0):.2f}%")
    
    if worst_trade:
        print(f"\nХудшая сделка:")
        print(f"  • Символ: {worst_trade.get('symbol')}")
        print(f"  • Направление: {worst_trade.get('direction')}")
        print(f"  • PnL: {worst_trade.get('pnl', 0):.2f} USDT")
        print(f"  • ROI: {worst_trade.get('roi', 0):.2f}%")

# Типы действий
action_types = Counter(h.get('action_type') for h in history)
print(f"\nТипы действий:")
for action_type, count in action_types.most_common():
    print(f"  - {action_type}: {count}")

# Символы
symbols = Counter(h.get('symbol') for h in history if h.get('symbol'))
print(f"\nТоп-10 символов по активности:")
for symbol, count in symbols.most_common(10):
    print(f"  • {symbol}: {count} действий")

print("\n" + "=" * 60)
print("Ограничения на количество записей ОТКЛЮЧЕНЫ")
print("Все данные сохраняются для обучения ботов")
print("=" * 60)

