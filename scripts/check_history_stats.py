#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка статистики истории ботов"""

import json
import sys
from pathlib import Path
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

root_dir = Path(__file__).parent.parent
history_file = root_dir / 'data' / 'bot_history.json'

data = json.load(open(history_file, 'r', encoding='utf-8'))
history = data.get('history', [])
trades = data.get('trades', [])

print("=" * 60)
print("СТАТИСТИКА ИСТОРИИ БОТОВ")
print("=" * 60)
print(f"\nВсего записей истории: {len(history)}")
print(f"Всего сделок: {len(trades)}")
print(f"\nПо типам действий:")
print(f"  - BOT_START: {sum(1 for h in history if h.get('action_type') == 'BOT_START')}")
print(f"  - POSITION_OPENED: {sum(1 for h in history if h.get('action_type') == 'POSITION_OPENED')}")
print(f"  - POSITION_CLOSED: {sum(1 for h in history if h.get('action_type') == 'POSITION_CLOSED')}")
print(f"  - BOT_STOP: {sum(1 for h in history if h.get('action_type') == 'BOT_STOP')}")

print(f"\nДанные для обучения AI:")
print(f"  - Записей с RSI: {sum(1 for h in history if h.get('rsi') is not None)}")
print(f"  - Записей с трендом: {sum(1 for h in history if h.get('trend') is not None)}")
print(f"  - Записей БЕЗ RSI: {sum(1 for h in history if h.get('rsi') is None and h.get('action_type') == 'POSITION_OPENED')}")
print(f"  - Записей БЕЗ тренда: {sum(1 for h in history if h.get('trend') is None and h.get('action_type') == 'POSITION_OPENED')}")

print(f"\nСделки:")
print(f"  - Открытых позиций: {sum(1 for t in trades if t.get('status') == 'OPEN')}")
print(f"  - Закрытых позиций: {sum(1 for t in trades if t.get('status') == 'CLOSED')}")

print("\n" + "=" * 60)

