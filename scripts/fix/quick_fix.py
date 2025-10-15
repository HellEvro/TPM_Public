#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Быстрое исправление всех ошибок отступов"""

# Список известных ошибок и их исправлений
fixes = [
    # (line_number, old_indent, new_indent)
    (1605, 16, 20),  # for future in concurrent...
    (1606, 20, 24),  # try:
    (1607, 28, 28),  # result = ...
    (1608, 24, 28),  # if result:
    (1609, 32, 32),  # batch_coins_data...
    (1615, 28, 32),  # with rsi_data_lock:
    (1616, 32, 36),  # coins_rsi_data['successful_coins'] += 1
    (1617, 24, 28),  # else:
    (1618, 28, 32),  # with rsi_data_lock:
    (1619, 32, 36),  # coins_rsi_data['failed_coins'] += 1
    (1620, 20, 24),  # except concurrent.futures.TimeoutError:
    (1621, 24, 28),  # symbol = future_to_symbol[future]
    (1622, 28, 28),  # # logger.warning...
    (1623, 24, 28),  # with rsi_data_lock:
    (1624, 28, 32),  # coins_rsi_data['failed_coins'] += 1
    (1625, 20, 24),  # except Exception as e:
    (1626, 24, 28),  # symbol = future_to_symbol[future]
    (1627, 28, 28),  # # logger.warning...
    (1628, 24, 28),  # with rsi_data_lock:
    (1629, 28, 32),  # coins_rsi_data['failed_coins'] += 1
]

with open('bots.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Применяем исправления
for line_num, old_indent, new_indent in fixes:
    idx = line_num - 1  # 0-based index
    if idx < len(lines):
        line = lines[idx]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        if current_indent == old_indent:
            lines[idx] = ' ' * new_indent + stripped

with open('bots.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed!")

