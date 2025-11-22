#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обновления истории: добавляет RSI и тренд в записи, где они отсутствуют
Использует данные из bots_state.json и coins_rsi_data для восстановления
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

def load_json_file(filepath):
    """Загружает JSON файл"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"ОШИБКА загрузки {filepath}: {e}")
        return None

def save_json_file(filepath, data):
    """Сохраняет JSON файл"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"ОШИБКА сохранения {filepath}: {e}")
        return False

def update_history_with_rsi():
    """Обновляет историю, добавляя RSI и тренд"""
    
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    history_file = root_dir / 'data' / 'bot_history.json'
    bots_state_file = root_dir / 'data' / 'bots_state.json'
    
    print("=" * 60)
    print("ОБНОВЛЕНИЕ ИСТОРИИ: ДОБАВЛЕНИЕ RSI И ТРЕНДА")
    print("=" * 60)
    
    # Загружаем историю
    history_data = load_json_file(history_file)
    if not history_data:
        print("ОШИБКА: Не удалось загрузить bot_history.json")
        return False
    
    # Загружаем состояние ботов
    bots_state = load_json_file(bots_state_file)
    bots_rsi_data = {}
    if bots_state:
        for symbol, bot_data in bots_state.get('bots', {}).items():
            rsi_info = bot_data.get('rsi_data', {})
            if rsi_info:
                bots_rsi_data[symbol] = {
                    'rsi': rsi_info.get('rsi6h'),
                    'trend': rsi_info.get('trend6h')
                }
    
    print(f"\nЗагружено RSI данных для {len(bots_rsi_data)} символов")
    
    # Обновляем записи истории
    history_entries = history_data.get('history', [])
    trades = history_data.get('trades', [])
    
    updated_history = 0
    updated_trades = 0
    
    # Обновляем записи открытия позиций
    for entry in history_entries:
        if entry.get('action_type') == 'POSITION_OPENED':
            symbol = entry.get('symbol')
            # КРИТИЧНО: Добавляем is_simulated если отсутствует
            if 'is_simulated' not in entry:
                decision_source = entry.get('decision_source', '')
                # EXCHANGE_IMPORT и SCRIPT - это реальные сделки
                # AI может быть как реальным (боты из bots.py), так и симуляцией (ai.py)
                if decision_source in ('EXCHANGE_IMPORT', 'SCRIPT'):
                    entry['is_simulated'] = False
                    updated_history += 1
                elif decision_source == 'AI':
                    # Для AI проверяем признаки симуляции по bot_id
                    bot_id = entry.get('bot_id', '')
                    # Реальные боты имеют короткий bot_id (символ монеты)
                    # AI симуляции имеют длинный bot_id с маркерами
                    if bot_id and len(bot_id) > 10 and ('_' in bot_id or 'SIMULATION' in bot_id.upper() or 'BACKTEST' in bot_id.upper()):
                        entry['is_simulated'] = True
                        updated_history += 1
                    else:
                        entry['is_simulated'] = False
                        updated_history += 1
            
            if symbol and (entry.get('rsi') is None or entry.get('trend') is None):
                # Пытаемся получить из bots_state
                rsi_data = bots_rsi_data.get(symbol, {})
                if entry.get('rsi') is None and rsi_data.get('rsi'):
                    entry['rsi'] = rsi_data['rsi']
                    updated_history += 1
                if entry.get('trend') is None and rsi_data.get('trend'):
                    entry['trend'] = rsi_data['trend']
                    updated_history += 1
    
    # Обновляем сделки
    for trade in trades:
        symbol = trade.get('symbol')
        # КРИТИЧНО: Добавляем is_simulated если отсутствует
        if 'is_simulated' not in trade:
            decision_source = trade.get('decision_source', '')
            # EXCHANGE_IMPORT и SCRIPT - это реальные сделки
            # AI может быть как реальным (боты из bots.py), так и симуляцией (ai.py)
            if decision_source in ('EXCHANGE_IMPORT', 'SCRIPT'):
                trade['is_simulated'] = False
                updated_trades += 1
            elif decision_source == 'AI':
                # Для AI проверяем признаки симуляции по bot_id
                bot_id = trade.get('bot_id', '')
                # Реальные боты имеют короткий bot_id (символ монеты)
                # AI симуляции имеют длинный bot_id с маркерами
                if bot_id and len(bot_id) > 10 and ('_' in bot_id or 'SIMULATION' in bot_id.upper() or 'BACKTEST' in bot_id.upper()):
                    trade['is_simulated'] = True
                    updated_trades += 1
                else:
                    trade['is_simulated'] = False
                    updated_trades += 1
        
        if symbol and (trade.get('rsi') is None or trade.get('trend') is None):
            rsi_data = bots_rsi_data.get(symbol, {})
            if trade.get('rsi') is None and rsi_data.get('rsi'):
                trade['rsi'] = rsi_data['rsi']
                updated_trades += 1
            if trade.get('trend') is None and rsi_data.get('trend'):
                trade['trend'] = rsi_data['trend']
                updated_trades += 1
    
    if updated_history > 0 or updated_trades > 0:
        history_data['history'] = history_entries
        history_data['trades'] = trades
        history_data['last_update'] = datetime.now().isoformat()
        
        if save_json_file(history_file, history_data):
            print(f"\nОбновлено:")
            print(f"  - Записей истории: {updated_history}")
            print(f"  - Сделок: {updated_trades}")
            print(f"Файл сохранен: {history_file}")
            return True
        else:
            print(f"\nОШИБКА: Не удалось сохранить файл истории")
            return False
    else:
        print("\nВсе записи уже содержат RSI и тренд, обновление не требуется")
        return True

if __name__ == '__main__':
    try:
        update_history_with_rsi()
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

