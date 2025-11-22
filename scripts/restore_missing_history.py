#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для восстановления отсутствующих записей истории из bots_state.json
Добавляет записи об открытии позиций для ботов, которые есть в bots_state.json,
но отсутствуют в bot_history.json
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from bot_engine.bots_database import get_bots_database

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

def restore_missing_history():
    """Восстанавливает отсутствующие записи истории"""
    
    bots_state_file = root_dir / 'data' / 'bots_state.json'
    history_file = root_dir / 'data' / 'bot_history.json'
    
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Инициализируем БД ботов
    try:
        bots_db = get_bots_database()
    except Exception as e:
        print(f"⚠️ Не удалось подключиться к Bots Database: {e}")
        bots_db = None
    
    print("=" * 60)
    print("ВОССТАНОВЛЕНИЕ ОТСУТСТВУЮЩЕЙ ИСТОРИИ")
    print("=" * 60)
    
    # Загружаем данные
    bots_state = load_json_file(bots_state_file)
    if not bots_state:
        print("ОШИБКА: Не удалось загрузить bots_state.json")
        return False
    
    history_data = load_json_file(history_file)
    if not history_data:
        print("Создаем новый файл истории")
        history_data = {
            'history': [],
            'trades': [],
            'last_update': datetime.now().isoformat()
        }
    
    # Получаем список ботов с открытыми позициями
    bots = bots_state.get('bots', {})
    open_positions = {}
    
    for symbol, bot_data in bots.items():
        status = bot_data.get('status', '')
        if 'in_position' in status.lower():
            entry_price = bot_data.get('entry_price')
            position_side = bot_data.get('position_side', 'LONG')
            position_size = bot_data.get('position_size') or bot_data.get('position_size_coins', 0)
            position_start_time = bot_data.get('position_start_time')
            
            if entry_price and position_size and position_start_time:
                open_positions[symbol] = {
                    'symbol': symbol,
                    'direction': position_side,
                    'entry_price': float(entry_price),
                    'size': float(position_size),
                    'position_start_time': position_start_time,
                    'stop_loss': bot_data.get('stop_loss'),
                    'take_profit': bot_data.get('take_profit')
                }
    
    print(f"\nНайдено ботов с открытыми позициями: {len(open_positions)}")
    
    # Проверяем, какие позиции уже есть в истории
    existing_trades = {trade.get('bot_id'): trade for trade in history_data.get('trades', []) 
                      if trade.get('status') == 'OPEN'}
    existing_history = {entry.get('bot_id'): entry for entry in history_data.get('history', [])
                       if entry.get('action_type') == 'POSITION_OPENED'}
    
    print(f"Уже есть в истории: {len(existing_trades)} открытых позиций")
    
    # Восстанавливаем отсутствующие записи
    restored_count = 0
    history_entries = history_data.get('history', [])
    trades = history_data.get('trades', [])
    
    for symbol, pos_data in open_positions.items():
        # Проверяем, есть ли уже запись
        if symbol in existing_trades or symbol in existing_history:
            continue
        
        # Создаем запись истории
        timestamp = pos_data['position_start_time']
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        else:
            dt = datetime.now()
        
        entry_id = f"open_{symbol}_{dt.timestamp()}"
        
        history_entry = {
            'id': entry_id,
            'timestamp': dt.isoformat(),
            'action_type': 'POSITION_OPENED',
            'action_name': 'Открытие позиции',
            'bot_id': symbol,
            'symbol': symbol,
            'direction': pos_data['direction'],
            'size': pos_data['size'],
            'entry_price': pos_data['entry_price'],
            'stop_loss': float(pos_data['stop_loss']) if pos_data.get('stop_loss') else None,
            'take_profit': float(pos_data['take_profit']) if pos_data.get('take_profit') else None,
            'decision_source': 'SCRIPT',
            'ai_decision_id': None,
            'ai_confidence': None,
            'ai_signal': None,
            'rsi': None,
            'trend': None,
            'is_simulated': False,  # КРИТИЧНО: восстановленные позиции - это реальные сделки!
            'details': f"Открыта позиция {pos_data['direction']} для {symbol}: размер {pos_data['size']}, цена входа {pos_data['entry_price']:.4f} [RESTORED]"
        }
        
        trade_entry = {
            'id': f"trade_{symbol}_{dt.timestamp()}",
            'timestamp': dt.isoformat(),
            'bot_id': symbol,
            'symbol': symbol,
            'direction': pos_data['direction'],
            'size': pos_data['size'],
            'entry_price': pos_data['entry_price'],
            'exit_price': None,
            'pnl': None,
            'status': 'OPEN',
            'decision_source': 'SCRIPT',
            'ai_decision_id': None,
            'ai_confidence': None,
            'rsi': None,
            'trend': None,
            'is_simulated': False  # КРИТИЧНО: восстановленные позиции - это реальные сделки!
        }
        
        history_entries.append(history_entry)
        trades.append(trade_entry)
        restored_count += 1
        
        # КРИТИЧНО: Также сохраняем в bots_data.db для истории торговли ботов
        if bots_db:
            try:
                # Конвертируем timestamp в нужный формат
            entry_timestamp = None
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    entry_timestamp = dt.timestamp() * 1000
                except:
                    pass
            elif isinstance(timestamp, (int, float)):
                entry_timestamp = timestamp * 1000 if timestamp < 1e10 else timestamp
            
            trade_data = {
                'bot_id': symbol,
                'symbol': symbol,
                'direction': pos_data['direction'],
                'entry_price': pos_data['entry_price'],
                'exit_price': None,  # Позиция еще открыта
                'entry_time': timestamp if isinstance(timestamp, str) else dt.isoformat() if 'dt' in locals() else datetime.now().isoformat(),
                'exit_time': None,
                'entry_timestamp': entry_timestamp,
                'exit_timestamp': None,
                'position_size_usdt': None,  # TODO: получить если есть
                'position_size_coins': pos_data['size'],
                'pnl': None,
                'roi': None,
                'status': 'OPEN',
                'close_reason': None,
                'decision_source': 'SCRIPT',
                'ai_decision_id': None,
                'ai_confidence': None,
                'entry_rsi': None,
                'exit_rsi': None,
                'entry_trend': None,
                'exit_trend': None,
                'entry_volatility': None,
                'entry_volume_ratio': None,
                'is_successful': None,
                'is_simulated': False,
                'source': 'script_restore',
                'order_id': None,
                'extra_data': {
                    'stop_loss': pos_data.get('stop_loss'),
                    'take_profit': pos_data.get('take_profit'),
                    'restored': True
                }
            }
            
                trade_id = bots_db.save_bot_trade_history(trade_data)
                if trade_id:
                    print(f"  ✅ История для {symbol} сохранена в bots_data.db (ID: {trade_id})")
            except Exception as bots_db_error:
                print(f"  ⚠️ Ошибка сохранения истории для {symbol} в bots_data.db: {bots_db_error}")
        
        print(f"  Восстановлена история для {symbol}")
    
    if restored_count > 0:
        # Обновляем данные
        history_data['history'] = history_entries
        history_data['trades'] = trades
        history_data['last_update'] = datetime.now().isoformat()
        
        # Сохраняем
        if save_json_file(history_file, history_data):
            print(f"\nВосстановлено {restored_count} записей истории")
            print(f"Файл сохранен: {history_file}")
            return True
        else:
            print(f"\nОШИБКА: Не удалось сохранить файл истории")
            return False
    else:
        print("\nВсе позиции уже есть в истории, восстановление не требуется")
        return True

if __name__ == '__main__':
    try:
        restore_missing_history()
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

