#!/usr/bin/env python3
"""Проверка данных ATOM сделок в БД"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_engine.bots_database import get_bots_database
from app.app_database import get_app_database
from datetime import datetime

# Проверяем bot_trades_history
print("=" * 80)
print("ПРОВЕРКА ATOM В БД")
print("=" * 80)

db = get_bots_database()
trades = db.get_bot_trades_history(symbol='ATOM', status='CLOSED', limit=10)

print(f"\n=== bot_trades_history ===")
print(f"Найдено {len(trades)} закрытых сделок ATOM в bot_trades_history:")
for i, t in enumerate(trades[:5]):
    pnl = t.get('pnl')
    exit_time = t.get('exit_time')
    exit_timestamp = t.get('exit_timestamp')
    close_reason = t.get('close_reason')
    print(f"  #{i+1}: pnl={pnl} (type={type(pnl).__name__}), exit_time={exit_time}, exit_timestamp={exit_timestamp}, reason={close_reason}")

# Проверяем closed_pnl_history
print(f"\n=== closed_pnl_history ===")
try:
    app_db = get_app_database()
    all_closed_pnl = app_db.load_closed_pnl_history(sort_by='time', period='all')
    atom_trades = [t for t in all_closed_pnl if t.get('symbol') == 'ATOM']
    print(f"Найдено {len(atom_trades)} закрытых сделок ATOM в closed_pnl_history:")
    for i, t in enumerate(atom_trades[:5]):
        pnl = t.get('closed_pnl')
        close_time = t.get('close_time')
        close_timestamp = t.get('close_timestamp')
        print(f"  #{i+1}: closed_pnl={pnl} (type={type(pnl).__name__}), close_time={close_time}, close_timestamp={close_timestamp}")
except Exception as e:
    print(f"Ошибка чтения closed_pnl_history: {e}")

# Проверяем last_close_timestamps
print(f"\n=== last_close_timestamps ===")
try:
    from bots_modules.imports_and_globals import bots_data, bots_data_lock
    with bots_data_lock:
        last_close_timestamps = bots_data.get('last_close_timestamps', {})
        atom_timestamp = last_close_timestamps.get('ATOM')
        if atom_timestamp:
            current_timestamp = datetime.now().timestamp()
            time_since_close = current_timestamp - float(atom_timestamp)
            hours = time_since_close / 3600
            print(f"ATOM last_close_timestamp: {atom_timestamp}")
            print(f"Прошло времени: {time_since_close:.0f} секунд ({hours:.2f} часов)")
            if time_since_close < 3600:
                print(f"⚠️ БЛОКИРОВКА АКТИВНА: осталось {3600 - time_since_close:.0f} секунд ({((3600 - time_since_close) / 60):.1f} минут)")
            else:
                print(f"✅ Блокировка снята (прошло больше 1 часа)")
        else:
            print("ATOM: timestamp последнего закрытия не найден")
except Exception as e:
    print(f"Ошибка проверки last_close_timestamps: {e}")

# Проверяем активных ботов ATOM
print(f"\n=== Активные боты ATOM ===")
try:
    from bots_modules.imports_and_globals import bots_data, bots_data_lock
    with bots_data_lock:
        atom_bot = bots_data['bots'].get('ATOM')
        if atom_bot:
            status = atom_bot.get('status')
            last_close_ts = atom_bot.get('last_position_close_timestamp')
            print(f"ATOM бот найден: status={status}, last_position_close_timestamp={last_close_ts}")
            if last_close_ts:
                current_timestamp = datetime.now().timestamp()
                time_since_close = current_timestamp - float(last_close_ts)
                hours = time_since_close / 3600
                print(f"  Прошло времени: {time_since_close:.0f} секунд ({hours:.2f} часов)")
        else:
            print("ATOM бот не найден в bots_data")
except Exception as e:
    print(f"Ошибка проверки бота: {e}")

print("\n" + "=" * 80)
