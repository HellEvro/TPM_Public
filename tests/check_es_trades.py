#!/usr/bin/env python3
"""Проверка данных ES сделок в БД"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_engine.bots_database import get_bots_database
from app.app_database import get_app_database

# Проверяем bot_trades_history
db = get_bots_database()
trades = db.get_bot_trades_history(symbol='ES', status='CLOSED', limit=5)

print(f"=== bot_trades_history ===")
print(f"Найдено {len(trades)} закрытых сделок ES в bot_trades_history:")
for i, t in enumerate(trades):
    pnl = t.get('pnl')
    print(f"  #{i+1}: pnl={pnl} (type={type(pnl).__name__}), exit_time={t.get('exit_time')}, close_reason={t.get('close_reason')}")

# Проверяем closed_pnl
print(f"\n=== closed_pnl ===")
app_db = get_app_database()
closed_pnl = app_db.get_closed_pnl(sort_by='time', period='all')
es_trades = [t for t in closed_pnl if t.get('symbol') == 'ES']
print(f"Найдено {len(es_trades)} закрытых сделок ES в closed_pnl:")
for i, t in enumerate(es_trades[:5]):
    pnl = t.get('closed_pnl')
    print(f"  #{i+1}: closed_pnl={pnl} (type={type(pnl).__name__}), close_time={t.get('close_time')}")
