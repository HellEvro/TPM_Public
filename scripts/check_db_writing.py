#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка записи данных в БД"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bot_engine.ai.ai_database import get_ai_database
from datetime import datetime

def main():
    print("=" * 80)
    print("ПРОВЕРКА ЗАПИСИ ДАННЫХ В БД")
    print("=" * 80)
    print()
    
    # Проверяем путь к БД
    try:
        db = get_ai_database()
        print(f"✅ БД инициализирована: {db.db_path}")
        print(f"   Существует: {os.path.exists(db.db_path)}")
        if os.path.exists(db.db_path):
            size = os.path.getsize(db.db_path)
            print(f"   Размер: {size / 1024 / 1024:.2f} MB")
        print()
    except Exception as e:
        print(f"❌ Ошибка инициализации БД: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Проверяем статистику
    try:
        stats = db.get_database_stats()
        print("📊 Статистика БД:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print()
    except Exception as e:
        print(f"⚠️ Ошибка получения статистики: {e}")
        print()
    
    # Пробуем записать тестовую симуляцию
    try:
        test_trade = {
            'symbol': 'TESTUSDT',
            'direction': 'LONG',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'entry_time': int(datetime.now().timestamp()),
            'exit_time': int(datetime.now().timestamp()),
            'entry_rsi': 30.0,
            'exit_rsi': 70.0,
            'entry_trend': 'UP',
            'exit_trend': 'UP',
            'entry_volatility': 0.02,
            'entry_volume_ratio': 1.5,
            'pnl': 5.0,
            'pnl_pct': 5.0,
            'roi': 5.0,
            'exit_reason': 'TAKE_PROFIT',
            'is_successful': True,
            'duration_candles': 10,
            'entry_idx': 0,
            'exit_idx': 10,
            'simulation_timestamp': datetime.now().isoformat(),
            'config_params': {'test': 'value'},
            'filters_params': {'test': 'value'},
            'entry_conditions': {'test': 'value'},
            'exit_conditions': {'test': 'value'},
            'restrictions': {'test': 'value'},
        }
        
        print("🧪 Тест записи симуляции в БД...")
        saved_count = db.save_simulated_trades([test_trade])
        print(f"✅ Сохранено симуляций: {saved_count}")
        
        # Проверяем, что данные действительно записались
        count = db.count_simulated_trades()
        print(f"✅ Всего симуляций в БД: {count}")
        print()
    except Exception as e:
        print(f"❌ Ошибка записи тестовой симуляции: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Пробуем записать тестовую сделку бота
    try:
        test_bot_trade = {
            'bot_id': 'TEST_BOT',
            'symbol': 'TESTUSDT',
            'direction': 'LONG',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'entry_time': datetime.now().isoformat(),
            'exit_time': datetime.now().isoformat(),
            'entry_rsi': 30.0,
            'exit_rsi': 70.0,
            'entry_trend': 'UP',
            'exit_trend': 'UP',
            'entry_volatility': 0.02,
            'entry_volume_ratio': 1.5,
            'pnl': 5.0,
            'pnl_pct': 5.0,
            'roi': 5.0,
            'exit_reason': 'TAKE_PROFIT',
            'is_successful': True,
            'decision_source': 'SCRIPT',
            'position_size_usdt': 100.0,
            'position_size_coins': 1.0,
        }
        
        print("🧪 Тест записи сделки бота в БД...")
        trade_id = db.save_bot_trade(test_bot_trade)
        if trade_id:
            print(f"✅ Сделка бота сохранена (ID: {trade_id})")
        else:
            print("⚠️ Сделка бота не была сохранена (возможно, дубликат)")
        print()
    except Exception as e:
        print(f"❌ Ошибка записи тестовой сделки бота: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Финальная статистика
    try:
        stats = db.get_database_stats()
        print("📊 Финальная статистика БД:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print()
    except Exception as e:
        print(f"⚠️ Ошибка получения финальной статистики: {e}")
        print()
    
    print("=" * 80)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)

if __name__ == "__main__":
    main()

