#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для синхронизации открытых позиций из app_data.db в bot_trades_history

Проблема: Открытые позиции из app_data.db -> positions не сохраняются
автоматически в bot_trades_history до тех пор, пока они не будут закрыты.

Этот скрипт:
1. Загружает открытые позиции из app_data.db -> positions
2. Загружает данные ботов из bots_data.db -> bots для получения entry_price, entry_time, entry_rsi, entry_trend
3. Проверяет, есть ли уже эти позиции в bot_trades_history со статусом OPEN
4. Если нет - создает записи в bot_trades_history со статусом OPEN
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("СИНХРОНИЗАЦИЯ ОТКРЫТЫХ ПОЗИЦИЙ В ИСТОРИЮ")
print("=" * 80)

try:
    from bot_engine.app_database import AppDatabase
    from bot_engine.bots_database import get_bots_database
    
    app_db = AppDatabase()
    bots_db = get_bots_database()
    
    # 1. Загружаем открытые позиции из app_data.db
    print("\n[1] Загрузка открытых позиций из app_data.db...")
    positions_data = app_db.load_positions_data()
    
    all_positions = []
    for category in ['high_profitable', 'profitable', 'losing']:
        positions = positions_data.get(category, [])
        all_positions.extend(positions)
    
    print(f"   [INFO] Найдено {len(all_positions)} открытых позиций")
    
    if not all_positions:
        print("   [WARNING] Нет открытых позиций!")
        sys.exit(0)
    
    # 2. Загружаем данные ботов для получения entry_price, entry_time, entry_rsi, entry_trend
    print("\n[2] Загрузка данных ботов из bots_data.db...")
    bots_state = bots_db.load_bots_state()
    bots_data = bots_state.get('bots', {})
    print(f"   [INFO] Загружено данных для {len(bots_data)} ботов")
    
    # 3. Загружаем RSI cache для получения текущего RSI/тренда
    print("\n[3] Загрузка RSI cache...")
    rsi_cache = bots_db.load_rsi_cache(max_age_hours=24.0)
    coins_rsi_data = rsi_cache.get('coins', {}) if rsi_cache else {}
    print(f"   [INFO] Загружено RSI данных для {len(coins_rsi_data)} монет")
    
    # 4. Проверяем существующие открытые позиции в bot_trades_history
    print("\n[4] Проверка существующих открытых позиций в bot_trades_history...")
    existing_open_trades = bots_db.get_bot_trades_history(status='OPEN', limit=None)
    existing_symbols = {trade.get('symbol') for trade in existing_open_trades}
    print(f"   [INFO] Уже есть {len(existing_open_trades)} открытых позиций в истории")
    
    # 5. Синхронизируем позиции
    print("\n[5] Синхронизация позиций...")
    synced_count = 0
    skipped_count = 0
    error_count = 0
    
    for position in all_positions:
        symbol = position.get('symbol', '')
        if not symbol:
            continue
        
        # Пропускаем, если уже есть в истории
        if symbol in existing_symbols:
            skipped_count += 1
            continue
        
        # Получаем данные бота для этой позиции
        bot_data = bots_data.get(symbol, {})
        
        # ✅ КРИТИЧНО: Получаем текущий таймфрейм из конфига
        try:
            from bot_engine.config_loader import get_current_timeframe, get_rsi_key, get_trend_key
            current_timeframe = get_current_timeframe()
            rsi_key = get_rsi_key(current_timeframe)
            trend_key = get_trend_key(current_timeframe)
        except Exception:
            current_timeframe = '6h'
            rsi_key = 'rsi6h'
            trend_key = 'trend6h'
        
        # Получаем текущий RSI/тренд из cache
        coin_rsi_data = coins_rsi_data.get(symbol, {})
        current_rsi = coin_rsi_data.get(rsi_key, coin_rsi_data.get('rsi6h'))
        current_trend = coin_rsi_data.get(trend_key, coin_rsi_data.get('trend6h', 'NEUTRAL'))
        
        # Получаем данные входа из бота
        entry_price = bot_data.get('entry_price')
        entry_time = bot_data.get('entry_time') or bot_data.get('position_start_time')
        entry_timestamp = bot_data.get('entry_timestamp')
        entry_rsi = bot_data.get('last_rsi') or current_rsi
        entry_trend = bot_data.get('entry_trend') or bot_data.get('last_trend', 'NEUTRAL') or current_trend
        position_side = bot_data.get('position_side') or position.get('side', 'LONG')
        
        # Если нет entry_price, пытаемся вычислить из текущей цены и PnL
        if not entry_price:
            current_price = coin_rsi_data.get('price')
            pnl = position.get('pnl', 0)
            size = position.get('size', 0)
            if current_price and pnl and size > 0:
                if position_side == 'LONG':
                    entry_price = current_price - (pnl / size)
                else:
                    entry_price = current_price + (pnl / size)
        
        # Если все еще нет entry_price, пропускаем
        if not entry_price:
            print(f"   [WARNING] {symbol}: Не удалось определить entry_price, пропускаем")
            error_count += 1
            continue
        
        # Формируем данные для сохранения
        try:
            if entry_time:
                try:
                    entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', ''))
                    if not entry_timestamp:
                        entry_timestamp = entry_time_dt.timestamp() * 1000
                except:
                    if not entry_timestamp:
                        entry_timestamp = datetime.now().timestamp() * 1000
            else:
                entry_time = datetime.now().isoformat()
                if not entry_timestamp:
                    entry_timestamp = datetime.now().timestamp() * 1000
            
            trade_data = {
                'bot_id': symbol,
                'symbol': symbol,
                'direction': position_side,
                'entry_price': entry_price,
                'exit_price': None,
                'entry_time': entry_time,
                'exit_time': None,
                'entry_timestamp': entry_timestamp,
                'exit_timestamp': None,
                'position_size_usdt': position.get('size'),
                'position_size_coins': position.get('size'),  # TODO: конвертировать если нужно
                'pnl': None,  # PnL будет обновляться при закрытии
                'roi': None,
                'status': 'OPEN',
                'close_reason': None,
                'decision_source': bot_data.get('decision_source', 'SCRIPT'),
                'ai_decision_id': bot_data.get('ai_decision_id'),
                'ai_confidence': bot_data.get('ai_confidence'),
                'entry_rsi': entry_rsi,
                'exit_rsi': None,
                'entry_trend': entry_trend,
                'exit_trend': None,
                'entry_volatility': None,
                'entry_volume_ratio': None,
                'is_successful': None,
                'is_simulated': False,
                'source': 'sync_from_positions',
                'order_id': None,
                'extra_data': {
                    'synced_at': datetime.now().isoformat(),
                    'position_data': position,
                    'bot_data': bot_data
                }
            }
            
            trade_id = bots_db.save_bot_trade_history(trade_data)
            if trade_id:
                print(f"   [OK] {symbol}: Позиция синхронизирована (ID: {trade_id})")
                synced_count += 1
            else:
                print(f"   [WARNING] {symbol}: Не удалось сохранить в историю")
                error_count += 1
        except Exception as e:
            print(f"   [ERROR] {symbol}: Ошибка синхронизации: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СИНХРОНИЗАЦИИ:")
    print(f"   Синхронизировано: {synced_count}")
    print(f"   Пропущено (уже есть): {skipped_count}")
    print(f"   Ошибок: {error_count}")
    print("=" * 80)
    
except Exception as e:
    print(f"\n[ERROR] Критическая ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

