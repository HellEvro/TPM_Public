#!/usr/bin/env python3
import requests
import time
import json
from datetime import datetime

def monitor_updates():
    """Мониторит обновления данных ботов каждую секунду"""
    print("🔍 Мониторинг обновлений данных ботов...")
    print("Нажмите Ctrl+C для остановки")
    
    last_update_time = None
    update_count = 0
    
    try:
        while True:
            try:
                # Получаем данные ботов
                response = requests.get('http://localhost:5001/api/bots/list')
                data = response.json()
                
                current_time = data.get('last_update', 'Неизвестно')
                bots = data.get('bots', [])
                
                # Находим бота AWE
                awe_bot = None
                for bot in bots:
                    if bot.get('symbol') == 'AWE':
                        awe_bot = bot
                        break
                
                if awe_bot:
                    pnl = awe_bot.get('unrealized_pnl_usdt', awe_bot.get('unrealized_pnl', 0))
                    current_price = awe_bot.get('current_price', 0)
                    entry_price = awe_bot.get('entry_price', 0)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Обновление #{update_count + 1}: "
                          f"PnL=${pnl:.3f}, "
                          f"Цена=${current_price:.6f}, "
                          f"Вход=${entry_price:.6f}, "
                          f"Время={current_time}")
                    
                    # Проверяем, изменилось ли время обновления
                    if last_update_time != current_time:
                        update_count += 1
                        last_update_time = current_time
                        print(f"  ✅ НОВОЕ ОБНОВЛЕНИЕ! #{update_count}")
                    else:
                        print(f"  ⏸️ То же время обновления")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Бот AWE не найден")
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
            
            time.sleep(1)  # Ждем 1 секунду
            
    except KeyboardInterrupt:
        print(f"\n📊 Итого обновлений за {update_count} секунд: {update_count}")
        print("Мониторинг остановлен")

if __name__ == "__main__":
    monitor_updates()
