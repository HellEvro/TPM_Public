#!/usr/bin/env python3
"""
Диагностика обновления данных ботов в реальном времени
Проверяет, почему данные о позициях не обновляются на странице
"""

import requests
import time
import json
from datetime import datetime

API = "http://127.0.0.1:5001"
UI = "http://127.0.0.1:5000"

def get(url, timeout=5):
    """Безопасный GET запрос"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text, None
    except requests.exceptions.RequestException as e:
        return 0, None, str(e)

def post(url, data=None, timeout=5):
    """Безопасный POST запрос"""
    try:
        response = requests.post(url, json=data, timeout=timeout)
        return response.status_code, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text, None
    except requests.exceptions.RequestException as e:
        return 0, None, str(e)

def check_bot_data_freshness():
    """Проверяет свежесть данных ботов"""
    print("\n🔍 ПРОВЕРКА СВЕЖЕСТИ ДАННЫХ БОТОВ:")
    print("=" * 60)
    
    # 1. Проверяем список ботов
    print("1. Список ботов:")
    code, js, err = get(f"{API}/api/bots/list")
    if code == 200 and isinstance(js, dict):
        bots = js.get('bots', [])
        print(f"   ✅ Ботов найдено: {len(bots)}")
        
        if bots:
            # Проверяем последнее обновление
            last_update = js.get('last_update', 'Неизвестно')
            print(f"   📅 Последнее обновление: {last_update}")
            
            # Проверяем данные первого бота
            first_bot = bots[0]
            print(f"   🤖 Первый бот: {first_bot.get('symbol')}")
            print(f"      Статус: {first_bot.get('status')}")
            print(f"      PnL: ${first_bot.get('unrealized_pnl', 0):.2f}")
            print(f"      Цена входа: ${first_bot.get('entry_price', 0):.6f}")
            print(f"      Сторона: {first_bot.get('position_side', 'None')}")
            print(f"      Создан: {first_bot.get('created_at', 'Неизвестно')}")
            
            # Проверяем время последнего обновления бота
            bot_last_update = first_bot.get('last_update')
            if bot_last_update:
                try:
                    bot_time = datetime.fromisoformat(bot_last_update.replace('Z', '+00:00'))
                    now = datetime.now()
                    time_diff = (now - bot_time.replace(tzinfo=None)).total_seconds()
                    print(f"      ⏰ Время с последнего обновления: {time_diff:.1f} сек")
                    
                    if time_diff > 60:
                        print(f"      ⚠️ ВНИМАНИЕ: Данные бота устарели на {time_diff:.1f} секунд!")
                    else:
                        print(f"      ✅ Данные бота свежие")
                except Exception as e:
                    print(f"      ❌ Ошибка парсинга времени: {e}")
        else:
            print("   📭 Ботов не найдено")
    else:
        print(f"   ❌ Ошибка получения списка ботов: {code}, {err}")
    
    # 2. Проверяем синхронизацию позиций
    print("\n2. Синхронизация позиций:")
    code, js, err = post(f"{API}/api/bots/sync-positions")
    if code == 200 and isinstance(js, dict):
        synced = js.get('synced', False)
        message = js.get('message', '')
        print(f"   ✅ Синхронизация: {'Выполнена' if synced else 'Не потребовалась'}")
        print(f"   📝 Сообщение: {message}")
    else:
        print(f"   ❌ Ошибка синхронизации: {code}, {err}")
    
    # 3. Проверяем статус сервиса
    print("\n3. Статус сервиса:")
    code, js, err = get(f"{API}/api/bots/status")
    if code == 200 and isinstance(js, dict):
        print(f"   ✅ Статус: {js.get('status')}")
        print(f"   📊 Монет загружено: {js.get('coins_loaded')}")
        print(f"   🔄 Обновление в процессе: {js.get('update_in_progress')}")
        print(f"   🤖 Ботов: {js.get('bots', {}).get('total')} (активных: {js.get('bots', {}).get('active')})")
        
        last_update = js.get('last_update')
        if last_update:
            try:
                service_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                now = datetime.now()
                time_diff = (now - service_time.replace(tzinfo=None)).total_seconds()
                print(f"   ⏰ Время с последнего обновления сервиса: {time_diff:.1f} сек")
                
                if time_diff > 30:
                    print(f"   ⚠️ ВНИМАНИЕ: Сервис не обновлялся {time_diff:.1f} секунд!")
                else:
                    print(f"   ✅ Сервис обновляется регулярно")
            except Exception as e:
                print(f"   ❌ Ошибка парсинга времени сервиса: {e}")
    else:
        print(f"   ❌ Ошибка получения статуса: {code}, {err}")

def monitor_bot_updates():
    """Мониторит обновления данных бота в реальном времени"""
    print("\n🔄 МОНИТОРИНГ ОБНОВЛЕНИЙ В РЕАЛЬНОМ ВРЕМЕНИ:")
    print("=" * 60)
    print("Наблюдаем за изменениями в течение 30 секунд...")
    print("Нажмите Ctrl+C для остановки")
    
    previous_data = {}
    
    try:
        for i in range(30):  # 30 секунд мониторинга
            print(f"\n⏰ Проверка #{i+1} ({datetime.now().strftime('%H:%M:%S')}):")
            
            # Получаем текущие данные
            code, js, err = get(f"{API}/api/bots/list")
            if code == 200 and isinstance(js, dict):
                bots = js.get('bots', [])
                
                if bots:
                    # Берем первый бот для мониторинга
                    current_bot = bots[0]
                    symbol = current_bot.get('symbol')
                    
                    # Создаем ключ для сравнения
                    current_key = f"{symbol}_{current_bot.get('status')}_{current_bot.get('unrealized_pnl', 0):.2f}_{current_bot.get('entry_price', 0):.6f}"
                    
                    if symbol in previous_data:
                        if previous_data[symbol] != current_key:
                            print(f"   🔄 ИЗМЕНЕНИЕ в {symbol}:")
                            print(f"      Статус: {current_bot.get('status')}")
                            print(f"      PnL: ${current_bot.get('unrealized_pnl', 0):.2f}")
                            print(f"      Цена входа: ${current_bot.get('entry_price', 0):.6f}")
                            print(f"      Сторона: {current_bot.get('position_side', 'None')}")
                            
                            # Обновляем предыдущие данные
                            previous_data[symbol] = current_key
                        else:
                            print(f"   ⏳ {symbol}: Данные не изменились")
                    else:
                        print(f"   🆕 Новый бот {symbol}:")
                        print(f"      Статус: {current_bot.get('status')}")
                        print(f"      PnL: ${current_bot.get('unrealized_pnl', 0):.2f}")
                        previous_data[symbol] = current_key
                else:
                    print("   📭 Ботов не найдено")
            else:
                print(f"   ❌ Ошибка получения данных: {code}, {err}")
            
            # Ждем 1 секунду
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Мониторинг остановлен пользователем")

def check_ui_proxy():
    """Проверяет работу UI прокси"""
    print("\n🌐 ПРОВЕРКА UI ПРОКСИ:")
    print("=" * 60)
    
    endpoints = [
        ("/api/bots/list", "Список ботов"),
        ("/api/bots/account-info", "Информация о счете"),
        ("/api/bots/sync-positions", "Синхронизация позиций"),
    ]
    
    for endpoint, description in endpoints:
        print(f"\n{description} ({endpoint}):")
        code, js, err = get(f"{UI}{endpoint}")
        if code == 200:
            print(f"   ✅ OK - {description}")
            if isinstance(js, dict) and 'bots' in js:
                print(f"   📊 Ботов через UI: {len(js.get('bots', []))}")
        else:
            print(f"   ❌ Ошибка {code}: {err}")

def main():
    print("🔍 ДИАГНОСТИКА ОБНОВЛЕНИЯ ДАННЫХ БОТОВ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 80)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ждем запуска серверов
    print("\n⏳ Ожидание запуска серверов (5 секунд)...")
    time.sleep(5)
    
    # Проверяем свежесть данных
    check_bot_data_freshness()
    
    # Проверяем UI прокси
    check_ui_proxy()
    
    # Мониторим обновления
    monitor_bot_updates()
    
    print("\n" + "=" * 80)
    print("🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 80)
    print("✅ Диагностика завершена")
    print("📊 Проверьте логи выше на предмет:")
    print("   - Устаревших данных ботов")
    print("   - Проблем с синхронизацией")
    print("   - Ошибок в UI прокси")
    print("   - Отсутствия обновлений в реальном времени")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Диагностика остановлена пользователем")
        exit(130)
