#!/usr/bin/env python3
import requests
import json

def check_awe_bot_data():
    try:
        # Получаем данные ботов
        response = requests.get('http://localhost:5001/api/bots/list')
        data = response.json()
        
        # Находим бота AWE
        awe_bot = None
        for bot in data.get('bots', []):
            if bot.get('symbol') == 'AWE':
                awe_bot = bot
                break
        
        if not awe_bot:
            print("❌ Бот AWE не найден в списке ботов")
            return
        
        print("🤖 Данные бота AWE из API:")
        print(f"   Символ: {awe_bot.get('symbol')}")
        print(f"   Цена входа: {awe_bot.get('entry_price')}")
        print(f"   Текущая цена: {awe_bot.get('current_price')}")
        print(f"   Размер позиции: {awe_bot.get('position_size')}")
        print(f"   Нереализованный PnL: {awe_bot.get('unrealized_pnl')}")
        print(f"   PnL в USDT: {awe_bot.get('unrealized_pnl_usdt')}")
        
        exchange_pos = awe_bot.get('exchange_position', {})
        print(f"\n📊 Данные с биржи:")
        print(f"   Стоп-лосс: {exchange_pos.get('stop_loss')}")
        print(f"   Тейк-профит: {exchange_pos.get('take_profit')}")
        print(f"   Марка цена: {exchange_pos.get('mark_price')}")
        print(f"   Средняя цена: {exchange_pos.get('avg_price')}")
        print(f"   PnL с биржи: {exchange_pos.get('pnl')}")
        print(f"   ROI с биржи: {exchange_pos.get('roi')}")
        
        # Проверяем расчеты
        entry_price = awe_bot.get('entry_price', 0)
        current_price = awe_bot.get('current_price', 0)
        position_size = awe_bot.get('position_size', 0)
        
        if entry_price and current_price and position_size:
            volume_usdt = position_size * entry_price
            price_change = ((current_price - entry_price) / entry_price) * 100
            pnl_calculated = (current_price - entry_price) * position_size
            
            print(f"\n🧮 Проверка расчетов:")
            print(f"   Объем в USDT (расчет): {volume_usdt:.2f}")
            print(f"   Изменение цены (%): {price_change:.2f}%")
            print(f"   PnL (расчет): {pnl_calculated:.3f}")
            
            # Сравниваем с UI
            print(f"\n🔄 Сравнение с UI:")
            print(f"   UI показывает объем: 5.19 USDT")
            print(f"   Расчетный объем: {volume_usdt:.2f} USDT")
            print(f"   Совпадение: {'✅' if abs(volume_usdt - 5.19) < 0.01 else '❌'}")
            
            print(f"   UI показывает изменение: 1.62%")
            print(f"   Расчетное изменение: {price_change:.2f}%")
            print(f"   Совпадение: {'✅' if abs(price_change - 1.62) < 0.1 else '❌'}")
            
            print(f"   UI показывает PnL: $0.083")
            print(f"   Расчетный PnL: ${pnl_calculated:.3f}")
            print(f"   Совпадение: {'✅' if abs(pnl_calculated - 0.083) < 0.001 else '❌'}")
        
    except Exception as e:
        print(f"❌ Ошибка при получении данных: {e}")

if __name__ == "__main__":
    check_awe_bot_data()
