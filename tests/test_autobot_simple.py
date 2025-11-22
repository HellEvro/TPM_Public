"""
Простой тест для проверки условий входа автобота
Использует HTTP API работающего сервера
"""

import requests
import sys
from collections import defaultdict

# URL API сервера
API_URL = "http://localhost:5001/api"

def print_header(text):
    """Красивый заголовок"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_section(text):
    """Секция"""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80)

def main():
    print_header("🧪 ТЕСТ УСЛОВИЙ АВТОБОТА")
    
    try:
        # 1. Получаем конфигурацию
        print("\n📋 Загрузка конфигурации...")
        response = requests.get(f"{API_URL}/bots/auto-bot", timeout=10)
        if response.status_code != 200:
            print(f"❌ Ошибка загрузки конфигурации: {response.status_code}")
            return
        
        config = response.json()
        print(f"\n📋 Текущая конфигурация:")
        print(f"   • Включен: {config.get('enabled', False)}")
        print(f"   • Макс. ботов: {config.get('max_concurrent', 5)}")
        print(f"   • RSI LONG: ≤{config.get('rsi_long_threshold', 29)}")
        print(f"   • RSI SHORT: ≥{config.get('rsi_short_threshold', 71)}")
        print(f"   • Проверка зрелости: {config.get('enable_maturity_check', True)}")
        print(f"   • ExitScam фильтр: {config.get('exit_scam_enabled', True)}")
        
        # 2. Получаем список монет с RSI
        print_section("📊 Загрузка данных монет с биржи...")
        response = requests.get(f"{API_URL}/bots/coins-with-rsi", timeout=30)
        if response.status_code != 200:
            print(f"❌ Ошибка загрузки монет: {response.status_code}")
            return
        
        response_data = response.json()
        coins_dict = response_data.get('coins', {})
        coins = list(coins_dict.values())
        print(f"✅ Загружено {len(coins)} монет")
        
        # Проверяем есть ли поле is_mature в данных
        sample_coins = list(coins_dict.keys())[:5]
        print(f"\n🔍 Проверка данных (первые 5 монет):")
        for sym in sample_coins:
            coin = coins_dict[sym]
            print(f"   {sym}: is_mature = {coin.get('is_mature', 'НЕТ ПОЛЯ')}")
        
        # 3. Анализируем фильтры
        print_section("🔍 Анализ монет через фильтры...")
        
        stats = {
            'total': len(coins),
            'has_rsi': 0,
            'in_long_zone': 0,
            'in_short_zone': 0,
            'has_signal': 0,
            'enter_long': 0,
            'enter_short': 0,
            'wait': 0,
            'has_position': 0,
            'has_bot': 0,
            'is_mature': 0,
            'blocked_reasons': defaultdict(int)
        }
        
        ready_coins = {
            'LONG': [],
            'SHORT': []
        }
        
        blocked_details = []  # Для детального анализа
        
        for coin in coins:
            symbol = coin.get('symbol', 'UNKNOWN')
            rsi = coin.get('rsi6h')
            signal = coin.get('signal', 'WAIT')
            effective_signal = coin.get('effective_signal', 'WAIT')
            blocked_by = coin.get('blocked_by', '')
            has_bot = coin.get('has_bot', False)
            has_position = coin.get('has_position', False)
            is_mature = coin.get('is_mature', False)
            
            # Статистика
            if rsi is not None:
                stats['has_rsi'] += 1
                
                if rsi <= config.get('rsi_long_threshold', 29):
                    stats['in_long_zone'] += 1
                elif rsi >= config.get('rsi_short_threshold', 71):
                    stats['in_short_zone'] += 1
            
            if signal != 'WAIT':
                stats['has_signal'] += 1
            
            if effective_signal == 'ENTER_LONG':
                stats['enter_long'] += 1
            elif effective_signal == 'ENTER_SHORT':
                stats['enter_short'] += 1
            elif effective_signal == 'WAIT':
                stats['wait'] += 1
            
            if has_bot:
                stats['has_bot'] += 1
            
            if has_position:
                stats['has_position'] += 1
            
            if is_mature:
                stats['is_mature'] += 1
            
            # Причины блокировки
            if blocked_by:
                for reason in blocked_by.split(','):
                    reason = reason.strip()
                    if reason:
                        stats['blocked_reasons'][reason] += 1
            
            # Если монета в зоне RSI но не готова к торговле - анализируем почему
            if (rsi is not None and 
                (rsi <= config.get('rsi_long_threshold', 29) or rsi >= config.get('rsi_short_threshold', 71)) and
                effective_signal == 'WAIT'):
                if not blocked_by:
                    stats['blocked_reasons']['unknown'] += 1
                    # Сохраняем детали для анализа
                    blocked_details.append({
                        'symbol': symbol,
                        'rsi': rsi,
                        'signal': signal,
                        'effective_signal': effective_signal,
                        'blocked_by': blocked_by,
                        'is_mature': is_mature,
                        'has_bot': has_bot,
                        'has_position': has_position
                    })
            
            # Готовые к торговле
            if effective_signal in ['ENTER_LONG', 'ENTER_SHORT'] and not has_bot and not has_position:
                direction = 'LONG' if effective_signal == 'ENTER_LONG' else 'SHORT'
                ready_coins[direction].append({
                    'symbol': symbol,
                    'rsi': rsi,
                    'signal': signal,
                    'effective_signal': effective_signal,
                    'blocked_by': blocked_by,
                    'is_mature': is_mature
                })
        
        # Выводим результаты
        print_section("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
        
        print(f"\n🎯 Воронка фильтров:")
        print(f"   1️⃣  Всего монет:                    {stats['total']}")
        print(f"   2️⃣  ✅ Имеют RSI данные:             {stats['has_rsi']} ({stats['has_rsi']/stats['total']*100:.1f}%)")
        print(f"   3️⃣  📈 В зоне LONG (RSI ≤{config.get('rsi_long_threshold', 29)}):   {stats['in_long_zone']} ({stats['in_long_zone']/stats['total']*100:.1f}%)")
        print(f"   4️⃣  📉 В зоне SHORT (RSI ≥{config.get('rsi_short_threshold', 71)}): {stats['in_short_zone']} ({stats['in_short_zone']/stats['total']*100:.1f}%)")
        print(f"   5️⃣  🎯 Имеют сигнал:                 {stats['has_signal']} ({stats['has_signal']/stats['total']*100:.1f}%)")
        print(f"   6️⃣  💎 Зрелые монеты:                {stats['is_mature']} ({stats['is_mature']/stats['total']*100:.1f}%)")
        print(f"   7️⃣  🤖 Уже есть бот:                 {stats['has_bot']} ({stats['has_bot']/stats['total']*100:.1f}%)")
        print(f"   8️⃣  ✋ Ручная позиция:               {stats['has_position']} ({stats['has_position']/stats['total']*100:.1f}%)")
        print(f"   9️⃣  🟢 ENTER_LONG:                   {stats['enter_long']} ({stats['enter_long']/stats['total']*100:.1f}%)")
        print(f"   🔟  🔴 ENTER_SHORT:                  {stats['enter_short']} ({stats['enter_short']/stats['total']*100:.1f}%)")
        print(f"   ⏸️   ⏸️  WAIT:                         {stats['wait']} ({stats['wait']/stats['total']*100:.1f}%)")
        
        if stats['blocked_reasons']:
            print(f"\n🚫 Причины блокировки:")
            for reason, count in sorted(stats['blocked_reasons'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / stats['total'] * 100
                print(f"   • {reason:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Показываем детали заблокированных монет
        if blocked_details:
            print(f"\n🔍 Детали заблокированных монет (первые 10):")
            for detail in blocked_details[:10]:
                mature_icon = "💎" if detail['is_mature'] else "❌"
                blocked_text = f"Blocked: {detail['blocked_by']}" if detail['blocked_by'] else "No blocked_by field"
                print(f"   {mature_icon} {detail['symbol']:12s} | RSI: {detail['rsi']:5.1f} | Signal: {detail['signal']:12s} | {blocked_text}")
        
        # Монеты готовые к торговле
        print_section("🎯 МОНЕТЫ ГОТОВЫЕ К ТОРГОВЛЕ")
        
        total_ready = len(ready_coins['LONG']) + len(ready_coins['SHORT'])
        
        if ready_coins['LONG']:
            print(f"\n📈 LONG позиции ({len(ready_coins['LONG'])}):")
            for coin in ready_coins['LONG'][:15]:  # Показываем первые 15
                mature_mark = "💎" if coin['is_mature'] else "  "
                print(f"   {mature_mark} {coin['symbol']:12s} | RSI: {coin['rsi']:5.1f}")
            if len(ready_coins['LONG']) > 15:
                print(f"   ... и еще {len(ready_coins['LONG']) - 15} монет")
        else:
            print("\n📈 LONG позиции: Нет монет готовых к входу")
        
        if ready_coins['SHORT']:
            print(f"\n📉 SHORT позиции ({len(ready_coins['SHORT'])}):")
            for coin in ready_coins['SHORT'][:15]:  # Показываем первые 15
                mature_mark = "💎" if coin['is_mature'] else "  "
                print(f"   {mature_mark} {coin['symbol']:12s} | RSI: {coin['rsi']:5.1f}")
            if len(ready_coins['SHORT']) > 15:
                print(f"   ... и еще {len(ready_coins['SHORT']) - 15} монет")
        else:
            print("\n📉 SHORT позиции: Нет монет готовых к входу")
        
        # Выводы и рекомендации
        print_section("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ")
        
        if total_ready == 0:
            print("\n⚠️  НЕТ МОНЕТ ГОТОВЫХ К ТОРГОВЛЕ!")
            print("\n📋 Возможные причины:")
            
            if stats['blocked_reasons']:
                # Анализируем самый строгий фильтр
                bottleneck = max(stats['blocked_reasons'].items(), key=lambda x: x[1])
                print(f"   • Самый строгий фильтр: {bottleneck[0]} (блокирует {bottleneck[1]} монет)")
                
                if 'maturity' in bottleneck[0].lower():
                    print(f"\n   💡 Рекомендация: Отключите или смягчите проверку зрелости")
                    print(f"      Текущие настройки:")
                    print(f"      - Минимум свечей: {config.get('min_candles_for_maturity', 400)}")
                    print(f"      - Мин RSI low: {config.get('min_rsi_low', 35)}")
                    print(f"      - Макс RSI high: {config.get('max_rsi_high', 65)}")
                
                elif 'rsi' in bottleneck[0].lower():
                    print(f"\n   💡 Рекомендация: Расширьте диапазон RSI для входа")
                    print(f"      Текущие пороги:")
                    print(f"      - LONG: RSI ≤ {config.get('rsi_long_threshold', 29)}")
                    print(f"      - SHORT: RSI ≥ {config.get('rsi_short_threshold', 71)}")
                    print(f"      Попробуйте: LONG ≤35, SHORT ≥65")
                
                elif 'trend' in bottleneck[0].lower():
                    print(f"\n   💡 Рекомендация: Отключите фильтры трендов")
                    print(f"      Текущие настройки:")
                    print(f"      - Избегать DOWN тренд для LONG: {config.get('avoid_down_trend', True)}")
                    print(f"      - Избегать UP тренд для SHORT: {config.get('avoid_up_trend', True)}")
            
            # Дополнительный анализ
            if stats['in_long_zone'] == 0 and stats['in_short_zone'] == 0:
                print(f"\n   ⚠️  НИ ОДНА монета не в зоне RSI для входа!")
                print(f"      Это означает, что рынок сейчас в нейтральной зоне")
                print(f"      Рекомендация: Расширьте пороги RSI или дождитесь экстремумов")
        
        elif total_ready < 5:
            print(f"\n⚠️  Мало монет готовых к торговле ({total_ready})")
            print(f"\n   💡 Рекомендация: Рассмотрите смягчение фильтров")
            print(f"      - Расширьте пороги RSI")
            print(f"      - Отключите Enhanced RSI")
            print(f"      - Смягчите требования к зрелости")
        
        else:
            print(f"\n✅ Отлично! {total_ready} монет готовы к торговле")
            print(f"   ({len(ready_coins['LONG'])} LONG + {len(ready_coins['SHORT'])} SHORT)")
            print(f"\n   Бот имеет достаточно возможностей для входа в сделки")
            
            if config.get('max_concurrent', 5) < total_ready:
                print(f"\n   💡 Подсказка: У вас установлено макс. {config.get('max_concurrent', 5)} ботов")
                print(f"      При наличии {total_ready} возможностей, можно увеличить лимит")
        
        print_header("🎉 ТЕСТ ЗАВЕРШЕН")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Не удалось подключиться к серверу!")
        print("   Убедитесь, что сервер запущен на http://localhost:5001")
    except Exception as e:
        print(f"\n❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    from datetime import datetime
    print(f"\n🚀 Запуск теста: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()

