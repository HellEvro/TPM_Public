"""
Тест для проверки условий входа автобота в сделку

Проверяет:
1. Сколько монет проходит каждый фильтр
2. Какие монеты готовы для входа в сделку
3. Почему монеты не проходят фильтры
4. Статистику по всем фильтрам
"""

import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from datetime import datetime
from collections import defaultdict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger('AutoBotTest')

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

async def test_autobot_conditions():
    """Основной тест условий автобота"""

    print_header("🧪 ТЕСТ УСЛОВИЙ АВТОБОТА")

    try:
        # Импортируем необходимые модули
        import bots_modules.imports_and_globals as globals_module
        from bots_modules.filters import get_coin_rsi_data
        from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG

        exchange = globals_module.exchange
        bots_data = globals_module.bots_data
        load_auto_bot_config = globals_module.load_auto_bot_config

        # Загружаем конфигурацию
        load_auto_bot_config()
        config = bots_data.get('auto_bot_config', DEFAULT_AUTO_BOT_CONFIG)

        print(f"\n📋 Текущая конфигурация:")
        print(f"   • Включен: {config.get('enabled', False)}")
        print(f"   • Макс. ботов: {config.get('max_concurrent', 5)}")
        print(f"   • RSI LONG: ≤{config.get('rsi_long_threshold', 29)}")
        print(f"   • RSI SHORT: ≥{config.get('rsi_short_threshold', 71)}")
        print(f"   • Проверка зрелости: {config.get('enable_maturity_check', True)}")
        print(f"   • ExitScam фильтр: {config.get('exit_scam_enabled', True)}")
        print(f"   • Enhanced RSI: {config.get('enhanced_rsi_enabled', True)}")

        # Инициализируем биржу через init_bot_service
        if globals_module.exchange is None:
            print("\n⚙️  Инициализация биржи...")
            from bots_modules.init_functions import init_exchange_sync
            try:
                init_exchange_sync()
                exchange = globals_module.exchange
                print("✅ Биржа инициализирована")
            except Exception as e:
                print(f"❌ Ошибка инициализации биржи: {e}")
                return
        else:
            exchange = globals_module.exchange

        print_section("📊 Загрузка данных с биржи...")

        # Получаем список всех пар
        try:
            trading_pairs = await globals_module.exchange.get_trading_pairs()
            print(f"✅ Загружено {len(trading_pairs)} торговых пар")
        except Exception as e:
            print(f"❌ Ошибка загрузки торговых пар: {e}")
            return

        # Статистика по фильтрам
        stats = {
            'total': 0,
            'passed_scope': 0,
            'passed_basic_rsi': 0,
            'passed_trend': 0,
            'passed_maturity': 0,
            'passed_enhanced_rsi': 0,
            'passed_exitscam': 0,
            'passed_rsi_time': 0,
            'ready_for_trade': 0,
            'has_position': 0,
            'blocked_reasons': defaultdict(int)
        }

        ready_coins = {
            'LONG': [],
            'SHORT': []
        }

        print_section("🔍 Анализ монет...")

        # Ограничиваем количество монет для теста (первые 100)
        test_pairs = trading_pairs[:100]
        print(f"📊 Анализируем первые {len(test_pairs)} монет для теста\n")

        for i, symbol in enumerate(test_pairs, 1):
            stats['total'] += 1

            # Прогресс
            if i % 10 == 0:
                print(f"   Проверено: {i}/{len(test_pairs)} монет...", end='\r')

            try:
                # Получаем данные о монете через get_coin_rsi_data
                coin_data = await get_coin_rsi_data(symbol, globals_module.exchange)

                if not coin_data:
                    stats['blocked_reasons']['no_data'] += 1
                    continue

                # Проверяем каждый фильтр
                blocked_by = coin_data.get('blocked_by')
                signal = coin_data.get('signal', 'WAIT')
                effective_signal = coin_data.get('effective_signal', 'WAIT')

                # Фильтр 1: Scope (Whitelist/Blacklist)
                if not blocked_by or 'scope' not in blocked_by:
                    stats['passed_scope'] += 1
                else:
                    stats['blocked_reasons']['scope'] += 1
                    continue

                # Фильтр 2: Basic RSI
                if coin_data.get('rsi6h'):
                    rsi = coin_data['rsi6h']
                    if rsi <= config.get('rsi_long_threshold', 29) or rsi >= config.get('rsi_short_threshold', 71):
                        stats['passed_basic_rsi'] += 1
                    else:
                        stats['blocked_reasons']['basic_rsi'] += 1
                        continue
                else:
                    stats['blocked_reasons']['no_rsi'] += 1
                    continue

                # Фильтр 3: Trend
                if 'trend' not in (blocked_by or ''):
                    stats['passed_trend'] += 1
                else:
                    stats['blocked_reasons']['trend'] += 1
                    continue

                # Фильтр 4: Maturity
                if 'maturity' not in (blocked_by or ''):
                    stats['passed_maturity'] += 1
                else:
                    stats['blocked_reasons']['maturity'] += 1
                    continue

                # Фильтр 5: Enhanced RSI
                if 'enhanced_rsi' not in (blocked_by or ''):
                    stats['passed_enhanced_rsi'] += 1
                else:
                    stats['blocked_reasons']['enhanced_rsi'] += 1
                    continue

                # Фильтр 6: ExitScam
                if 'exitscam' not in (blocked_by or ''):
                    stats['passed_exitscam'] += 1
                else:
                    stats['blocked_reasons']['exitscam'] += 1
                    continue

                # Фильтр 7: RSI Time Filter
                if 'rsi_time' not in (blocked_by or ''):
                    stats['passed_rsi_time'] += 1
                else:
                    stats['blocked_reasons']['rsi_time'] += 1
                    continue

                # Проверка на существующую позицию
                if coin_data.get('has_bot') or coin_data.get('has_position'):
                    stats['has_position'] += 1
                    stats['blocked_reasons']['has_position'] += 1
                    continue

                # Если дошли сюда - монета готова к торговле!
                if effective_signal in ['ENTER_LONG', 'ENTER_SHORT']:
                    stats['ready_for_trade'] += 1

                    direction = 'LONG' if effective_signal == 'ENTER_LONG' else 'SHORT'
                    ready_coins[direction].append({
                        'symbol': symbol,
                        'rsi': coin_data.get('rsi6h'),
                        'trend': coin_data.get('trend'),
                        'signal': signal,
                        'effective_signal': effective_signal
                    })

            except Exception as e:
                stats['blocked_reasons']['error'] += 1

                continue

        print("\n")  # Очистка строки прогресса

        # Выводим результаты
        print_section("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")

        print(f"\n🎯 Воронка фильтров:")
        print(f"   1️⃣  Всего монет:                    {stats['total']}")
        print(f"   2️⃣  ✅ Прошли Scope фильтр:          {stats['passed_scope']} ({stats['passed_scope']/stats['total']*100:.1f}%)")
        print(f"   3️⃣  ✅ Прошли Basic RSI:              {stats['passed_basic_rsi']} ({stats['passed_basic_rsi']/stats['total']*100:.1f}%)")
        print(f"   4️⃣  ✅ Прошли Trend фильтр:           {stats['passed_trend']} ({stats['passed_trend']/stats['total']*100:.1f}%)")
        print(f"   5️⃣  ✅ Прошли Maturity фильтр:        {stats['passed_maturity']} ({stats['passed_maturity']/stats['total']*100:.1f}%)")
        print(f"   6️⃣  ✅ Прошли Enhanced RSI:           {stats['passed_enhanced_rsi']} ({stats['passed_enhanced_rsi']/stats['total']*100:.1f}%)")
        print(f"   7️⃣  ✅ Прошли ExitScam фильтр:        {stats['passed_exitscam']} ({stats['passed_exitscam']/stats['total']*100:.1f}%)")
        print(f"   8️⃣  ✅ Прошли RSI Time фильтр:        {stats['passed_rsi_time']} ({stats['passed_rsi_time']/stats['total']*100:.1f}%)")
        print(f"   9️⃣  ❌ Есть позиция:                  {stats['has_position']}")
        print(f"   🎯 ✅ ГОТОВЫ К ТОРГОВЛЕ:             {stats['ready_for_trade']} ({stats['ready_for_trade']/stats['total']*100:.1f}%)")

        print(f"\n🚫 Причины блокировки:")
        for reason, count in sorted(stats['blocked_reasons'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total'] * 100
            print(f"   • {reason:20s}: {count:3d} ({percentage:5.1f}%)")

        # Монеты готовые к торговле
        print_section("🎯 МОНЕТЫ ГОТОВЫЕ К ТОРГОВЛЕ")

        if ready_coins['LONG']:
            print(f"\n📈 LONG позиции ({len(ready_coins['LONG'])}):")
            for coin in ready_coins['LONG'][:10]:  # Показываем первые 10
                print(f"   • {coin['symbol']:12s} | RSI: {coin['rsi']:5.1f} | Trend: {coin['trend']:8s}")
        else:
            print("\n📈 LONG позиции: Нет монет готовых к входу")

        if ready_coins['SHORT']:
            print(f"\n📉 SHORT позиции ({len(ready_coins['SHORT'])}):")
            for coin in ready_coins['SHORT'][:10]:  # Показываем первые 10
                print(f"   • {coin['symbol']:12s} | RSI: {coin['rsi']:5.1f} | Trend: {coin['trend']:8s}")
        else:
            print("\n📉 SHORT позиции: Нет монет готовых к входу")

        # Выводы и рекомендации
        print_section("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ")

        if stats['ready_for_trade'] == 0:
            print("\n⚠️  НЕТ МОНЕТ ГОТОВЫХ К ТОРГОВЛЕ!")
            print("\n📋 Возможные причины:")

            # Анализируем самый строгий фильтр
            bottleneck = max(stats['blocked_reasons'].items(), key=lambda x: x[1])
            print(f"   • Самый строгий фильтр: {bottleneck[0]} (блокирует {bottleneck[1]} монет)")

            if bottleneck[0] == 'maturity':
                print(f"\n   💡 Рекомендация: Отключите или смягчите проверку зрелости")
                print(f"      Текущие настройки:")
                print(f"      - Минимум свечей: {config.get('min_candles_for_maturity', 400)}")
                print(f"      - Мин RSI low: {config.get('min_rsi_low', 35)}")
                print(f"      - Макс RSI high: {config.get('max_rsi_high', 65)}")

            elif bottleneck[0] == 'basic_rsi':
                print(f"\n   💡 Рекомендация: Расширьте диапазон RSI для входа")
                print(f"      Текущие пороги:")
                print(f"      - LONG: RSI ≤ {config.get('rsi_long_threshold', 29)}")
                print(f"      - SHORT: RSI ≥ {config.get('rsi_short_threshold', 71)}")
                print(f"      Попробуйте: LONG ≤35, SHORT ≥65")

            elif bottleneck[0] == 'enhanced_rsi':
                print(f"\n   💡 Рекомендация: Отключите Enhanced RSI или смягчите требования")
                print(f"      Это дополнительный фильтр, который может быть слишком строгим")

            elif bottleneck[0] == 'trend':
                print(f"\n   💡 Рекомендация: Отключите фильтры трендов")
                print(f"      Текущие настройки:")
                print(f"      - Избегать DOWN тренд для LONG: {config.get('avoid_down_trend', True)}")
                print(f"      - Избегать UP тренд для SHORT: {config.get('avoid_up_trend', True)}")

        elif stats['ready_for_trade'] < 5:
            print(f"\n⚠️  Мало монет готовых к торговле ({stats['ready_for_trade']})")
            print(f"\n   💡 Рекомендация: Рассмотрите смягчение фильтров для увеличения возможностей")

        else:
            print(f"\n✅ Хорошо! {stats['ready_for_trade']} монет готовы к торговле")
            print(f"   Бот имеет достаточно возможностей для входа в сделки")

        print_header("🎉 ТЕСТ ЗАВЕРШЕН")

    except Exception as e:
        print(f"\n❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print(f"\n🚀 Запуск теста: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    asyncio.run(test_autobot_conditions())
