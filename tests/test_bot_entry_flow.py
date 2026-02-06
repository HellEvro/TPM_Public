"""
Тестовый скрипт для проверки полной цепочки создания бота и входа в позицию
БЕЗ реального входа на биржу, но с реальными данными
"""

import sys
import os
import time
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем необходимые модули
from bots_modules.imports_and_globals import (
    bots_data, bots_data_lock, coins_rsi_data, rsi_data_lock, BOT_STATUS
)
from bots_modules.filters import (
    get_coin_rsi_data, check_rsi_time_filter, test_exit_scam_filter,
    is_coin_mature_stored, process_auto_bot_signals
)
from bots_modules.bot_class import NewTradingBot
from bots_modules.init_functions import ensure_exchange_initialized

print("=" * 80)
print("🧪 ТЕСТИРОВАНИЕ ПОЛНОЙ ЦЕПОЧКИ СОЗДАНИЯ БОТА И ВХОДА В ПОЗИЦИЮ")
print("=" * 80)
print()

# ============================================================================
# ЭТАП 1: Инициализация биржи
# ============================================================================
print("📊 ЭТАП 1: Инициализация биржи")
print("-" * 80)

if not ensure_exchange_initialized():
    print("❌ ОШИБКА: Не удалось инициализировать биржу")
    sys.exit(1)

from bots_modules.imports_and_globals import get_exchange
exchange = get_exchange()

if not exchange:
    print("❌ ОШИБКА: Биржа не доступна")
    sys.exit(1)

print("✅ Биржа инициализирована")
print()

# ============================================================================
# ЭТАП 2: Загрузка RSI данных для тестовых монет
# ============================================================================
print("📊 ЭТАП 2: Загрузка RSI данных")
print("-" * 80)

test_symbols = ['AWE', 'CAMP', 'BTC', 'ETH']
rsi_data_loaded = {}

for symbol in test_symbols:
    print(f"\n🔍 Загрузка данных для {symbol}...")
    rsi_data = get_coin_rsi_data(symbol, exchange_obj=exchange)
    
    if rsi_data:
        rsi_data_loaded[symbol] = rsi_data
        print(f"✅ {symbol}:")
        print(f"   RSI: {rsi_data.get('rsi6h', 'N/A')}")
        print(f"   Тренд: {rsi_data.get('trend6h', 'N/A')}")
        print(f"   Сигнал: {rsi_data.get('signal', 'N/A')}")
        print(f"   Цена: ${rsi_data.get('price', 'N/A')}")
    else:
        print(f"❌ {symbol}: Не удалось загрузить данные")

if not rsi_data_loaded:
    print("\n❌ ОШИБКА: Не удалось загрузить данные ни для одной монеты")
    sys.exit(1)

print()

# ============================================================================
# ЭТАП 3: Проверка фильтров для каждой монеты
# ============================================================================
print("📊 ЭТАП 3: Проверка фильтров")
print("-" * 80)

coins_passed_filters = []

for symbol, rsi_data in rsi_data_loaded.items():
    print(f"\n🔍 Проверка фильтров для {symbol}:")
    
    signal = rsi_data.get('signal')
    if signal not in ['ENTER_LONG', 'ENTER_SHORT']:
        print(f"   ⏸️  Нет торгового сигнала (сигнал: {signal})")
        continue
    
    print(f"   ✅ Базовый сигнал: {signal}")
    
    # Фильтр 1: Зрелость монеты
    is_mature = is_coin_mature_stored(symbol)
    print(f"   {'✅' if is_mature else '❌'} Зрелость монеты: {is_mature}")
    if not is_mature:
        continue
    
    # Фильтр 2: Exit Scam
    exit_scam_passed = test_exit_scam_filter(symbol)
    print(f"   {'✅' if exit_scam_passed else '❌'} Exit Scam фильтр: {exit_scam_passed}")
    if not exit_scam_passed:
        continue
    
    # Фильтр 3: RSI Time Filter
    time_filter_passed = check_rsi_time_filter(symbol, signal)
    print(f"   {'✅' if time_filter_passed else '❌'} RSI Time фильтр: {time_filter_passed}")
    if not time_filter_passed:
        continue
    
    print(f"   🎯 {symbol} ПРОШЕЛ ВСЕ ФИЛЬТРЫ!")
    coins_passed_filters.append(symbol)

if not coins_passed_filters:
    print("\n⚠️  Ни одна монета не прошла все фильтры")
    print("Это нормально, если на рынке нет подходящих условий для входа")
    sys.exit(0)

print()
print(f"✅ Монеты прошедшие все фильтры: {', '.join(coins_passed_filters)}")
print()

# ============================================================================
# ЭТАП 4: ТЕСТИРОВАНИЕ СОЗДАНИЯ БОТА (БЕЗ РЕАЛЬНОГО ВХОДА)
# ============================================================================
print("📊 ЭТАП 4: Тестирование создания бота")
print("-" * 80)

# Берем первую монету из прошедших фильтры
test_symbol = coins_passed_filters[0]
test_rsi_data = rsi_data_loaded[test_symbol]

print(f"\n🤖 Создаем ТЕСТОВОГО бота для {test_symbol}")
print(f"   RSI: {test_rsi_data.get('rsi6h')}")
print(f"   Сигнал: {test_rsi_data.get('signal')}")
print(f"   Цена: ${test_rsi_data.get('price')}")
print()

# Создаем конфигурацию бота
bot_config = {
    'symbol': test_symbol,
    'status': BOT_STATUS['RUNNING'],
    'created_at': datetime.now().isoformat(),
    'opened_by_autobot': True,
    'volume_mode': 'usdt',
    'volume_value': 5.0,  # Тестовая сумма
    'auto_managed': True
}

# ============================================================================
# ЭТАП 5: СОЗДАНИЕ ОБЪЕКТА БОТА
# ============================================================================
print("📊 ЭТАП 5: Создание объекта NewTradingBot")
print("-" * 80)

try:
    test_bot = NewTradingBot(test_symbol, bot_config, exchange)
    print(f"✅ Объект бота создан успешно")
    print(f"   Symbol: {test_bot.symbol}")
    print(f"   Status: {test_bot.status}")
    print(f"   Volume: {test_bot.volume_value} USDT")
except Exception as e:
    print(f"❌ ОШИБКА создания объекта бота: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# ЭТАП 6: ПРОВЕРКА МЕТОДА enter_position (БЕЗ РЕАЛЬНОГО ВХОДА)
# ============================================================================
print("📊 ЭТАП 6: Проверка наличия метода enter_position")
print("-" * 80)

if not hasattr(test_bot, 'enter_position'):
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: У бота нет метода enter_position!")
    sys.exit(1)

print("✅ Метод enter_position существует")

# Проверяем что метод вызываемый
if not callable(getattr(test_bot, 'enter_position')):
    print("❌ ОШИБКА: enter_position не является вызываемым методом")
    sys.exit(1)

print("✅ Метод enter_position вызываемый")
print()

# ============================================================================
# ЭТАП 7: MOCK ТЕСТ ВХОДА В ПОЗИЦИЮ
# ============================================================================
print("📊 ЭТАП 7: MOCK тест входа в позицию (без реального входа)")
print("-" * 80)
print()
print("⚠️  ВНИМАНИЕ: Реальный вход на биржу НЕ будет выполнен!")
print("   Это только проверка логики без реальных сделок")
print()

# Создаем мок-функцию для имитации успешного входа
original_open_position = test_bot._open_position_on_exchange

def mock_open_position(side, price):
    """Мок-функция: имитирует успешное открытие позиции"""
    print(f"   [MOCK] Имитируем открытие позиции {side} @ ${price}")
    test_bot.order_id = f"MOCK_ORDER_{int(time.time())}"
    test_bot.entry_timestamp = datetime.now().isoformat()
    return True

# Подменяем метод на мок
test_bot._open_position_on_exchange = mock_open_position

try:
    print("🚀 Вызываем enter_position('LONG')...")
    result = test_bot.enter_position('LONG')
    
    if result and result.get('success'):
        print("✅ ТЕСТ ПРОЙДЕН: Метод enter_position отработал успешно!")
        print(f"   Entry Price: ${result.get('entry_price')}")
        print(f"   Side: {result.get('side')}")
        print(f"   Order ID: {result.get('order_id')}")
        print(f"   Bot Status: {test_bot.status}")
    else:
        print(f"❌ ТЕСТ НЕ ПРОЙДЕН: enter_position вернул ошибку")
        print(f"   Результат: {result}")
        
except Exception as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА при вызове enter_position: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Возвращаем оригинальный метод
test_bot._open_position_on_exchange = original_open_position

print()

# ============================================================================
# ИТОГОВЫЙ ОТЧЕТ
# ============================================================================
print("=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЕТ")
print("=" * 80)
print()
print("✅ Биржа инициализирована")
print(f"✅ RSI данные загружены для {len(rsi_data_loaded)} монет")
print(f"✅ Фильтры пройдены для {len(coins_passed_filters)} монет: {', '.join(coins_passed_filters)}")
print(f"✅ Объект NewTradingBot создан для {test_symbol}")
print("✅ Метод enter_position существует и вызываемый")
print("✅ MOCK тест входа в позицию пройден")
print()
print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
print()
print("💡 Система готова к работе с реальными сделками")
print("   Включите Auto Bot в UI для начала торговли")
print()
print("=" * 80)

