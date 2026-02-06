#!/usr/bin/env python3
"""
Скрипт для проверки статуса монеты и причин, почему бот не запускается
"""
import sys
import os
import json
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bots_modules.imports_and_globals import (
    coins_rsi_data, bots_data, rsi_data_lock, bots_data_lock,
    get_exchange
)
from bots_modules.filters import get_effective_signal, get_coin_rsi_data

def check_coin_status(symbol):
    """Проверяет статус монеты и причины блокировки"""
    print(f"\n{'='*60}")
    print(f"🔍 ДИАГНОСТИКА МОНЕТЫ: {symbol}")
    print(f"{'='*60}\n")
    
    # Получаем данные монеты
    with rsi_data_lock:
        coin_data = coins_rsi_data['coins'].get(symbol)
    
    if not coin_data:
        print(f"❌ Монета {symbol} не найдена в данных RSI")
        print("   Попытка получить данные напрямую...")
        exchange = get_exchange()
        if exchange:
            coin_data = get_coin_rsi_data(symbol, exchange)
            if coin_data:
                print(f"✅ Данные получены напрямую")
            else:
                print(f"❌ Не удалось получить данные для {symbol}")
                return
        else:
            print(f"❌ Биржа не инициализирована")
            return
    
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
    
    # Основные данные
    rsi = coin_data.get(rsi_key, coin_data.get('rsi6h', 0))
    trend = coin_data.get(trend_key, coin_data.get('trend', coin_data.get('trend6h', 'UNKNOWN')))
    base_signal = coin_data.get('signal', 'WAIT')
    price = coin_data.get('price', 0)
    is_mature = coin_data.get('is_mature', False)
    
    # Enhanced RSI данные
    enhanced_rsi = coin_data.get('enhanced_rsi', {})
    enhanced_signal = enhanced_rsi.get('enhanced_signal') if enhanced_rsi else None
    enhanced_enabled = enhanced_rsi.get('enabled', False) if enhanced_rsi else False
    
    print(f"📊 ОСНОВНЫЕ ДАННЫЕ (ТФ: {current_timeframe}):")
    print(f"   RSI {current_timeframe.upper()}: {rsi:.2f}")
    print(f"   Тренд {current_timeframe.upper()}: {trend}")
    print(f"   Базовый сигнал: {base_signal}")
    print(f"   Цена: ${price:.6f}")
    print(f"   Зрелость: {'✅ Зрелая' if is_mature else '❌ Незрелая'}")
    print(f"\n🔬 ENHANCED RSI:")
    print(f"   Включена: {'✅ ДА' if enhanced_enabled else '❌ НЕТ'}")
    if enhanced_enabled and enhanced_signal:
        print(f"   Enhanced сигнал: {enhanced_signal}")
        if enhanced_rsi.get('enhanced_reason'):
            print(f"   Причина: {enhanced_rsi.get('enhanced_reason')}")
        if enhanced_rsi.get('warning_message'):
            print(f"   Предупреждение: {enhanced_rsi.get('warning_message')}")
    elif enhanced_enabled:
        print(f"   Enhanced сигнал: не определен")
    
    # Получаем эффективный сигнал
    effective_signal = get_effective_signal(coin_data)
    print(f"\n🎯 ЭФФЕКТИВНЫЙ СИГНАЛ: {effective_signal}")
    
    # Проверяем настройки автобота
    with bots_data_lock:
        auto_config = bots_data.get('auto_bot_config', {})
    
    avoid_down_trend = auto_config.get('avoid_down_trend', True)
    avoid_up_trend = auto_config.get('avoid_up_trend', True)
    rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)
    rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)
    
    print(f"\n⚙️ НАСТРОЙКИ АВТОБОТА:")
    print(f"   RSI для LONG: ≤{rsi_long_threshold}")
    print(f"   RSI для SHORT: ≥{rsi_short_threshold}")
    print(f"   Избегать DOWN тренд: {'✅ ВКЛ' if avoid_down_trend else '❌ ВЫКЛ'}")
    print(f"   Избегать UP тренд: {'✅ ВКЛ' if avoid_up_trend else '❌ ВЫКЛ'}")
    
    # Проверяем фильтры
    print(f"\n🔍 ПРОВЕРКА ФИЛЬТРОВ:")
    
    # 1. Проверка зрелости
    if not is_mature:
        print(f"   ❌ БЛОКИРОВКА: Монета незрелая")
        maturity_info = coin_data.get('maturity_info', {})
        if maturity_info:
            print(f"      Причина: {maturity_info.get('reason', 'Неизвестно')}")
    else:
        print(f"   ✅ Зрелость: пройдена")
    
    # 2. Проверка ExitScam
    blocked_by_exit_scam = coin_data.get('blocked_by_exit_scam', False)
    if blocked_by_exit_scam:
        print(f"   ❌ БЛОКИРОВКА: ExitScam фильтр")
        exit_scam_info = coin_data.get('exit_scam_info', {})
        if exit_scam_info:
            print(f"      Причина: {exit_scam_info.get('reason', 'Неизвестно')}")
    else:
        print(f"   ✅ ExitScam: пройден")
    
    # 3. Проверка RSI Time фильтра
    blocked_by_rsi_time = coin_data.get('blocked_by_rsi_time', False)
    if blocked_by_rsi_time:
        print(f"   ❌ БЛОКИРОВКА: RSI Time фильтр")
        rsi_time_info = coin_data.get('rsi_time_filter_info', {})
        if rsi_time_info:
            print(f"      Причина: {rsi_time_info.get('reason', 'Неизвестно')}")
    else:
        print(f"   ✅ RSI Time фильтр: пройден")
    
    # 4. Проверка тренда для LONG
    if base_signal == 'ENTER_LONG' or effective_signal == 'ENTER_LONG':
        if avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
            print(f"   ❌ БЛОКИРОВКА: LONG заблокирован фильтром тренда")
            print(f"      RSI {rsi:.2f} <= {rsi_long_threshold} И тренд = DOWN")
            print(f"      Решение: Отключите 'Избегать нисходящий тренд' в настройках")
        elif rsi > rsi_long_threshold:
            print(f"   ❌ БЛОКИРОВКА: RSI {rsi:.2f} > {rsi_long_threshold}")
        else:
            print(f"   ✅ LONG: условия выполнены")
    
    # 5. Проверка тренда для SHORT
    if base_signal == 'ENTER_SHORT' or effective_signal == 'ENTER_SHORT':
        if avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':
            print(f"   ❌ БЛОКИРОВКА: SHORT заблокирован фильтром тренда")
            print(f"      RSI {rsi:.2f} >= {rsi_short_threshold} И тренд = UP")
            print(f"      Решение: Отключите 'Избегать восходящий тренд' в настройках")
        elif rsi < rsi_short_threshold:
            print(f"   ❌ БЛОКИРОВКА: RSI {rsi:.2f} < {rsi_short_threshold}")
        else:
            print(f"   ✅ SHORT: условия выполнены")
    
    # Проверяем наличие бота
    with bots_data_lock:
        bot_exists = symbol in bots_data['bots']
    
    print(f"\n🤖 СТАТУС БОТА:")
    if bot_exists:
        bot_data = bots_data['bots'][symbol]
        print(f"   ✅ Бот существует")
        print(f"   Статус: {bot_data.get('status', 'UNKNOWN')}")
        print(f"   Позиция: {bot_data.get('position_side', 'Нет')}")
    else:
        print(f"   ❌ Бот не создан")
        if effective_signal == 'WAIT':
            print(f"   Причина: Эффективный сигнал = WAIT (блокировка фильтрами)")
        else:
            print(f"   Причина: Бот не был создан вручную или автоботом")
    
    # Итоговый вывод
    print(f"\n{'='*60}")
    print(f"📋 ИТОГОВЫЙ ВЫВОД:")
    print(f"{'='*60}")
    
    if effective_signal == 'WAIT':
        # Проверяем причину блокировки
        if base_signal == 'WAIT':
            if enhanced_enabled and enhanced_signal == 'WAIT':
                print(f"❌ BANK не запускается из-за Enhanced RSI анализа")
                print(f"   Enhanced RSI система изменила сигнал на WAIT")
                if enhanced_rsi.get('enhanced_reason'):
                    print(f"   Причина: {enhanced_rsi.get('enhanced_reason')}")
                print(f"\n📌 РЕШЕНИЕ:")
                print(f"   1. Отключите Enhanced RSI систему в настройках")
                print(f"   2. Или создайте бота вручную через кнопку 'Включить'")
            else:
                print(f"❌ BANK не запускается: базовый сигнал = WAIT")
                print(f"   RSI = {rsi:.2f}, но сигнал не ENTER_LONG")
                print(f"   Возможные причины:")
                print(f"   - Enhanced RSI изменил сигнал")
                print(f"   - Фильтр зрелости заблокировал")
                print(f"   - Другие фильтры (см. выше)")
        elif avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
            print(f"❌ BANK не запускается из-за фильтра 'Избегать нисходящий тренд'")
            print(f"")
            print(f"📌 РЕШЕНИЕ:")
            print(f"   1. Отключите настройку 'Избегать нисходящий тренд' в интерфейсе")
            print(f"   2. Или дождитесь изменения тренда на NEUTRAL/UP")
            print(f"   3. Или создайте бота вручную через кнопку 'Включить'")
        elif avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':
            print(f"❌ BANK не запускается из-за фильтра 'Избегать восходящий тренд'")
        else:
            print(f"❌ BANK не запускается из-за других фильтров (см. выше)")
    elif effective_signal in ['ENTER_LONG', 'ENTER_SHORT']:
        if not bot_exists:
            print(f"✅ Сигнал активен ({effective_signal}), но бот не создан")
            print(f"   Создайте бота вручную через кнопку 'Включить'")
        else:
            print(f"✅ Бот существует и должен работать")
    else:
        print(f"❓ Неизвестный статус: {effective_signal}")
    
    print(f"\n")

if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BANK'
    check_coin_status(symbol)

