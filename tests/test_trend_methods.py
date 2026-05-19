#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест для сравнения разных методов определения тренда
"""

import sys
sys.path.append('.')

from bots_modules.init_functions import ensure_exchange_initialized
from bots_modules.filters import get_coin_rsi_data
from bots_modules.calculations import calculate_ema
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_trend_current(candles, ema_short_period, ema_long_period):
    """Текущая логика (строгие условия)"""
    closes = [c['close'] for c in candles]
    
    ema_short = calculate_ema(closes, ema_short_period)
    ema_long = calculate_ema(closes, ema_long_period)
    
    if ema_short is None or ema_long is None:
        return 'NEUTRAL'
    
    current_close = closes[-1]
    
    # Наклон EMA_long
    if len(closes) >= ema_long_period + 1:
        prev_ema_long = calculate_ema(closes[:-1], ema_long_period)
        ema_long_slope = ema_long - prev_ema_long if prev_ema_long else 0
    else:
        ema_long_slope = 0
    
    # 3 свечи подряд
    recent_closes = closes[-3:]
    all_above_ema_long = all(close > ema_long for close in recent_closes)
    all_below_ema_long = all(close < ema_long for close in recent_closes)
    
    # Определяем тренд (все 4 условия)
    if (current_close > ema_long and 
        ema_short > ema_long and 
        ema_long_slope > 0 and 
        all_above_ema_long):
        return 'UP'
    
    elif (current_close < ema_long and 
          ema_short < ema_long and 
          ema_long_slope < 0 and 
          all_below_ema_long):
        return 'DOWN'
    
    return 'NEUTRAL'


def analyze_trend_improved(candles, ema_short_period, ema_long_period):
    """Улучшенная логика (мягкие условия - 2 из 3 подтверждений)"""
    closes = [c['close'] for c in candles]
    
    ema_short = calculate_ema(closes, ema_short_period)
    ema_long = calculate_ema(closes, ema_long_period)
    
    if ema_short is None or ema_long is None:
        return 'NEUTRAL'
    
    current_close = closes[-1]
    
    # Наклон EMA_long (опциональный)
    if len(closes) >= ema_long_period + 1:
        prev_ema_long = calculate_ema(closes[:-1], ema_long_period)
        ema_long_slope = ema_long - prev_ema_long if prev_ema_long else 0
    else:
        ema_long_slope = 0
    
    # Мягкое подтверждение свечами (2 из 3)
    recent_closes = closes[-3:]
    closes_above = sum(1 for c in recent_closes if c > ema_long)
    closes_below = sum(1 for c in recent_closes if c < ema_long)
    
    # Основной сигнал: крест EMA
    ema_cross_up = ema_short > ema_long
    ema_cross_down = ema_short < ema_long
    
    # UP: если крест вверх + минимум 2 подтверждения
    if ema_cross_up:
        confirmations = 0
        if current_close > ema_long: confirmations += 1
        if ema_long_slope > 0: confirmations += 1
        if closes_above >= 2: confirmations += 1
        
        if confirmations >= 2:
            return 'UP'
    
    # DOWN: если крест вниз + минимум 2 подтверждения
    elif ema_cross_down:
        confirmations = 0
        if current_close < ema_long: confirmations += 1
        if ema_long_slope < 0: confirmations += 1
        if closes_below >= 2: confirmations += 1
        
        if confirmations >= 2:
            return 'DOWN'
    
    return 'NEUTRAL'


def analyze_trend_score_based(candles, ema_short_period, ema_long_period):
    """Score-based логика (гибкая оценка)"""
    closes = [c['close'] for c in candles]
    
    ema_short = calculate_ema(closes, ema_short_period)
    ema_long = calculate_ema(closes, ema_long_period)
    
    if ema_short is None or ema_long is None:
        return 'NEUTRAL'
    
    current_close = closes[-1]
    
    # Наклон EMA_long
    if len(closes) >= ema_long_period + 1:
        prev_ema_long = calculate_ema(closes[:-1], ema_long_period)
        ema_long_slope = ema_long - prev_ema_long if prev_ema_long else 0
    else:
        ema_long_slope = 0
    
    recent_closes = closes[-3:]
    all_above_ema_long = all(close > ema_long for close in recent_closes)
    all_below_ema_long = all(close < ema_long for close in recent_closes)
    
    # Считаем баллы
    score = 0
    
    # Основной критерий (+2 балла)
    if ema_short > ema_long:
        score += 2
    elif ema_short < ema_long:
        score -= 2
    
    # Цена относительно EMA_long (+1 балл)
    if current_close > ema_long:
        score += 1
    elif current_close < ema_long:
        score -= 1
    
    # Наклон длинной EMA (+1 балл)
    if ema_long_slope > 0:
        score += 1
    elif ema_long_slope < 0:
        score -= 1
    
    # Последние 3 свечи (+1 балл)
    if all_above_ema_long:
        score += 1
    elif all_below_ema_long:
        score -= 1
    
    # Определяем тренд
    if score >= 3:
        return 'UP'
    elif score <= -3:
        return 'DOWN'
    else:
        return 'NEUTRAL'


def test_trend_methods():
    """Сравнивает разные методы определения тренда"""
    print("\n" + "="*80)
    print("🧪 ТЕСТ МЕТОДОВ ОПРЕДЕЛЕНИЯ ТРЕНДА")
    print("="*80)
    
    # Инициализируем биржу
    print("\n📊 Инициализация биржи...")
    ensure_exchange_initialized()
    
    # Тестовые монеты (смесь с разными RSI)
    test_symbols = [
        'ATH',   # RSI: 25.6, trend: DOWN (текущая логика)
        'CARV',  # RSI: 23.5, trend: DOWN
        'FLR',   # RSI: 24.0, trend: DOWN
        'APT',   # RSI: 28.2, trend: DOWN
        'BEAM',  # RSI: 28.0, trend: DOWN
        'BTC',   # RSI: 31.8, trend: DOWN
        'ETH',   # RSI: 40.9
        'BNB',   # RSI: 40.9
    ]
    
    print(f"\n🎯 Тестируем {len(test_symbols)} монет...")
    print()
    
    results = {
        'current': {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0},
        'improved': {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0},
        'score_based': {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
    }
    
    from bots_modules.imports_and_globals import get_exchange
    exchange = get_exchange()
    
    for symbol in test_symbols:
        # Получаем данные
        coin_data = get_coin_rsi_data(symbol)
        if not coin_data:
            continue
        
        rsi = coin_data.get('rsi6h', 0)
        current_trend = coin_data.get('trend6h', 'NEUTRAL')
        
        # Получаем свечи
        chart_response = exchange.get_chart_data(symbol, '6h', '60d')
        if not chart_response or not chart_response.get('success'):
            continue
        
        candles = chart_response['data']['candles']
        if len(candles) < 250:
            continue
        
        # Используем базовые EMA-параметры (legacy модуль удален)
        ema_short = 50
        ema_long = 200
        
        # Применяем разные методы
        trend_current = analyze_trend_current(candles, ema_short, ema_long)
        trend_improved = analyze_trend_improved(candles, ema_short, ema_long)
        trend_score = analyze_trend_score_based(candles, ema_short, ema_long)
        
        # Обновляем статистику
        results['current'][trend_current] += 1
        results['improved'][trend_improved] += 1
        results['score_based'][trend_score] += 1
        
        # Выводим результаты
        in_long_zone = '📈' if rsi <= 29 else '  '
        in_short_zone = '📉' if rsi >= 71 else '  '
        mature = '💎' if coin_data.get('is_mature') else '  '
        
        print(f"{in_long_zone}{in_short_zone}{mature} {symbol:12} | RSI: {rsi:5.1f} | EMA: ({ema_short:3d},{ema_long:3d})")
        print(f"     Текущая:      {trend_current:7} (оригинал)")
        print(f"     Улучшенная:   {trend_improved:7}")
        print(f"     Score-based:  {trend_score:7}")
        
        # Анализ различий
        if trend_current != trend_improved or trend_current != trend_score:
            print(f"     💡 Различия обнаружены!")
            if rsi <= 29 and trend_current == 'DOWN' and trend_improved != 'DOWN':
                print(f"        ⚠️  Текущая логика блокирует потенциальный LONG!")
        print()
    
    # Итоговая статистика
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    print()
    
    total = sum(results['current'].values())
    
    for method_name, method_results in results.items():
        print(f"📌 {method_name.upper()}:")
        for trend_type, count in method_results.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   {trend_type:7}: {count:2} ({percentage:5.1f}%)")
        neutral_ratio = (method_results['NEUTRAL'] / total * 100) if total > 0 else 0
        print(f"   → NEUTRAL ratio: {neutral_ratio:.1f}%")
        print()
    
    # Анализ
    print("="*80)
    print("💡 АНАЛИЗ")
    print("="*80)
    print()
    
    neutral_current = results['current']['NEUTRAL'] / total * 100
    neutral_improved = results['improved']['NEUTRAL'] / total * 100
    neutral_score = results['score_based']['NEUTRAL'] / total * 100
    
    print(f"Текущая логика дает {neutral_current:.1f}% NEUTRAL")
    print(f"Улучшенная логика дает {neutral_improved:.1f}% NEUTRAL")
    print(f"Score-based логика дает {neutral_score:.1f}% NEUTRAL")
    print()
    
    if neutral_current > 50:
        print("⚠️  Текущая логика слишком консервативна (>50% NEUTRAL)")
        print("   Рекомендуется использовать улучшенную или score-based логику")
    
    print()
    print("="*80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*80)


if __name__ == '__main__':
    test_trend_methods()

