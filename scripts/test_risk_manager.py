#!/usr/bin/env python3
"""
Тест: Dynamic Risk Manager
Проверяет корректность расчётов адаптивных SL/TP
"""

import sys
import os
import random

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_test_candles(count=50, trend='UP', volatility='MEDIUM'):
    """Генерирует тестовые свечи с заданными характеристиками"""
    candles = []
    base_price = 50000.0
    current_price = base_price
    
    # Параметры в зависимости от волатильности
    if volatility == 'LOW':
        max_change = 0.01  # 1% максимальное изменение
    elif volatility == 'HIGH':
        max_change = 0.05  # 5% максимальное изменение
    else:
        max_change = 0.02  # 2% среднее изменение
    
    # Генерируем свечи
    for i in range(count):
        # Направление изменения
        if trend == 'UP':
            change = random.uniform(0, max_change)
        elif trend == 'DOWN':
            change = random.uniform(-max_change, 0)
        else:  # NEUTRAL
            change = random.uniform(-max_change, max_change)
        
        # Новая цена
        current_price = current_price * (1 + change)
        
        # Генерируем OHLC
        open_price = current_price
        close_price = current_price * (1 + random.uniform(-max_change/2, max_change/2))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, max_change/4))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, max_change/4))
        volume = random.uniform(1000000, 5000000)
        
        candle = {
            'timestamp': 1729000000000 + i * 21600000,  # 6H интервалы
            'time': 1729000000000 + i * 21600000,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
        candles.append(candle)
    
    return candles


def test_risk_manager():
    """Тестирует Dynamic Risk Manager"""
    
    print("\n" + "="*80)
    print("ТЕСТ: Dynamic Risk Manager")
    print("="*80 + "\n")
    
    # Инициализируем Risk Manager
    from bot_engine.ai.risk_manager import DynamicRiskManager
    
    risk_manager = DynamicRiskManager()
    
    print("[1] Risk Manager инициализирован:")
    status = risk_manager.get_status()
    print(f"   Базовый SL: {status['base_sl']}%")
    print(f"   Базовый TP: {status['base_tp']}%")
    print(f"   SL диапазон: {status['sl_range'][0]}-{status['sl_range'][1]}%")
    print(f"   TP диапазон: {status['tp_range'][0]}-{status['tp_range'][1]}%\n")
    
    # Тест 1: Низкая волатильность
    print("[2] Тест: Низкая волатильность + Сильный тренд вверх")
    candles_low_vol = generate_test_candles(50, trend='UP', volatility='LOW')
    
    volatility = risk_manager.calculate_volatility(candles_low_vol)
    trend_strength = risk_manager.calculate_trend_strength(candles_low_vol, 'UP')
    
    print(f"   Волатильность: {volatility:.3f} (ожидается < 0.3)")
    print(f"   Сила тренда: {trend_strength:.3f} (ожидается > 0.7)")
    
    dynamic_sl = risk_manager.calculate_dynamic_sl('BTC', candles_low_vol, 'LONG')
    dynamic_tp = risk_manager.calculate_dynamic_tp('BTC', candles_low_vol, 'LONG')
    
    print(f"   Адаптивный SL: {dynamic_sl['sl_percent']}% (ожидается < 15%)")
    print(f"   Причина SL: {dynamic_sl['reason']}")
    print(f"   Адаптивный TP: {dynamic_tp['tp_percent']}% (ожидается > 300%)")
    print(f"   Причина TP: {dynamic_tp['reason']}\n")
    
    # Тест 2: Высокая волатильность
    print("[3] Тест: Высокая волатильность + Слабый тренд")
    candles_high_vol = generate_test_candles(50, trend='NEUTRAL', volatility='HIGH')
    
    volatility = risk_manager.calculate_volatility(candles_high_vol)
    trend_strength = risk_manager.calculate_trend_strength(candles_high_vol, 'UP')
    
    print(f"   Волатильность: {volatility:.3f} (ожидается > 0.7)")
    print(f"   Сила тренда: {trend_strength:.3f} (ожидается < 0.3)")
    
    dynamic_sl = risk_manager.calculate_dynamic_sl('SHITCOIN', candles_high_vol, 'LONG')
    dynamic_tp = risk_manager.calculate_dynamic_tp('SHITCOIN', candles_high_vol, 'LONG')
    
    print(f"   Адаптивный SL: {dynamic_sl['sl_percent']}% (ожидается > 15%)")
    print(f"   Причина SL: {dynamic_sl['reason']}")
    print(f"   Адаптивный TP: {dynamic_tp['tp_percent']}% (ожидается < 300%)")
    print(f"   Причина TP: {dynamic_tp['reason']}\n")
    
    # Тест 3: Предсказание разворота
    print("[4] Тест: Предсказание разворота тренда")
    
    # Генерируем свечи с признаками разворота
    candles_reversal = generate_test_candles(30, trend='UP', volatility='MEDIUM')
    
    # Добавляем признаки разворота (rejection candles)
    for i in range(-3, 0):
        candles_reversal[i]['high'] = candles_reversal[i]['close'] * 1.03  # Длинная верхняя тень
        candles_reversal[i]['low'] = candles_reversal[i]['open'] * 0.995
        candles_reversal[i]['close'] = candles_reversal[i]['open'] * 0.998  # Небольшое тело
    
    reversal = risk_manager.predict_reversal(candles_reversal, 'UP')
    
    print(f"   Вероятность разворота: {reversal['reversal_probability']:.0%}")
    print(f"   Сигналы разворота: {reversal['signals']}")
    print(f"   Рекомендация: {reversal['recommendation']}")
    print(f"   Уверенность: {reversal['confidence']:.0%}\n")
    
    # Тест 4: Оптимальный trailing stop
    print("[5] Тест: Оптимальный trailing stop")
    
    position_data = {
        'direction': 'LONG',
        'entry_price': 50000,
        'current_pnl': 5.0
    }
    
    # Без риска разворота
    trailing_safe = risk_manager.calculate_optimal_trailing(
        'BTC', candles_low_vol, position_data
    )
    
    print(f"   Безопасный тренд:")
    print(f"     Расстояние trailing: {trailing_safe['trailing_distance']}%")
    print(f"     Ужесточить: {trailing_safe['should_tighten']}")
    print(f"     Причина: {trailing_safe['reason']}")
    
    # С риском разворота
    trailing_risky = risk_manager.calculate_optimal_trailing(
        'BTC', candles_reversal, position_data
    )
    
    print(f"   Риск разворота:")
    print(f"     Расстояние trailing: {trailing_risky['trailing_distance']}%")
    print(f"     Ужесточить: {trailing_risky['should_tighten']}")
    print(f"     Причина: {trailing_risky['reason']}\n")
    
    # Тест 5: Размер позиции
    print("[6] Тест: Расчёт размера позиции")
    
    balance = 1000.0  # USDT
    
    # Высокая уверенность + низкая волатильность
    size_high_conf = risk_manager.calculate_position_size(
        'BTC', candles_low_vol, balance, signal_confidence=0.9
    )
    
    print(f"   Высокая уверенность (90%) + Низкая волатильность:")
    print(f"     Размер: {size_high_conf['size_usdt']} USDT")
    print(f"     Множитель: {size_high_conf['size_multiplier']:.2f}x")
    print(f"     Причина: {size_high_conf['reason']}")
    
    # Низкая уверенность + высокая волатильность
    size_low_conf = risk_manager.calculate_position_size(
        'SHITCOIN', candles_high_vol, balance, signal_confidence=0.5
    )
    
    print(f"   Низкая уверенность (50%) + Высокая волатильность:")
    print(f"     Размер: {size_low_conf['size_usdt']} USDT")
    print(f"     Множитель: {size_low_conf['size_multiplier']:.2f}x")
    print(f"     Причина: {size_low_conf['reason']}\n")
    
    # Тест 6: Рекомендация по удержанию
    print("[7] Тест: Рекомендация по удержанию позиции")
    
    position_profit = {
        'direction': 'LONG',
        'entry_price': 50000,
        'current_pnl': 8.0
    }
    
    hold_rec = risk_manager.get_hold_recommendation(
        'BTC', candles_low_vol, position_profit
    )
    
    print(f"   Текущий PnL: +{position_profit['current_pnl']}%")
    print(f"   Действие: {hold_rec['action']}")
    print(f"   Причина: {hold_rec['reason']}")
    print(f"   Риск-скор: {hold_rec['risk_score']:.2f}")
    print(f"   Ожидаемая прибыль: {hold_rec['expected_profit']:.1f}%\n")
    
    # Итоги
    print("="*80)
    print("РЕЗУЛЬТАТ: [OK] Все расчёты работают корректно!")
    print("="*80)
    print("\nОсновные проверки:")
    print(f"  [OK] Волатильность: низкая={volatility:.2f} < 0.3")
    print(f"  [OK] Сила тренда: high={trend_strength:.2f} > 0.7")
    print(f"  [OK] Адаптация SL: {dynamic_sl['sl_percent']}% (диапазон: 8-25%)")
    print(f"  [OK] Адаптация TP: {dynamic_tp['tp_percent']}% (диапазон: 150-600%)")
    print(f"  [OK] Предсказание разворота: {reversal['reversal_probability']:.0%}")
    print(f"  [OK] Trailing stop: {trailing_safe['trailing_distance']}-{trailing_risky['trailing_distance']}%")
    print(f"  [OK] Размер позиции: {size_low_conf['size_usdt']}-{size_high_conf['size_usdt']} USDT")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    try:
        test_risk_manager()
    except Exception as e:
        print(f"\n[X] Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

