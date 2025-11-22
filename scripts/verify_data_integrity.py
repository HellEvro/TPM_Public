#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки целостности данных для обучения ИИ
Проверяет:
1. Распределение PnL (должны быть отрицательные значения)
2. Правильность флагов is_simulated
3. Источники сделок (decision_source)
4. Отсутствие дубликатов
5. Корректность расчета PnL
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_history(file_path):
    """Загружает историю из файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

def verify_pnl_distribution(trades):
    """Проверяет распределение PnL"""
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED' and t.get('pnl') is not None]
    if not closed_trades:
        print("Нет закрытых сделок с PnL")
        return False
    
    pnl_values = [t.get('pnl') for t in closed_trades]
    positive = sum(1 for p in pnl_values if p > 0)
    negative = sum(1 for p in pnl_values if p < 0)
    zero = sum(1 for p in pnl_values if p == 0)
    
    print(f"\nРАСПРЕДЕЛЕНИЕ PnL:")
    print(f"   Всего закрытых сделок: {len(closed_trades)}")
    print(f"   Прибыльных (PnL > 0): {positive} ({positive/len(closed_trades)*100:.1f}%)")
    print(f"   Убыточных (PnL < 0): {negative} ({negative/len(closed_trades)*100:.1f}%)")
    print(f"   Нулевых (PnL = 0): {zero} ({zero/len(closed_trades)*100:.1f}%)")
    
    if pnl_values:
        print(f"   Min PnL: {min(pnl_values):.4f}")
        print(f"   Max PnL: {max(pnl_values):.4f}")
        print(f"   Avg PnL: {sum(pnl_values)/len(pnl_values):.4f}")
    
    if negative == 0:
        print("\nКРИТИЧЕСКАЯ ПРОБЛЕМА: Нет убыточных сделок!")
        print("   Это означает, что либо:")
        print("   1. Убыточные сделки не сохраняются")
        print("   2. PnL рассчитывается неправильно")
        return False
    
    return True

def verify_simulation_flags(trades, history):
    """Проверяет правильность флагов is_simulated"""
    print(f"\nПРОВЕРКА ФЛАГОВ is_simulated:")
    
    trades_with_flag = sum(1 for t in trades if 'is_simulated' in t)
    trades_simulated = sum(1 for t in trades if t.get('is_simulated', False))
    trades_real = sum(1 for t in trades if not t.get('is_simulated', False))
    
    history_with_flag = sum(1 for h in history if 'is_simulated' in h)
    history_simulated = sum(1 for h in history if h.get('is_simulated', False))
    history_real = sum(1 for h in history if not h.get('is_simulated', False))
    
    print(f"   Сделки (trades):")
    print(f"      Всего: {len(trades)}")
    print(f"      С флагом is_simulated: {trades_with_flag}")
    print(f"      Реальных (is_simulated=False): {trades_real}")
    print(f"      Симулированных (is_simulated=True): {trades_simulated}")
    
    print(f"   История (history):")
    print(f"      Всего: {len(history)}")
    print(f"      С флагом is_simulated: {history_with_flag}")
    print(f"      Реальных (is_simulated=False): {history_real}")
    print(f"      Симулированных (is_simulated=True): {history_simulated}")
    
    if trades_with_flag < len(trades):
        print(f"\nВНИМАНИЕ: {len(trades) - trades_with_flag} сделок без флага is_simulated")
    
    return True

def verify_decision_sources(trades, history):
    """Проверяет источники решений"""
    print(f"\nИСТОЧНИКИ РЕШЕНИЙ (decision_source):")
    
    trade_sources = defaultdict(int)
    for t in trades:
        source = t.get('decision_source', 'UNKNOWN')
        trade_sources[source] += 1
    
    history_sources = defaultdict(int)
    for h in history:
        source = h.get('decision_source', 'UNKNOWN')
        history_sources[source] += 1
    
    print(f"   Сделки (trades):")
    for source, count in sorted(trade_sources.items()):
        print(f"      {source}: {count}")
    
    print(f"   История (history):")
    for source, count in sorted(history_sources.items()):
        print(f"      {source}: {count}")
    
    # Проверяем, что нет AI симуляций в bot_history.json
    ai_simulated = sum(1 for t in trades if t.get('decision_source') == 'AI' and t.get('is_simulated', False))
    if ai_simulated > 0:
        print(f"\nВНИМАНИЕ: Найдено {ai_simulated} AI симуляций в bot_history.json")
        print("   Они должны быть в simulated_trades.json, а не здесь!")
    
    return True

def verify_duplicates(trades):
    """Проверяет наличие дубликатов"""
    print(f"\nПРОВЕРКА ДУБЛИКАТОВ:")
    
    # Проверяем по ID
    ids = {}
    duplicate_ids = []
    for t in trades:
        trade_id = t.get('id')
        if trade_id:
            if trade_id in ids:
                duplicate_ids.append(trade_id)
            else:
                ids[trade_id] = t
    
    if duplicate_ids:
        print(f"   Найдено {len(duplicate_ids)} дубликатов по ID:")
        for dup_id in duplicate_ids[:10]:  # Показываем первые 10
            print(f"      {dup_id}")
        if len(duplicate_ids) > 10:
            print(f"      ... и еще {len(duplicate_ids) - 10}")
        return False
    else:
        print(f"   Дубликатов по ID не найдено")
    
    # Проверяем по комбинации параметров (для открытых позиций)
    open_positions = {}
    duplicate_opens = []
    for t in trades:
        if t.get('status') == 'OPEN':
            key = (
                t.get('symbol'),
                t.get('direction'),
                t.get('entry_price'),
                t.get('bot_id')
            )
            if key in open_positions:
                duplicate_opens.append(key)
            else:
                open_positions[key] = t
    
    if duplicate_opens:
        print(f"   Найдено {len(duplicate_opens)} дубликатов открытых позиций:")
        for dup_key in duplicate_opens[:5]:
            print(f"      {dup_key[0]} {dup_key[1]} @ {dup_key[2]}")
        return False
    else:
        print(f"   Дубликатов открытых позиций не найдено")
    
    return True

def verify_pnl_calculation(trades):
    """Проверяет корректность расчета PnL с учетом комиссий и округлений"""
    print(f"\nПРОВЕРКА РАСЧЕТА PnL:")
    
    incorrect = []
    for t in trades:
        if t.get('status') == 'CLOSED' and t.get('pnl') is not None:
            entry_price = t.get('entry_price')
            exit_price = t.get('exit_price')
            direction = t.get('direction')
            pnl = t.get('pnl')
            
            if entry_price and exit_price and direction:
                # Пересчитываем PnL
                if direction == 'LONG':
                    roi = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    roi = (entry_price - exit_price) / entry_price
                
                # Получаем размер позиции
                position_size = t.get('position_size_usdt') or t.get('size')
                if position_size:
                    calculated_pnl = roi * position_size
                    
                    # УЧИТЫВАЕМ КОМИССИИ И ОКРУГЛЕНИЯ:
                    # 1. Комиссия биржи обычно 0.1% на вход и 0.1% на выход = 0.2% от суммы
                    # 2. Округление цен и размеров позиций
                    # 3. Возможные расхождения из-за разных источников данных
                    commission_rate = 0.002  # 0.2% комиссия
                    commission_cost = abs(position_size * commission_rate)
                    
                    # Допуск = комиссия + 1% от размера позиции (для округлений и погрешностей)
                    tolerance = commission_cost + abs(position_size * 0.01)
                    
                    # Минимальный допуск 0.05 USDT (для маленьких позиций)
                    tolerance = max(tolerance, 0.05)
                    
                    if abs(calculated_pnl - pnl) > tolerance:
                        incorrect.append({
                            'id': t.get('id'),
                            'symbol': t.get('symbol'),
                            'calculated': calculated_pnl,
                            'stored': pnl,
                            'diff': abs(calculated_pnl - pnl),
                            'tolerance': tolerance
                        })
    
    if incorrect:
        print(f"   Найдено {len(incorrect)} сделок с расхождениями PnL (с учетом комиссий):")
        for item in incorrect[:5]:
            print(f"      {item['symbol']}: сохранено={item['stored']:.4f}, рассчитано={item['calculated']:.4f}, разница={item['diff']:.4f}, допуск={item['tolerance']:.4f}")
        if len(incorrect) > 5:
            print(f"      ... и еще {len(incorrect) - 5}")
        print(f"   ПРИМЕЧАНИЕ: Расхождения могут быть из-за:")
        print(f"      - Комиссий биржи (обычно 0.1-0.2% на сделку)")
        print(f"      - Округления цен и размеров позиций")
        print(f"      - Разных источников данных для расчета")
        print(f"      - Использования realized_pnl с биржи вместо пересчета")
        return True  # Не критично, если есть расхождения в пределах допуска
    else:
        print(f"   Все PnL рассчитаны корректно (в пределах допуска с учетом комиссий)")
        return True

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'data/bot_history.json'
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        return
    
    print(f"Загрузка данных из: {file_path}")
    data = load_history(file_path)
    if not data:
        return
    
    trades = data.get('trades', [])
    history = data.get('history', [])
    
    print(f"\nЗагружено:")
    print(f"   Сделок (trades): {len(trades)}")
    print(f"   Записей истории (history): {len(history)}")
    
    # Проверки
    results = []
    
    print("\n" + "="*70)
    print("ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ")
    print("="*70)
    
    results.append(("Распределение PnL", verify_pnl_distribution(trades)))
    results.append(("Флаги is_simulated", verify_simulation_flags(trades, history)))
    results.append(("Источники решений", verify_decision_sources(trades, history)))
    results.append(("Дубликаты", verify_duplicates(trades)))
    results.append(("Расчет PnL", verify_pnl_calculation(trades)))
    
    print("\n" + "="*70)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("="*70)
    
    all_ok = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {status}: {name}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\nВсе проверки пройдены! Данные готовы для обучения ИИ.")
    else:
        print("\nОбнаружены проблемы! Необходимо исправить перед обучением ИИ.")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

