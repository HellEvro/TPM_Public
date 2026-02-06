#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки и анализа AI Database

Показывает:
- Статистику по всем таблицам
- Количество записей
- Размер базы данных
- Примеры данных
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_engine.ai.ai_database import get_ai_database


def main():
    """Основная функция"""
    print("=" * 80)
    print("ПРОВЕРКА AI DATABASE")
    print("=" * 80)
    
    try:
        db = get_ai_database()
        
        # Получаем общую статистику
        stats = db.get_database_stats()
        
        print("\n📊 ОБЩАЯ СТАТИСТИКА:")
        print("-" * 80)
        print(f"Размер базы данных: {stats.get('database_size_mb', 0):.2f} MB")
        print(f"Уникальных символов (симуляции): {stats.get('unique_symbols_simulated', 0)}")
        print(f"Уникальных символов (реальные): {stats.get('unique_symbols_real', 0)}")
        
        print("\n📈 КОЛИЧЕСТВО ЗАПИСЕЙ:")
        print("-" * 80)
        print(f"Симулированных сделок: {stats.get('simulated_trades_count', 0):,}")
        print(f"Реальных сделок ботов: {stats.get('bot_trades_count', 0):,}")
        print(f"Сделок биржи: {stats.get('exchange_trades_count', 0):,}")
        print(f"Решений AI: {stats.get('ai_decisions_count', 0):,}")
        print(f"Сессий обучения: {stats.get('training_sessions_count', 0):,}")
        
        # Получаем примеры симуляций
        print("\n🎮 ПРИМЕРЫ СИМУЛЯЦИЙ:")
        print("-" * 80)
        sim_trades = db.get_simulated_trades(limit=5)
        if sim_trades:
            for i, trade in enumerate(sim_trades, 1):
                print(f"{i}. {trade.get('symbol')} {trade.get('direction')} | "
                      f"PnL: {trade.get('pnl', 0):.4f} | "
                      f"Успешна: {'Да' if trade.get('is_successful') else 'Нет'}")
        else:
            print("Нет симуляций")
        
        # Получаем примеры реальных сделок
        print("\n🤖 ПРИМЕРЫ РЕАЛЬНЫХ СДЕЛОК БОТОВ:")
        print("-" * 80)
        bot_trades = db.get_bot_trades(limit=5)
        if bot_trades:
            for i, trade in enumerate(bot_trades, 1):
                print(f"{i}. {trade.get('symbol')} {trade.get('direction')} | "
                      f"PnL: {trade.get('pnl', 0):.4f} | "
                      f"Источник: {trade.get('decision_source', 'N/A')}")
        else:
            print("Нет реальных сделок")
        
        # Получаем статистику сессий обучения
        print("\n🎓 ПОСЛЕДНИЕ СЕССИИ ОБУЧЕНИЯ:")
        print("-" * 80)
        sessions = db.get_training_statistics(limit=5)
        if sessions:
            for i, session in enumerate(sessions, 1):
                print(f"{i}. Тип: {session.get('session_type')} | "
                      f"Статус: {session.get('status')} | "
                      f"Монет: {session.get('coins_processed', 0)} | "
                      f"Win Rate: {session.get('win_rate', 0):.2f}%")
        else:
            print("Нет сессий обучения")
        
        # Сравнение симуляций и реальных сделок
        print("\n📊 СРАВНЕНИЕ СИМУЛЯЦИЙ И РЕАЛЬНЫХ СДЕЛОК:")
        print("-" * 80)
        comparison = db.compare_simulated_vs_real()
        sim_stats = comparison.get('simulated', {})
        real_stats = comparison.get('real', {})
        comp = comparison.get('comparison', {})
        
        sim_avg_pnl = sim_stats.get('avg_pnl') or 0
        sim_win_rate = sim_stats.get('win_rate') or 0
        real_avg_pnl = real_stats.get('avg_pnl') or 0
        pnl_diff = comp.get('pnl_diff') or 0
        
        print(f"Симуляции: {sim_stats.get('count', 0):,} сделок, "
              f"Средний PnL: {sim_avg_pnl:.4f}, "
              f"Win Rate: {sim_win_rate:.2f}%")
        print(f"Реальные: {real_stats.get('count', 0):,} сделок, "
              f"Средний PnL: {real_avg_pnl:.4f}")
        print(f"Разница PnL: {pnl_diff:.4f}")
        
        print("\n" + "=" * 80)
        print("✅ Проверка завершена")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

