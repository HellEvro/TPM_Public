#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рекомендации по фильтрам на основе результатов торговли.

Запускает аналитику сделок и выводит:
- Краткую сводку (PnL, Win Rate, причины закрытия)
- Конкретные рекомендации по ужесточению фильтров
- Готовый чёрный список монет для вставки в configs/bot_config.py

Запуск:
  python scripts/analysis/filter_recommendations.py
  python scripts/analysis/filter_recommendations.py --limit 10000
  python scripts/analysis/filter_recommendations.py --blacklist-top 30  # топ-30 убыточных в blacklist
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Рекомендации по фильтрам по результатам торговли")
    parser.add_argument("--limit", type=int, default=5000, help="Макс. сделок из БД (default: 5000)")
    parser.add_argument("--blacklist-top", type=int, default=0,
                        help="Сколько неудачных монет вывести в виде BLACKLIST для конфига (0 = не выводить)")
    args = parser.parse_args()

    from bot_engine.trading_analytics import run_full_analytics

    report = run_full_analytics(
        load_bot_trades_from_db=True,
        load_exchange_from_api=False,
        bots_db_limit=args.limit,
    )

    bot = report.get("bot_analytics") or {}
    total_trades = bot.get("total_trades", 0)
    total_pnl = bot.get("total_pnl_usdt", 0)
    win_rate = bot.get("win_rate_pct", 0)
    win_count = bot.get("win_count", 0)
    loss_count = bot.get("loss_count", 0)
    by_reason = bot.get("by_close_reason") or {}
    unsuccessful = bot.get("unsuccessful_coins", [])
    series = bot.get("consecutive_series") or {}
    drawdown = bot.get("drawdown") or {}

    # --- Краткая сводка ---
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ ТОРГОВЛИ И РЕКОМЕНДАЦИИ ПО ФИЛЬТРАМ")
    print("=" * 70)
    print(f"  Сделок:        {total_trades}")
    print(f"  Общий PnL:     {total_pnl:.2f} USDT")
    print(f"  Win Rate:      {win_rate:.1f}%  (прибыльных: {win_count}, убыточных: {loss_count})")
    print(f"  Макс. серия убытков: {series.get('max_consecutive_losses', 0)}")
    print(f"  Макс. просадка:      {drawdown.get('max_drawdown_usdt', 0):.2f} USDT  ({drawdown.get('max_drawdown_pct', 0):.1f}%)")
    print()

    # Причины закрытия
    print("--- По причинам закрытия ---")
    for reason, data in sorted(by_reason.items(), key=lambda x: -x[1].get("count", 0)):
        r = data.get("count", 0), data.get("pnl", 0), data.get("wins", 0), data.get("losses", 0)
        print(f"  {reason}: count={r[0]}, pnl={r[1]:.2f}, wins={r[2]}, losses={r[3]}")
    print()

    # --- Рекомендации ---
    print("=" * 70)
    print("РЕКОМЕНДАЦИИ: УЖЕСТОЧИТЬ ФИЛЬТРЫ, ЧТОБЫ СНИЗИТЬ УБЫТКИ")
    print("=" * 70)
    print()
    print("1. ExitScam (резкие движения цены)")
    print("   Сейчас в example: EXIT_SCAM_SINGLE_CANDLE_PERCENT = 15, EXIT_SCAM_MULTI_CANDLE_PERCENT = 50")
    print("   Рекомендация: уменьшить до 10 и 40 — меньше пропускать волатильные моменты.")
    print()
    print("2. AI Anomaly (порог блокировки аномалий)")
    print("   Сейчас: ANOMALY_BLOCK_THRESHOLD = 0.7 (70%)")
    print("   Рекомендация: понизить до 0.5–0.6 — чаще блокировать сомнительные входы.")
    print()
    print("3. RSI зоны входа")
    print("   Сейчас: RSI_LONG_THRESHOLD = 29, RSI_SHORT_THRESHOLD = 71")
    print("   Рекомендация: сузить — например LONG <= 25, SHORT >= 75 (входить только в более экстремальных зонах).")
    print()
    print("4. Стоп-лосс")
    print("   Убедитесь, что max_loss_percent не слишком большой (2–3% на сделку снижают просадку).")
    print()
    print("5. Защита от повторного входа после убытка")
    print("   Включите LOSS_REENTRY_PROTECTION = True и при необходимости увеличьте LOSS_REENTRY_CANDLES.")
    print()
    print("6. Чёрный список")
    print("   Добавьте в BLACKLIST монеты с стабильно отрицательным PnL (см. ниже --blacklist-top).")
    print()

    if unsuccessful:
        print("--- Неудачные монеты (топ по убыткам) ---")
        for u in unsuccessful[:25]:
            print(f"  {u.get('symbol')}: PnL={u.get('pnl_usdt')} USDT, Win Rate={u.get('win_rate_pct')}%, сделок={u.get('trades_count')}")
        print()

    if args.blacklist_top > 0 and unsuccessful:
        symbols = [u["symbol"] for u in unsuccessful[: args.blacklist_top]]
        print("--- Готовый BLACKLIST для configs/bot_config.py ---")
        print("   Скопируйте в класс AutoBotConfig / DefaultAutoBotConfig:")
        print()
        print("   BLACKLIST = [")
        for s in symbols:
            print(f"       '{s}',")
        print("   ]")
        print()
        print("   И установите SCOPE = 'blacklist' если хотите торговать ТОЛЬКО по whitelist,")
        print("   или оставьте SCOPE = 'all' — тогда blacklist только исключает эти монеты.")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
