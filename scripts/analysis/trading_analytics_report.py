#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт полной аналитики торговли.

Анализирует все сделки на бирже и сделки ботов из БД, строит отчёт:
- Сверка биржа vs bot_trades_history
- Win Rate, PnL, просадка, серии убытков
- Разбивка по причинам закрытия, символам, ботам
- Потенциальные ошибки и рекомендации для AI

Запуск:
    python scripts/analysis/trading_analytics_report.py
    python scripts/analysis/trading_analytics_report.py --exchange   # с загрузкой сделок с биржи
    python scripts/analysis/trading_analytics_report.py --output report.json --ai-summary
"""

import argparse
import json
import sys
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Кодировка вывода для Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_exchange():
    """Загружает биржу из app.config (как в rebuild_bot_history_from_exchange)."""
    try:
        from app.config import EXCHANGES, ACTIVE_EXCHANGE  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Не удалось импортировать app.config. Убедитесь, что config.py существует."
        ) from exc
    exchange_name = ACTIVE_EXCHANGE
    exchange_cfg = EXCHANGES.get(exchange_name, {})
    if not exchange_cfg or not exchange_cfg.get("enabled", True):
        raise RuntimeError(f"Для {exchange_name} нет активных API ключей в configs/keys.py (или configs/app_config.py).")
    api_key = exchange_cfg.get("api_key")
    api_secret = exchange_cfg.get("api_secret")
    passphrase = exchange_cfg.get("passphrase")
    if not api_key or not api_secret:
        raise RuntimeError(f"API ключи для {exchange_name} не заполнены.")
    from exchanges.exchange_factory import ExchangeFactory
    exchange = ExchangeFactory.create_exchange(exchange_name, api_key, api_secret, passphrase)
    return exchange, exchange_name


def print_report(report: dict) -> None:
    """Печатает человекочитаемый отчёт в консоль."""
    summary = report.get("summary", {})
    bot = report.get("bot_analytics", {})
    recon = report.get("reconciliation", {})
    ex = report.get("exchange_analytics", {})

    print("=" * 70)
    print("ПОЛНЫЙ АНАЛИТИЧЕСКИЙ ОТЧЁТ ТОРГОВЛИ")
    print("=" * 70)
    print(f"Сформирован: {report.get('generated_at', '')}")
    print()

    print("--- СВОДКА ---")
    print(f"  Сделок на бирже (загружено):     {summary.get('exchange_trades_count', 0)}")
    print(f"  Сделок ботов в БД:               {summary.get('bot_trades_count', 0)}")
    print(f"  Совпадений (биржа = бот):        {summary.get('reconciliation_matched', 0)}")
    print(f"  Только на бирже:                 {summary.get('reconciliation_only_exchange', 0)}")
    print(f"  Только в истории ботов:          {summary.get('reconciliation_only_bots', 0)}")
    print(f"  Расхождений PnL:                 {summary.get('reconciliation_pnl_mismatches', 0)}")
    print()

    if bot:
        print("--- АНАЛИТИКА СДЕЛОК БОТОВ ---")
        print(f"  Всего сделок:                  {bot.get('total_trades', 0)}")
        print(f"  Общий PnL (USDT):              {bot.get('total_pnl_usdt', 0)}")
        print(f"  Win Rate (%):                  {bot.get('win_rate_pct', 0)}")
        print(f"  Прибыльных:                    {bot.get('win_count', 0)}")
        print(f"  Убыточных:                     {bot.get('loss_count', 0)}")
        print(f"  Средняя прибыль (USDT):        {bot.get('avg_win_usdt', 0)}")
        print(f"  Средний убыток (USDT):        {bot.get('avg_loss_usdt', 0)}")
        series = bot.get("consecutive_series", {})
        print(f"  Макс. серия побед подряд:     {series.get('max_consecutive_wins', 0)}")
        print(f"  Макс. серия убытков подряд:   {series.get('max_consecutive_losses', 0)}")
        dd = bot.get("drawdown", {})
        print(f"  Макс. просадка (USDT):        {dd.get('max_drawdown_usdt', 0)}")
        print(f"  Макс. просадка (%):            {dd.get('max_drawdown_pct', 0)}")
        print(f"  Потенциальных ошибок:          {bot.get('possible_errors_count', 0)}")
        print()

        by_reason = bot.get("by_close_reason", {})
        if by_reason:
            print("--- ПО ПРИЧИНАМ ЗАКРЫТИЯ ---")
            for reason, data in sorted(by_reason.items(), key=lambda x: -x[1].get("count", 0)):
                print(f"  {reason}: count={data.get('count', 0)}, pnl={data.get('pnl', 0):.2f}, wins={data.get('wins', 0)}, losses={data.get('losses', 0)}")
            print()

        by_src = bot.get("by_decision_source", {})
        if by_src:
            print("--- ПО ИСТОЧНИКУ РЕШЕНИЯ ---")
            for src, data in sorted(by_src.items(), key=lambda x: -x[1].get("count", 0)):
                print(f"  {src}: count={data.get('count', 0)}, pnl={data.get('pnl', 0):.2f}")
            print()

        # Неудачные монеты
        uc_list = bot.get("unsuccessful_coins", [])
        if uc_list:
            print("--- НЕУДАЧНЫЕ МОНЕТЫ ---")
            print(f"  (монеты с PnL < 0 или Win Rate < 45%, минимум 3 сделки)")
            for uc in uc_list:
                print(f"  {uc.get('symbol')}: сделок={uc.get('trades_count')}, PnL={uc.get('pnl_usdt')} USDT, Win Rate={uc.get('win_rate_pct')}%, причины={uc.get('reasons', [])}")
            print()

        # Неудачные настройки по RSI и тренду для этих монет
        us_list = bot.get("unsuccessful_settings", [])
        if us_list:
            print("--- НЕУДАЧНЫЕ НАСТРОЙКИ (по RSI и тренду) ---")
            for us in us_list:
                sym = us.get("symbol", "")
                bad_rsi = us.get("bad_rsi_ranges", [])
                bad_trends = us.get("bad_trends", [])
                if not bad_rsi and not bad_trends:
                    continue
                print(f"  Монета {sym}:")
                if bad_rsi:
                    for r in bad_rsi:
                        print(f"    RSI {r.get('rsi_range')}: сделок={r.get('trades_count')}, PnL={r.get('pnl_usdt')}, Win Rate={r.get('win_rate_pct')}%")
                if bad_trends:
                    for tr in bad_trends:
                        print(f"    Тренд {tr.get('trend')}: сделок={tr.get('trades_count')}, PnL={tr.get('pnl_usdt')}, Win Rate={tr.get('win_rate_pct')}%")
            print()

    if ex:
        print("--- АНАЛИТИКА СДЕЛОК С БИРЖИ ---")
        print(f"  Всего: {ex.get('total_trades', 0)}, PnL: {ex.get('total_pnl_usdt', 0)} USDT, Win Rate: {ex.get('win_rate_pct', 0)}%")
        print()

    if recon and (recon.get("only_on_exchange_count", 0) > 0 or recon.get("only_in_bots_count", 0) > 0):
        print("--- СВЕРКА: ВНИМАНИЕ ---")
        if recon.get("only_on_exchange_count", 0) > 0:
            print(f"  Сделки только на бирже (нет в истории ботов): {recon['only_on_exchange_count']}")
            for item in (recon.get("only_on_exchange") or [])[:5]:
                print(f"    - {item.get('symbol')} exit_ts={item.get('exit_timestamp')} pnl={item.get('pnl')}")
        if recon.get("only_in_bots_count", 0) > 0:
            print(f"  Сделки только в истории ботов (нет на бирже): {recon['only_in_bots_count']}")
            for item in (recon.get("only_in_bots") or [])[:5]:
                print(f"    - {item.get('symbol')} bot_id={item.get('bot_id')} pnl={item.get('pnl')}")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Полная аналитика торговли: биржа + боты, сверка, метрики, ошибки"
    )
    parser.add_argument(
        "--exchange",
        action="store_true",
        help="Загружать сделки с биржи (get_closed_pnl). Требует app.config и API ключи.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="all",
        help="Период для биржи: all, day, week, month (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Макс. сделок ботов из БД (default: 50000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь к файлу для сохранения JSON отчёта",
    )
    parser.add_argument(
        "--ai-summary",
        action="store_true",
        help="Дополнительно сохранить/вывести блок для AI (проблемы и рекомендации)",
    )
    args = parser.parse_args()

    exchange_instance = None
    if args.exchange:
        try:
            exchange_instance, _ = load_exchange()
            print("✅ Биржа загружена, запрашиваю историю закрытых позиций...")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить биржу: {e}. Аналитика будет только по БД ботов.")
    else:
        print("ℹ️ Без --exchange сделки с биржи не загружаются. Сверка будет пустой.")

    from bot_engine.trading_analytics import run_full_analytics, get_analytics_for_ai

    report = run_full_analytics(
        load_bot_trades_from_db=True,
        load_exchange_from_api=args.exchange,
        exchange_instance=exchange_instance,
        exchange_period=args.period,
        bots_db_limit=args.limit,
    )

    print_report(report)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON отчёт сохранён: {out_path}")

    if args.ai_summary:
        ai_block = get_analytics_for_ai(report)
        print()
        print("--- ДЛЯ AI МОДУЛЯ ---")
        print("Проблемы:", ai_block.get("problems", []))
        print("Рекомендации:", ai_block.get("recommendations", []))
        print("Метрики:", json.dumps(ai_block.get("metrics", {}), ensure_ascii=False, indent=2))
        if args.output:
            ai_path = Path(args.output).with_name(
                Path(args.output).stem + "_ai_summary.json"
            )
            if not ai_path.is_absolute():
                ai_path = PROJECT_ROOT / ai_path
            with open(ai_path, "w", encoding="utf-8") as f:
                json.dump(ai_block, f, ensure_ascii=False, indent=2)
            print(f"✅ AI summary сохранён: {ai_path}")


if __name__ == "__main__":
    main()
