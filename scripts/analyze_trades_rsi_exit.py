#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит сделок по RSI: загружает сделки из БД (bot_trades_history) ИЛИ с биржи (get_closed_pnl),
сравнивает момент входа/выхода с порогами RSI из конфига и выявляет расхождения.

Запуск:
    python scripts/analyze_trades_rsi_exit.py
    python scripts/analyze_trades_rsi_exit.py --from-exchange
    python scripts/analyze_trades_rsi_exit.py --symbol 1000XECUSDT --from-exchange
    python scripts/analyze_trades_rsi_exit.py --limit 200 --output report.txt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ts_ms_to_iso(ts_ms):
    if ts_ms is None:
        return None
    try:
        s = float(ts_ms) / 1000.0 if float(ts_ms) > 1e12 else float(ts_ms)
        return datetime.fromtimestamp(s, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _infer_direction(side, entry_price, exit_price, pnl):
    if (side or "").upper() in ("BUY", "LONG"):
        return "LONG"
    if (side or "").upper() in ("SELL", "SHORT"):
        return "SHORT"
    if entry_price and exit_price:
        if exit_price >= entry_price:
            return "LONG" if pnl >= 0 else "SHORT"
        return "SHORT" if pnl >= 0 else "LONG"
    return "LONG" if (pnl or 0) >= 0 else "SHORT"


def load_trades_from_exchange(symbol_filter=None, limit=None, period="all"):
    """Загружает закрытые сделки с биржи через get_closed_pnl. Требует configs/keys.py и configs/app_config."""
    try:
        from app.config import EXCHANGES, ACTIVE_EXCHANGE
    except ImportError:
        try:
            from configs.app_config import EXCHANGES, ACTIVE_EXCHANGE
        except ImportError:
            raise RuntimeError("Нужен app.config или configs.app_config с EXCHANGES, ACTIVE_EXCHANGE")
    exchange_name = ACTIVE_EXCHANGE
    cfg = EXCHANGES.get(exchange_name, {})
    if not cfg or not cfg.get("enabled", True):
        raise RuntimeError(f"Биржа {exchange_name} не включена в конфиге")
    api_key = cfg.get("api_key")
    api_secret = cfg.get("api_secret")
    passphrase = cfg.get("passphrase")
    if not api_key or not api_secret:
        raise RuntimeError("В configs/keys.py (или app_config) не заполнены API ключи биржи")
    from exchanges.exchange_factory import ExchangeFactory
    exchange = ExchangeFactory.create_exchange(exchange_name, api_key, api_secret, passphrase)
    raw = exchange.get_closed_pnl(sort_by="time", period=period) or []
    trades = []
    for r in raw:
        sym = r.get("symbol") or ""
        if symbol_filter and sym != symbol_filter:
            continue
        entry_price = float(r.get("entry_price") or 0) or 0.0
        exit_price = float(r.get("exit_price") or 0) or 0.0
        pnl = float(r.get("closed_pnl") or r.get("closedPnl") or 0) or 0.0
        close_ts = r.get("close_timestamp") or r.get("closeTime") or 0
        entry_ts = r.get("created_timestamp") or r.get("createdTime") or close_ts
        if close_ts and close_ts < 1e12:
            close_ts = int(close_ts) * 1000
        if entry_ts and entry_ts < 1e12:
            entry_ts = int(entry_ts) * 1000
        direction = _infer_direction(r.get("side"), entry_price, exit_price, pnl)
        trades.append({
            "symbol": sym,
            "direction": direction,
            "entry_time": _ts_ms_to_iso(entry_ts) or r.get("created_time", ""),
            "exit_time": _ts_ms_to_iso(close_ts) or r.get("close_time", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_rsi": None,
            "exit_rsi": None,
            "entry_trend": "NEUTRAL",
            "close_reason": "EXCHANGE",
            "pnl": pnl,
            "source": "exchange",
        })
    trades.sort(key=lambda x: (x.get("exit_time") or ""), reverse=True)
    if limit:
        trades = trades[:limit]
    return trades


def main():
    parser = argparse.ArgumentParser(description="Аудит сделок: вход/выход vs RSI и конфиг")
    parser.add_argument("--symbol", type=str, default=None, help="Фильтр по символу (например 1000XECUSDT)")
    parser.add_argument("--limit", type=int, default=None, help="Макс. число сделок (по умолчанию все)")
    parser.add_argument("--output", type=str, default=None, help="Файл для отчёта (иначе stdout)")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод по каждой сделке")
    parser.add_argument("--from-exchange", action="store_true", help="Брать сделки с биржи (get_closed_pnl), а не из БД")
    parser.add_argument("--period", type=str, default="all", help="Период для биржи: all, day, week, month (при --from-exchange)")
    args = parser.parse_args()

    # Загрузка конфига
    from bot_engine.config_loader import get_current_timeframe, reload_config
    try:
        reload_config()
    except Exception:
        pass
    timeframe = get_current_timeframe() or "1m"

    # Пороги выхода из конфига (как в bot_class / filters)
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            auto_config = bots_data.get("auto_bot_config", {})
    except Exception:
        auto_config = {}
    if not auto_config:
        try:
            from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG
            auto_config = DEFAULT_AUTO_BOT_CONFIG or {}
        except Exception:
            auto_config = {}
    exit_long_with = auto_config.get("rsi_exit_long_with_trend") or 65
    exit_long_against = auto_config.get("rsi_exit_long_against_trend") or 60
    exit_short_with = auto_config.get("rsi_exit_short_with_trend") or 35
    exit_short_against = auto_config.get("rsi_exit_short_against_trend") or 40

    # Источник сделок: биржа или БД
    if args.from_exchange:
        try:
            trades = load_trades_from_exchange(
                symbol_filter=args.symbol,
                limit=args.limit,
                period=args.period,
            )
            source_note = "с биржи (get_closed_pnl)"
        except Exception as e:
            print(f"Ошибка загрузки с биржи: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            symbol=args.symbol,
            status="CLOSED",
            limit=args.limit,
        )
        source_note = "из БД (bot_trades_history)"

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    def w(line=""):
        out.write(line + "\n")
        if out != sys.stdout:
            print(line)

    w("=" * 80)
    w("АУДИТ СДЕЛОК: ВХОД/ВЫХОД vs RSI И КОНФИГ")
    w("=" * 80)
    w(f"Источник сделок: {source_note}")
    w(f"Таймфрейм из конфига: {timeframe}")
    w(f"Пороги выхода LONG: with_trend >={exit_long_with}, against_trend >={exit_long_against}")
    w(f"Пороги выхода SHORT: with_trend <={exit_short_with}, against_trend <={exit_short_against}")
    w(f"Всего закрытых сделок: {len(trades)}")
    w("")

    errors_no_exit_rsi = 0
    errors_should_close_earlier = 0
    ok_exit_by_rsi = 0
    other_close = 0

    for i, t in enumerate(trades):
        symbol = t.get("symbol", "")
        direction = (t.get("direction") or "LONG").upper()
        entry_time = t.get("entry_time") or ""
        exit_time = t.get("exit_time") or ""
        entry_price = t.get("entry_price")
        exit_price = t.get("exit_price")
        entry_rsi = t.get("entry_rsi")
        exit_rsi = t.get("exit_rsi")
        entry_trend = (t.get("entry_trend") or "NEUTRAL").upper()
        close_reason = t.get("close_reason") or ""
        pnl = t.get("pnl")

        if direction == "LONG":
            thr = exit_long_with if entry_trend == "UP" else exit_long_against
            exit_ok_by_rsi = exit_rsi is not None and exit_rsi >= thr
            should_exit_condition = "RSI >= %s" % thr
        else:
            thr = exit_short_with if entry_trend == "DOWN" else exit_short_against
            exit_ok_by_rsi = exit_rsi is not None and exit_rsi <= thr
            should_exit_condition = "RSI <= %s" % thr

        if exit_rsi is None:
            errors_no_exit_rsi += 1
            if t.get("source") == "exchange":
                verdict = "📡 Сделка с биржи — RSI в API биржи не приходит; для проверки по RSI используйте сделку из БД (без --from-exchange)."
            else:
                verdict = "⚠️ В БД НЕТ exit_rsi — при закрытии RSI не был записан (система могла не видеть RSI по таймфрейму)"
        elif exit_ok_by_rsi:
            ok_exit_by_rsi += 1
            verdict = "✅ На выходе RSI соответствовал порогу"
        else:
            other_close += 1
            verdict = f"ℹ️ Закрыто по другой причине (close_reason={close_reason}); на выходе RSI={exit_rsi} (порог: {should_exit_condition})"

        if args.verbose or exit_rsi is None or (direction == "LONG" and exit_rsi is not None and exit_rsi < thr) or (direction == "SHORT" and exit_rsi is not None and exit_rsi > thr):
            w(f"--- Сделка #{i+1} ---")
            w(f"  Символ: {symbol}  Направление: {direction}  Тренд входа: {entry_trend}")
            w(f"  Вход:  {entry_time}  цена={entry_price}  RSI={entry_rsi}")
            w(f"  Выход: {exit_time}  цена={exit_price}  RSI={exit_rsi}  PnL={pnl}  причина={close_reason}")
            w(f"  {verdict}")
            w("")

    w("=" * 80)
    w("ИТОГ")
    w("=" * 80)
    w(f"Сделок без exit_rsi в БД (невозможно проверить): {errors_no_exit_rsi}")
    w(f"Сделок с выходом по RSI в пороге: {ok_exit_by_rsi}")
    w(f"Сделок закрыто по другим причинам: {other_close}")
    w("")
    if errors_no_exit_rsi > 0:
        w("Рекомендация: при закрытии позиции передавать и сохранять exit_rsi (и entry_timeframe) в save_bot_trade_history.")
        w("Проверить: обновляется ли coins_rsi_data по таймфрейму бота (1m) с интервалом RSI_UPDATE_INTERVAL.")
    if args.output:
        out.close()
        print(f"Отчёт записан в {args.output}")


if __name__ == "__main__":
    main()
