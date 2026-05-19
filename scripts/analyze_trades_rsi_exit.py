#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит сделок по RSI: сравнение входа/выхода с порогами конфига.
Классификация: bot_matched (есть в bot_trades_history), bot_no_db, manual_or_external.

Запуск:
    python scripts/analyze_trades_rsi_exit.py --from-exchange
    python scripts/analyze_trades_rsi_exit.py --from-exchange --output data/rsi_mismatch_report.txt
    python scripts/analyze_trades_rsi_exit.py --limit 500
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


def _load_config_thresholds():
    from bot_engine.config_loader import get_config_value, get_current_timeframe, reload_config
    try:
        reload_config()
    except Exception:
        pass
    timeframe = get_current_timeframe() or "1m"
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
    rsi_long = get_config_value(auto_config, "rsi_long_threshold") if auto_config else 29
    rsi_short = get_config_value(auto_config, "rsi_short_threshold") if auto_config else 71
    return timeframe, auto_config, rsi_long, rsi_short


def _entry_ok_by_threshold(direction, entry_rsi, rsi_long, rsi_short):
    if entry_rsi is None:
        return None
    if direction == "LONG":
        return entry_rsi <= rsi_long
    return entry_rsi >= rsi_short


def _build_db_index(db_trades):
    """Индекс сделок из БД: (symbol, entry_time_iso_prefix) -> trade dict."""
    index = {}
    for t in db_trades or []:
        sym = t.get("symbol") or ""
        et = (t.get("entry_time") or "")[:19]
        key = (sym, et)
        index[key] = t
    return index


def load_trades_from_exchange(symbol_filter=None, limit=None, period="all"):
    """Загружает закрытые сделки с биржи через get_closed_pnl."""
    try:
        from app.config import EXCHANGES, ACTIVE_EXCHANGE
    except ImportError:
        from configs.app_config import EXCHANGES, ACTIVE_EXCHANGE
    exchange_name = ACTIVE_EXCHANGE
    cfg = EXCHANGES.get(exchange_name, {})
    if not cfg or not cfg.get("enabled", True):
        raise RuntimeError(f"Биржа {exchange_name} не включена в конфиге")
    api_key = cfg.get("api_key")
    api_secret = cfg.get("api_secret")
    passphrase = cfg.get("passphrase")
    if not api_key or not api_secret:
        raise RuntimeError("В configs не заполнены API ключи биржи")
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
            "entry_timeframe": None,
            "close_reason": "EXCHANGE",
            "pnl": pnl,
            "source": "exchange",
            "trade_class": None,
        })
    trades.sort(key=lambda x: (x.get("exit_time") or ""), reverse=True)
    if limit:
        trades = trades[:limit]
    return trades


def classify_trades(trades, db_index, rsi_long, rsi_short):
    stats = {
        "bot_matched_ok": 0,
        "bot_matched_bad_rsi": 0,
        "manual_or_external": 0,
        "no_entry_rsi": 0,
    }
    for t in trades:
        sym = t.get("symbol", "")
        et = (t.get("entry_time") or "")[:19]
        db_row = db_index.get((sym, et))
        direction = (t.get("direction") or "LONG").upper()
        if db_row:
            t["trade_class"] = "bot_matched"
            if t.get("entry_rsi") is None:
                t["entry_rsi"] = db_row.get("entry_rsi")
            t["entry_timeframe"] = db_row.get("entry_timeframe") or t.get("entry_timeframe")
            t["close_reason"] = db_row.get("close_reason") or t.get("close_reason")
        else:
            t["trade_class"] = "manual_or_external"
            stats["manual_or_external"] += 1
            if t.get("entry_rsi") is None:
                stats["no_entry_rsi"] += 1
            continue
        entry_rsi = t.get("entry_rsi")
        entry_ok = _entry_ok_by_threshold(direction, entry_rsi, rsi_long, rsi_short)
        if entry_rsi is None:
            stats["no_entry_rsi"] += 1
            stats["bot_matched_bad_rsi"] += 1
        elif entry_ok:
            stats["bot_matched_ok"] += 1
        else:
            stats["bot_matched_bad_rsi"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Аудит сделок: вход/выход vs RSI и конфиг")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--from-exchange", action="store_true")
    parser.add_argument("--period", type=str, default="all")
    args = parser.parse_args()

    timeframe, auto_config, rsi_long, rsi_short = _load_config_thresholds()
    exit_long_with = auto_config.get("rsi_exit_long_with_trend") or 65
    exit_long_against = auto_config.get("rsi_exit_long_against_trend") or 60
    exit_short_with = auto_config.get("rsi_exit_short_with_trend") or 35
    exit_short_against = auto_config.get("rsi_exit_short_against_trend") or 40

    from bot_engine.bots_database import get_bots_database
    db = get_bots_database()
    db_trades = db.get_bot_trades_history(
        symbol=args.symbol,
        status="CLOSED",
        limit=args.limit,
    )
    db_index = _build_db_index(db_trades)

    if args.from_exchange:
        try:
            trades = load_trades_from_exchange(
                symbol_filter=args.symbol,
                limit=args.limit,
                period=args.period,
            )
            source_note = "с биржи (get_closed_pnl) + сопоставление с БД"
        except Exception as e:
            print(f"Ошибка загрузки с биржи: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        trades = db_trades
        for t in trades:
            t["trade_class"] = "bot_matched"
            t.setdefault("source", "database")
        source_note = "из БД (bot_trades_history)"

    stats = classify_trades(trades, db_index, rsi_long, rsi_short) if args.from_exchange else {
        "bot_matched_ok": 0,
        "bot_matched_bad_rsi": 0,
        "bot_no_db": 0,
        "manual_or_external": 0,
        "no_entry_rsi": 0,
    }
    if not args.from_exchange:
        for t in trades:
            direction = (t.get("direction") or "LONG").upper()
            entry_rsi = t.get("entry_rsi")
            entry_ok = _entry_ok_by_threshold(direction, entry_rsi, rsi_long, rsi_short)
            if entry_rsi is None:
                stats["no_entry_rsi"] += 1
                stats["bot_matched_bad_rsi"] += 1
            elif entry_ok:
                stats["bot_matched_ok"] += 1
            else:
                stats["bot_matched_bad_rsi"] += 1

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    def w(line=""):
        out.write(line + "\n")
        if out != sys.stdout:
            print(line)

    w("=" * 90)
    w("АУДИТ СДЕЛОК: ВХОД/ВЫХОД vs RSI И КОНФИГ")
    w("=" * 90)
    w(f"Источник: {source_note}")
    w(f"Таймфрейм из конфига: {timeframe}")
    w(f"Вход LONG:  RSI <= {rsi_long}")
    w(f"Вход SHORT: RSI >= {rsi_short}")
    w(f"Пороги выхода LONG: with>={exit_long_with}, against>={exit_long_against}")
    w(f"Пороги выхода SHORT: with<={exit_short_with}, against<={exit_short_against}")
    w(f"Всего сделок в выборке: {len(trades)}")
    w("")
    w("--- КЛАССИФИКАЦИЯ ---")
    w(f"  bot_matched, вход по порогу RSI:     {stats['bot_matched_ok']}")
    w(f"  bot_matched, вход ВНЕ порога / нет RSI: {stats['bot_matched_bad_rsi']}")
    w(f"  manual_or_external (нет в БД):       {stats['manual_or_external']}")
    w(f"  без entry_rsi в данных:              {stats['no_entry_rsi']}")
    w("")

    shown = 0
    max_verbose = 50 if args.verbose else 15
    for i, t in enumerate(trades):
        symbol = t.get("symbol", "")
        direction = (t.get("direction") or "LONG").upper()
        entry_rsi = t.get("entry_rsi")
        trade_class = t.get("trade_class") or "unknown"
        entry_ok = _entry_ok_by_threshold(direction, entry_rsi, rsi_long, rsi_short)
        if trade_class == "bot_matched" and entry_ok is True:
            continue
        if shown >= max_verbose and not args.verbose:
            break
        shown += 1
        w("-" * 90)
        w(f"Сделка #{i + 1}  {symbol}  {direction}  class={trade_class}")
        w(f"  Вход:  {t.get('entry_time')}  цена={t.get('entry_price')}  RSI={entry_rsi}  TF={t.get('entry_timeframe')}")
        w(f"  Выход: {t.get('exit_time')}  PnL={t.get('pnl')}  reason={t.get('close_reason')}")
        if entry_rsi is None:
            w("  Вход по эталону: нет RSI в данных")
        elif entry_ok is False:
            w(f"  Вход по эталону: НЕ по порогу (LONG RSI<={rsi_long}, SHORT RSI>={rsi_short})")
        elif entry_ok is True:
            w("  Вход по эталону: OK")
        if trade_class == "manual_or_external":
            w("  БД: запись не найдена (ручная сделка или вне бота)")
        w("")

    w("=" * 90)
    w("ИТОГ")
    w("=" * 90)
    total_bot = stats["bot_matched_ok"] + stats["bot_matched_bad_rsi"]
    if total_bot:
        pct_ok = 100.0 * stats["bot_matched_ok"] / total_bot
        w(f"Доля входов бота в пороге RSI: {pct_ok:.1f}% ({stats['bot_matched_ok']}/{total_bot})")
    if stats["manual_or_external"]:
        w(
            f"Сделок без записи в БД: {stats['manual_or_external']} — "
            "часто это ручная торговля на бирже, не баг бота."
        )
    if stats["bot_matched_bad_rsi"]:
        w("Рекомендация: проверить ENTRY_CHECK в логах, единый check_entry_allowed, таймфрейм свечей.")
    if args.output:
        out.close()
        print(f"Отчёт записан в {args.output}")


if __name__ == "__main__":
    main()
