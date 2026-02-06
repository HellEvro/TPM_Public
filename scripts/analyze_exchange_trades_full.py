#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальный анализ ВСЕХ сделок с биржи: загрузка get_closed_pnl, для каждой сделки — свечи по таймфрейму
из конфига, расчёт RSI на момент входа и выхода, сверка с текущим конфигом (эталон).

Эталон конфига: текущий configs/bot_config.py (AutoBotConfig + пороги RSI входа/выхода).

Запуск:
    python scripts/analyze_exchange_trades_full.py
    python scripts/analyze_exchange_trades_full.py --symbol 1000XECUSDT
    python scripts/analyze_exchange_trades_full.py --output report.txt
    python scripts/analyze_exchange_trades_full.py --db --output report.txt   # сопоставление с БД (close_reason, entry_rsi, exit_rsi)
    python scripts/analyze_exchange_trades_full.py --log-file logs/bots.log  # строки лога по символу и дате сделки
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Интервал свечи в мс по таймфрейму
TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


def _ts_to_ms(ts):
    """Приводит timestamp к миллисекундам."""
    if ts is None:
        return None
    try:
        t = float(ts)
        if t < 1e12:
            t *= 1000
        return int(t)
    except (TypeError, ValueError):
        return None


def _ts_ms_to_iso(ts_ms):
    if ts_ms is None:
        return ""
    try:
        s = float(ts_ms) / 1000.0
        return datetime.fromtimestamp(s, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts_ms)


def _infer_direction(side, entry_price, exit_price, pnl):
    if (side or "").upper() in ("BUY", "LONG"):
        return "LONG"
    if (side or "").upper() in ("SELL", "SHORT"):
        return "SHORT"
    if entry_price and exit_price:
        if exit_price >= entry_price:
            return "LONG" if (pnl or 0) >= 0 else "SHORT"
        return "SHORT" if (pnl or 0) >= 0 else "LONG"
    return "LONG" if (pnl or 0) >= 0 else "SHORT"


def get_exchange():
    try:
        from app.config import EXCHANGES, ACTIVE_EXCHANGE
    except ImportError:
        try:
            from configs.app_config import EXCHANGES, ACTIVE_EXCHANGE
        except ImportError:
            raise RuntimeError("Нужен app.config или configs.app_config (EXCHANGES, ACTIVE_EXCHANGE)")
    from exchanges.exchange_factory import ExchangeFactory
    name = ACTIVE_EXCHANGE
    cfg = EXCHANGES.get(name, {})
    if not cfg or not cfg.get("enabled", True):
        raise RuntimeError(f"Биржа {name} не включена")
    key = cfg.get("api_key")
    secret = cfg.get("api_secret")
    passphrase = cfg.get("passphrase")
    if not key or not secret:
        raise RuntimeError("В configs/keys.py не заполнены API ключи биржи")
    return ExchangeFactory.create_exchange(name, key, secret, passphrase), name


def load_config_etalon():
    """Загружает текущий конфиг как эталон (пороги входа/выхода по RSI)."""
    from bot_engine.config_loader import reload_config, get_current_timeframe
    try:
        reload_config()
    except Exception:
        pass
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            c = bots_data.get("auto_bot_config", {})
    except Exception:
        c = {}
    if not c:
        try:
            from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG
            c = DEFAULT_AUTO_BOT_CONFIG or {}
        except Exception:
            c = {}
    timeframe = get_current_timeframe() or "1m"
    return {
        "timeframe": timeframe,
        "rsi_long_threshold": c.get("rsi_long_threshold") or 29,
        "rsi_short_threshold": c.get("rsi_short_threshold") or 71,
        "rsi_exit_long_with_trend": c.get("rsi_exit_long_with_trend") or 65,
        "rsi_exit_long_against_trend": c.get("rsi_exit_long_against_trend") or 60,
        "rsi_exit_short_with_trend": c.get("rsi_exit_short_with_trend") or 35,
        "rsi_exit_short_against_trend": c.get("rsi_exit_short_against_trend") or 40,
    }, timeframe


def load_trades_from_exchange(exchange, symbol_filter=None, period="all"):
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
        entry_ms = _ts_to_ms(entry_ts)
        exit_ms = _ts_to_ms(close_ts)
        direction = _infer_direction(r.get("side"), entry_price, exit_price, pnl)
        trades.append({
            "symbol": sym,
            "direction": direction,
            "entry_time_iso": _ts_ms_to_iso(entry_ms),
            "exit_time_iso": _ts_ms_to_iso(exit_ms),
            "entry_ts_ms": entry_ms,
            "exit_ts_ms": exit_ms,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
        })
    trades.sort(key=lambda x: x.get("exit_ts_ms") or 0, reverse=True)
    return trades


# RSI(14) требует 15+ свечей до точки. Достаточно ~20 свечей до входа и до выхода — не грузим 1000.
RSI_CANDLES_NEEDED = 20


def get_candles_for_trade(exchange, symbol, timeframe, entry_ts_ms, exit_ts_ms, interval_ms):
    """Загружает минимум свечей для расчёта RSI в точке входа и выхода: 20 до входа + свечи между входом и выходом.
    Один запрос с limit ≈ 20 + (exit - entry) / interval, макс. 100."""
    if not hasattr(exchange, "get_chart_data_end_limit"):
        return None
    # Сколько свечей между входом и выходом
    span_ms = max(0, exit_ts_ms - entry_ts_ms)
    span_candles = int(span_ms / interval_ms) + 2 if interval_ms else 0
    limit = min(100, RSI_CANDLES_NEEDED + span_candles)
    limit = max(limit, 15)
    end_ms = exit_ts_ms + interval_ms
    try:
        resp = exchange.get_chart_data_end_limit(symbol, timeframe, end_ms, limit=limit)
        if not resp or not resp.get("success"):
            return None
        candles = (resp.get("data") or {}).get("candles") or []
        return sorted(candles, key=lambda c: c["time"]) if candles else None
    except Exception:
        return None


def rsi_at_timestamp(candles, ts_ms, interval_ms, period=14):
    """
    RSI на момент времени ts_ms: находим свечу, которая содержит ts_ms,
    и возвращаем RSI по закрытию этой свечи (по истории до неё включительно).
    """
    from bot_engine.utils.rsi_utils import calculate_rsi_history
    if not candles or len(candles) < period + 1:
        return None
    # Индекс свечи: последняя свеча с candle['time'] <= ts_ms
    idx = -1
    for i, c in enumerate(candles):
        if c["time"] <= ts_ms:
            idx = i
        else:
            break
    if idx < period:
        return None
    closes = [c["close"] for c in candles[: idx + 1]]
    hist = calculate_rsi_history(closes, period=period)
    if not hist:
        return None
    return round(hist[-1], 2)


def rsi_at_entry_last_closed_candle(candles, entry_ts_ms, interval_ms, period=14):
    """
    RSI на ВХОД: только по последней УЖЕ ЗАКРЫТОЙ свече до момента входа.
    Бот принимает решение по закрытой свече (не по текущей формирующейся).
    Свеча закрыта, если candle['time'] + interval_ms <= entry_ts_ms.
    """
    from bot_engine.utils.rsi_utils import calculate_rsi_history
    if not candles or len(candles) < period + 1:
        return None
    # Последняя свеча, которая успела закрыться до entry_ts_ms
    idx = -1
    for i, c in enumerate(candles):
        candle_end_ms = c["time"] + interval_ms
        if candle_end_ms <= entry_ts_ms:
            idx = i
        else:
            break
    if idx < period:
        return None
    closes = [c["close"] for c in candles[: idx + 1]]
    hist = calculate_rsi_history(closes, period=period)
    if not hist:
        return None
    return round(hist[-1], 2)


def load_db_trades_lookup():
    """Загружает закрытые сделки из bot_trades_history и возвращает список для сопоставления по (symbol, exit_ts_ms)."""
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        rows = db.get_bot_trades_history(status="CLOSED", limit=50000) or []
        lookup = {}
        for r in rows:
            sym = (r.get("symbol") or "").replace("USDT", "")
            exit_ts = r.get("exit_timestamp")
            if exit_ts is None:
                continue
            try:
                if isinstance(exit_ts, (int, float)) and exit_ts < 1e12:
                    exit_ts = int(exit_ts) * 1000
                else:
                    exit_ts = int(float(exit_ts))
            except (TypeError, ValueError):
                continue
            key = (sym, exit_ts)
            lookup[key] = r
        return lookup
    except Exception:
        return {}


def grep_log_for_trade(log_path, symbol, entry_iso, exit_iso, max_lines=5):
    """Ищет в лог-файле строки, содержащие symbol и дату входа/выхода. Возвращает список строк."""
    if not log_path or not Path(log_path).exists():
        return []
    entry_date = (entry_iso or "")[:10]
    exit_date = (exit_iso or "")[:10]
    sym_clean = symbol.replace("USDT", "") if symbol else ""
    found = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if sym_clean not in line and symbol not in line:
                    continue
                if entry_date in line or exit_date in line:
                    found.append(line.rstrip())
                    if len(found) >= max_lines:
                        break
    except Exception:
        pass
    return found


def main():
    parser = argparse.ArgumentParser(description="Детальный анализ сделок с биржи: даты, RSI входа/выхода, конфиг")
    parser.add_argument("--symbol", type=str, default=None, help="Фильтр по символу")
    parser.add_argument("--output", type=str, default=None, help="Файл отчёта")
    parser.add_argument("--limit", type=int, default=None, help="Макс. сделок для разбора (по умолчанию все)")
    parser.add_argument("--period", type=str, default="all", help="Период сделок с биржи: all, day, week, month")
    parser.add_argument("--db", action="store_true", help="Сопоставить с bot_trades_history (БД): вывести close_reason, entry_rsi, exit_rsi из БД")
    parser.add_argument("--log-file", type=str, default=None, help="Путь к лог-файлу: для каждой сделки вывести совпадающие строки лога по символу и дате")
    args = parser.parse_args()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    def w(line=""):
        out.write(line + "\n")
        if out != sys.stdout:
            print(line)

    w("=" * 90)
    w("ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК С БИРЖИ (даты, время, вход/выход, RSI, эталон конфиг)")
    w("=" * 90)

    # Эталон конфига
    config_etalon, timeframe = load_config_etalon()
    w("")
    w("ЭТАЛОН КОНФИГА (текущий configs/bot_config.py):")
    w(f"  Таймфрейм: {timeframe}")
    w(f"  Вход LONG:  RSI <= {config_etalon['rsi_long_threshold']}")
    w(f"  Вход SHORT: RSI >= {config_etalon['rsi_short_threshold']}")
    w(f"  Выход LONG:  (по тренду) RSI >= {config_etalon['rsi_exit_long_with_trend']}, (против) RSI >= {config_etalon['rsi_exit_long_against_trend']}")
    w(f"  Выход SHORT: (по тренду) RSI <= {config_etalon['rsi_exit_short_with_trend']}, (против) RSI <= {config_etalon['rsi_exit_short_against_trend']}")
    w("")

    exchange, ex_name = get_exchange()
    w(f"Биржа: {ex_name}")
    trades = load_trades_from_exchange(exchange, symbol_filter=args.symbol, period=args.period)
    if args.limit:
        trades = trades[: args.limit]
    w(f"Загружено сделок с биржи: {len(trades)}")
    db_lookup = load_db_trades_lookup() if args.db else {}
    if args.db:
        w(f"Загружено записей из БД (bot_trades_history): {len(db_lookup)}")
    w("")

    interval_ms = TF_MS.get(timeframe, 60_000)
    rsi_period = 14
    # Для каждой сделки — один лёгкий запрос: ~20–100 свечей (RSI 14 нужно 15+), не 1000 и не 30d
    for i, t in enumerate(trades):
        symbol = t["symbol"]
        direction = t["direction"]
        entry_ts_ms = t["entry_ts_ms"]
        exit_ts_ms = t["exit_ts_ms"]
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        pnl = t["pnl"]

        candles = get_candles_for_trade(exchange, symbol, timeframe, entry_ts_ms, exit_ts_ms, interval_ms)
        entry_rsi = None
        exit_rsi = None
        if candles:
            # Вход: RSI по последней ЗАКРЫТОЙ свече до входа (как в боте)
            entry_rsi = rsi_at_entry_last_closed_candle(candles, entry_ts_ms, interval_ms, rsi_period)
            # Выход: RSI на момент выхода (по свече, содержащей момент выхода)
            exit_rsi = rsi_at_timestamp(candles, exit_ts_ms, interval_ms, rsi_period)
        t["_entry_rsi"] = entry_rsi
        t["_exit_rsi"] = exit_rsi

        # Оценка по эталону (тренд входа неизвестен — считаем with_trend для LONG/SHORT)
        entry_ok = None
        exit_ok = None
        if direction == "LONG":
            entry_ok = entry_rsi is not None and entry_rsi <= config_etalon["rsi_long_threshold"]
            exit_ok = exit_rsi is not None and exit_rsi >= config_etalon["rsi_exit_long_with_trend"]
        else:
            entry_ok = entry_rsi is not None and entry_rsi >= config_etalon["rsi_short_threshold"]
            exit_ok = exit_rsi is not None and exit_rsi <= config_etalon["rsi_exit_short_with_trend"]

        w("-" * 90)
        w(f"Сделка #{i+1}  {symbol}  {direction}")
        w(f"  Вход:  {t['entry_time_iso']}  цена={entry_price}  RSI(на момент входа)={entry_rsi}")
        w(f"  Выход: {t['exit_time_iso']}  цена={exit_price}  RSI(на момент выхода)={exit_rsi}  PnL={pnl}")
        if entry_rsi is not None:
            w(f"  Вход по эталону:  {'OK (RSI в пороге)' if entry_ok else 'НЕ по порогу (LONG: RSI<=%s, SHORT: RSI>=%s)' % (config_etalon['rsi_long_threshold'], config_etalon['rsi_short_threshold'])}")
        else:
            w("  Вход по эталону:  RSI не рассчитан (нет свечей за период входа)")
        if exit_rsi is not None:
            w(f"  Выход по эталону: {'OK (RSI в пороге выхода)' if exit_ok else 'НЕ по порогу выхода (см. конфиг)'}")
        else:
            w("  Выход по эталону:  RSI не рассчитан (нет свечей за период выхода)")
        if args.db and db_lookup:
            sym_clean = (symbol or "").replace("USDT", "")
            db_rec = None
            for tol in [0, 60000, -60000]:
                db_rec = db_lookup.get((sym_clean, exit_ts_ms + tol))
                if db_rec:
                    break
            if db_rec:
                w(f"  БД: close_reason={db_rec.get('close_reason')} entry_rsi={db_rec.get('entry_rsi')} exit_rsi={db_rec.get('exit_rsi')}")
            else:
                w("  БД: запись не найдена")
        if args.log_file:
            log_lines = grep_log_for_trade(args.log_file, symbol, t["entry_time_iso"], t["exit_time_iso"], max_lines=3)
            if log_lines:
                w("  Лог:")
                for ll in log_lines:
                    w(f"    {ll[:120]}")
        w("")

    # Итог
    with_entry_rsi = sum(1 for t in trades if t.get("_entry_rsi") is not None)
    with_exit_rsi = sum(1 for t in trades if t.get("_exit_rsi") is not None)
    w("=" * 90)
    w("ИТОГ")
    w("=" * 90)
    w(f"Всего сделок: {len(trades)}")
    w(f"С RSI на входе (по свечам): {with_entry_rsi}")
    w(f"С RSI на выходе (по свечам): {with_exit_rsi}")
    w("")
    w("Примечание: RSI считается по свечам таймфрейма из конфига (эталон). На каждую сделку запрашивается минимум свечей: ~20 до точки входа + свечи до выхода (до 100), т.к. RSI(14) достаточно 15+ свечей.")
    w("")
    w("RSI входа: по последней ЗАКРЫТОЙ свече до входа (как в боте). Выход: по свече на момент выхода. Выходы не по порогу часто — TP/SL или ручное закрытие. Подробно: docs/RSI_ENTRY_EXIT_MISMATCH.md")
    if args.output:
        out.close()
        print(f"Отчёт записан в {args.output}")


if __name__ == "__main__":
    main()
