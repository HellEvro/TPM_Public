# -*- coding: utf-8 -*-
"""
Аудит RSI входа/выхода: загрузка сделок с биржи, расчёт RSI в точке входа и выхода,
сверка с текущим конфигом. LONG: вход корректен при RSI <= порог; SHORT: при RSI >= порог.
Всё вне диапазона — ошибочные входы/выходы для разбора в коде и логах.
"""

from datetime import datetime, timezone

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
}

RSI_CANDLES_NEEDED = 20


def _ts_to_ms(ts):
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


def load_config_etalon():
    """Текущий конфиг (не дефолт): пороги входа/выхода по RSI."""
    try:
        from bot_engine.config_loader import reload_config, get_current_timeframe
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
    from bot_engine.config_loader import get_current_timeframe
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


def get_candles_for_trade(exchange, symbol, timeframe, entry_ts_ms, exit_ts_ms, interval_ms):
    if not hasattr(exchange, "get_chart_data_end_limit"):
        return None
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
    from bot_engine.utils.rsi_utils import calculate_rsi_history
    if not candles or len(candles) < period + 1:
        return None
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
    from bot_engine.utils.rsi_utils import calculate_rsi_history
    if not candles or len(candles) < period + 1:
        return None
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


def run_rsi_audit(exchange, limit=None, symbol_filter=None, period="all"):
    """
    Загружает сделки с биржи, считает RSI входа/выхода, сверяет с текущим конфигом.
    LONG: вход корректен при entry_rsi <= rsi_long_threshold; иначе — ошибочный вход.
    SHORT: вход корректен при entry_rsi >= rsi_short_threshold; иначе — ошибочный вход.

    Returns:
        dict: config, trades[], summary, generated_at
    """
    config_etalon, timeframe = load_config_etalon()
    trades_raw = load_trades_from_exchange(exchange, symbol_filter=symbol_filter, period=period)
    if limit:
        trades_raw = trades_raw[:limit]
    interval_ms = TF_MS.get(timeframe, 60_000)
    rsi_period = 14
    long_th = config_etalon["rsi_long_threshold"]
    short_th = config_etalon["rsi_short_threshold"]
    exit_long_th = config_etalon["rsi_exit_long_with_trend"]
    exit_short_th = config_etalon["rsi_exit_short_with_trend"]

    trades = []
    entry_ok = 0
    entry_error = 0
    entry_no_rsi = 0
    exit_ok = 0
    exit_error = 0
    exit_no_rsi = 0

    for t in trades_raw:
        symbol = t["symbol"]
        direction = t["direction"]
        entry_ts_ms = t["entry_ts_ms"]
        exit_ts_ms = t["exit_ts_ms"]
        pnl = t["pnl"]
        candles = get_candles_for_trade(exchange, symbol, timeframe, entry_ts_ms, exit_ts_ms, interval_ms)
        entry_rsi = None
        exit_rsi = None
        if candles:
            entry_rsi = rsi_at_entry_last_closed_candle(candles, entry_ts_ms, interval_ms, rsi_period)
            exit_rsi = rsi_at_timestamp(candles, exit_ts_ms, interval_ms, rsi_period)

        if direction == "LONG":
            entry_ok_this = entry_rsi is not None and entry_rsi <= long_th
            entry_error_this = entry_rsi is not None and entry_rsi > long_th
            entry_threshold = long_th
            exit_ok_this = exit_rsi is not None and exit_rsi >= exit_long_th
            exit_error_this = exit_rsi is not None and exit_rsi < exit_long_th
            exit_threshold = exit_long_th
        else:
            entry_ok_this = entry_rsi is not None and entry_rsi >= short_th
            entry_error_this = entry_rsi is not None and entry_rsi < short_th
            entry_threshold = short_th
            exit_ok_this = exit_rsi is not None and exit_rsi <= exit_short_th
            exit_error_this = exit_rsi is not None and exit_rsi > exit_short_th
            exit_threshold = exit_short_th

        if entry_rsi is None:
            entry_no_rsi += 1
        elif entry_ok_this:
            entry_ok += 1
        else:
            entry_error += 1
        if exit_rsi is None:
            exit_no_rsi += 1
        elif exit_ok_this:
            exit_ok += 1
        else:
            exit_error += 1

        trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry_time_iso": t["entry_time_iso"],
            "exit_time_iso": t["exit_time_iso"],
            "entry_rsi": entry_rsi,
            "exit_rsi": exit_rsi,
            "entry_ok": entry_ok_this,
            "exit_ok": exit_ok_this,
            "entry_error": entry_error_this,
            "exit_error": exit_error_this,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "pnl": pnl,
        })

    return {
        "config": config_etalon,
        "timeframe": timeframe,
        "trades": trades,
        "summary": {
            "total": len(trades),
            "entry_ok": entry_ok,
            "entry_error": entry_error,
            "entry_no_rsi": entry_no_rsi,
            "exit_ok": exit_ok,
            "exit_error": exit_error,
            "exit_no_rsi": exit_no_rsi,
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
