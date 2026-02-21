#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аналитический модуль торговли.

Анализирует ВСЕ сделки на бирже и сделки ботов из БД:
- Сверка биржа vs bot_trades_history (потерянные/лишние/расхождения)
- Метрики: Win Rate, PnL, просадка, серии убытков
- Анализ по причинам закрытия (Stop Loss, Take Profit, ошибки)
- Разбивка по символам, ботам, источникам решений
- Отчёт в виде словаря/JSON для использования AI модулем
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Допуск при сверке: разница времени выхода (секунды)
RECONCILE_TIME_TOLERANCE_SEC = 120
# Допуск по PnL (абсолютный USDT) для совпадения
RECONCILE_PNL_TOLERANCE = 0.5

# Пороги для «неудачных» монет и настроек
MIN_TRADES_FOR_UNSUCCESSFUL_COIN = 3       # минимум сделок по монете, чтобы считать её неудачной
UNSUCCESSFUL_WIN_RATE_THRESHOLD_PCT = 45  # ниже этого Win Rate — монета считается неудачной
BAD_RSI_WIN_RATE_THRESHOLD_PCT = 40       # диапазон RSI с Win Rate ниже — «неудачная настройка»
MIN_TRADES_FOR_BAD_RSI_BUCKET = 2         # минимум сделок в диапазоне RSI для вывода

# Пороги для «удачных» монет и настроек
SUCCESSFUL_WIN_RATE_THRESHOLD_PCT = 55    # выше этого Win Rate и PnL > 0 — удачная монета
GOOD_RSI_WIN_RATE_THRESHOLD_PCT = 55      # диапазон RSI/тренд с Win Rate выше — «удачная настройка»

# Диапазоны RSI для аналитики (вход в сделку)
RSI_BUCKETS = [
    (0, 25, "0-25 (сильный перепроданность)"),
    (26, 30, "26-30"),
    (31, 35, "31-35"),
    (36, 40, "36-40"),
    (41, 50, "41-50 (нейтрал)"),
    (51, 60, "51-60"),
    (61, 65, "61-65"),
    (66, 70, "66-70"),
    (71, 100, "71-100 (перекупленность)"),
]


@dataclass
class TradeSummary:
    """Нормализованная краткая сводка по сделке для сверки и аналитики."""
    symbol: str
    exit_timestamp: float  # секунды (Unix)
    pnl: float
    entry_price: float
    exit_price: float
    position_size_usdt: Optional[float]
    direction: str
    source: str  # 'exchange' | 'bot'
    bot_id: Optional[str] = None
    close_reason: Optional[str] = None
    decision_source: Optional[str] = None
    raw_id: Any = None
    raw: Optional[Dict[str, Any]] = None


def _ts_to_seconds(ts: Any) -> Optional[float]:
    """Приводит timestamp к секундам (Unix)."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                return ts / 1000.0
            return float(ts)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.timestamp()
    except Exception:
        pass
    return None


def _normalize_symbol(s: Optional[str]) -> str:
    if not s:
        return ""
    return (s or "").replace("USDT", "").strip().upper()


def exchange_trades_to_summaries(
    raw_list: List[Dict[str, Any]],
    exchange_name: str = "exchange",
) -> List[TradeSummary]:
    """Преобразует ответ get_closed_pnl биржи в список TradeSummary."""
    result = []
    for i, t in enumerate(raw_list):
        try:
            symbol = _normalize_symbol(t.get("symbol"))
            close_ts = t.get("close_timestamp") or t.get("closeTime") or 0
            ts_sec = _ts_to_seconds(close_ts)
            if ts_sec is None:
                ts_sec = 0.0
            pnl = float(t.get("closed_pnl") or t.get("closedPnl") or 0)
            entry_price = float(t.get("entry_price") or t.get("avgEntryPrice") or 0)
            exit_price = float(t.get("exit_price") or t.get("avgExitPrice") or 0)
            position_value = t.get("position_value")
            if position_value is None and entry_price and t.get("qty"):
                position_value = abs(float(t.get("qty", 0)) * entry_price)
            if position_value is not None:
                position_value = float(position_value)
            side = (t.get("side") or "").upper()
            direction = "LONG" if side in ("BUY", "LONG") else "SHORT"
            result.append(
                TradeSummary(
                    symbol=symbol,
                    exit_timestamp=ts_sec,
                    pnl=pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size_usdt=position_value,
                    direction=direction,
                    source="exchange",
                    raw_id=i,
                    raw=t,
                )
            )
        except Exception as e:
            logger.debug("Пропуск записи биржи %s: %s", t, e)
    return result


def _recalc_pnl_from_prices(
    entry_price: float,
    exit_price: float,
    direction: str,
    position_size_usdt: Optional[float],
    position_size_coins: Optional[float],
) -> Optional[float]:
    """Пересчитывает PnL в USDT из цен входа/выхода и размера позиции. Возвращает None если не хватает данных."""
    if not entry_price or entry_price <= 0 or not exit_price:
        return None
    direction = (direction or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        return None
    if direction == "LONG":
        roi_fraction = (exit_price - entry_price) / entry_price
    else:
        roi_fraction = (entry_price - exit_price) / entry_price
    position_value = position_size_usdt
    if (position_value is None or position_value == 0) and position_size_coins and position_size_coins > 0:
        position_value = position_size_coins * entry_price
    if position_value is None or position_value == 0:
        return None
    return roi_fraction * position_value


def bot_trades_to_summaries(
    raw_list: List[Dict[str, Any]],
) -> List[TradeSummary]:
    """Преобразует записи bot_trades_history в список TradeSummary.
    КРИТИЧНО: если в БД pnl отсутствует или 0 — пересчитывает PnL из entry/exit цен и размера позиции,
    иначе отчёт показывает неверный общий минус (много сделок с pnl=0)."""
    result = []
    for t in raw_list:
        try:
            symbol = _normalize_symbol(t.get("symbol"))
            exit_ts = t.get("exit_timestamp")
            ts_sec = _ts_to_seconds(exit_ts)
            if ts_sec is None and t.get("exit_time"):
                ts_sec = _ts_to_seconds(t.get("exit_time"))
            if ts_sec is None:
                continue
            entry_price = float(t.get("entry_price") or 0)
            exit_price = float(t.get("exit_price") or 0)
            position_size = t.get("position_size_usdt")
            if position_size is not None:
                position_size = float(position_size)
            position_size_coins = t.get("position_size_coins")
            if position_size_coins is not None:
                position_size_coins = float(position_size_coins)
            direction = (t.get("direction") or "LONG").upper()
            if direction not in ("LONG", "SHORT"):
                direction = "LONG"
            # Берём сохранённый pnl; если нет или 0 — пересчитываем из цен
            stored_pnl = t.get("pnl")
            if stored_pnl is not None and stored_pnl != "":
                try:
                    pnl = float(stored_pnl)
                except (TypeError, ValueError):
                    pnl = 0.0
            else:
                pnl = 0.0
            if (pnl == 0.0 or stored_pnl is None) and entry_price and exit_price:
                recalc = _recalc_pnl_from_prices(
                    entry_price, exit_price, direction, position_size, position_size_coins
                )
                if recalc is not None:
                    pnl = recalc
            result.append(
                TradeSummary(
                    symbol=symbol,
                    exit_timestamp=ts_sec,
                    pnl=pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size_usdt=position_size,
                    direction=direction,
                    source="bot",
                    bot_id=t.get("bot_id"),
                    close_reason=t.get("close_reason"),
                    decision_source=t.get("decision_source"),
                    raw_id=t.get("id"),
                    raw=t,
                )
            )
        except Exception as e:
            logger.debug("Пропуск записи бота %s: %s", t.get("id"), e)
    return result


def reconcile_trades(
    exchange_summaries: List[TradeSummary],
    bot_summaries: List[TradeSummary],
    time_tolerance_sec: float = RECONCILE_TIME_TOLERANCE_SEC,
    pnl_tolerance: float = RECONCILE_PNL_TOLERANCE,
) -> Dict[str, Any]:
    """
    Сверка сделок биржи и ботов.
    Возвращает: matched, only_on_exchange, only_in_bots, pnl_mismatches.
    """
    only_on_exchange: List[Dict[str, Any]] = []
    only_in_bots: List[Dict[str, Any]] = []
    matched: List[Dict[str, Any]] = []
    pnl_mismatches: List[Dict[str, Any]] = []

    used_bot = set()

    for ex in exchange_summaries:
        best_bot: Optional[TradeSummary] = None
        best_diff = float("inf")
        for bot in bot_summaries:
            if bot.raw_id in used_bot:
                continue
            if bot.symbol != ex.symbol:
                continue
            time_diff = abs(bot.exit_timestamp - ex.exit_timestamp)
            if time_diff > time_tolerance_sec:
                continue
            pnl_diff = abs(bot.pnl - ex.pnl)
            total_diff = time_diff + pnl_diff * 10
            if total_diff < best_diff:
                best_diff = total_diff
                best_bot = bot

        if best_bot is None:
            only_on_exchange.append({
                "symbol": ex.symbol,
                "exit_timestamp": ex.exit_timestamp,
                "pnl": ex.pnl,
                "entry_price": ex.entry_price,
                "exit_price": ex.exit_price,
                "position_size_usdt": ex.position_size_usdt,
                "direction": ex.direction,
                "raw": ex.raw,
            })
            continue

        used_bot.add(best_bot.raw_id)
        pnl_diff = abs(best_bot.pnl - ex.pnl)
        if pnl_diff > pnl_tolerance:
            pnl_mismatches.append({
                "symbol": ex.symbol,
                "exchange_pnl": ex.pnl,
                "bot_pnl": best_bot.pnl,
                "diff": best_bot.pnl - ex.pnl,
                "exit_timestamp": ex.exit_timestamp,
                "bot_id": best_bot.bot_id,
            })
        matched.append({
            "symbol": ex.symbol,
            "exit_timestamp": ex.exit_timestamp,
            "pnl": ex.pnl,
            "entry_price": ex.entry_price,
            "exit_price": ex.exit_price,
            "position_size_usdt": ex.position_size_usdt,
            "direction": ex.direction,
            "bot_id": best_bot.bot_id,
            "bot_db_id": best_bot.raw_id,
            "close_reason": best_bot.close_reason,
            "decision_source": best_bot.decision_source,
            "bot_raw": best_bot.raw,
        })

    for bot in bot_summaries:
        if bot.raw_id in used_bot:
            continue
        only_in_bots.append({
            "symbol": bot.symbol,
            "exit_timestamp": bot.exit_timestamp,
            "pnl": bot.pnl,
            "bot_id": bot.bot_id,
            "close_reason": bot.close_reason,
            "decision_source": bot.decision_source,
            "bot": bot,
        })

    return {
        "matched_count": len(matched),
        "only_on_exchange_count": len(only_on_exchange),
        "only_in_bots_count": len(only_in_bots),
        "pnl_mismatch_count": len(pnl_mismatches),
        "matched": matched,
        "only_on_exchange": only_on_exchange,
        "only_in_bots": only_in_bots,
        "pnl_mismatches": pnl_mismatches,
    }


def _compute_series(trades: List[TradeSummary]) -> Dict[str, Any]:
    """Считает серии прибыльных/убыточных сделок."""
    if not trades:
        return {"max_consecutive_wins": 0, "max_consecutive_losses": 0, "current_streak": 0}
    sorted_trades = sorted(trades, key=lambda x: x.exit_timestamp)
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0
    for t in sorted_trades:
        if t.pnl > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        elif t.pnl < 0:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
        else:
            cur_wins = 0
            cur_losses = 0
    last = sorted_trades[-1]
    current_streak = cur_wins if last.pnl > 0 else (-cur_losses if last.pnl < 0 else 0)
    return {
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "current_streak": current_streak,
    }


def _compute_drawdown(trades: List[TradeSummary]) -> Dict[str, Any]:
    """Считает просадку по эквити (кумулятивный PnL)."""
    if not trades:
        return {"max_drawdown_usdt": 0.0, "max_drawdown_pct": 0.0, "equity_curve": []}
    sorted_trades = sorted(trades, key=lambda x: x.exit_timestamp)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    curve = []
    for t in sorted_trades:
        equity += t.pnl
        curve.append({"exit_timestamp": t.exit_timestamp, "equity": equity})
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        if peak > 0 and peak - equity > 0:
            pct = 100.0 * (peak - equity) / peak
            if pct > max_dd_pct:
                max_dd_pct = pct
    return {
        "max_drawdown_usdt": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "final_equity": round(equity, 2),
        "equity_curve": curve[-500:],  # последние 500 точек для экономии
    }


def _get_entry_rsi(t: TradeSummary) -> Optional[float]:
    """Извлекает RSI на входе из сырой записи сделки."""
    if not t.raw:
        return None
    v = t.raw.get("entry_rsi") or t.raw.get("rsi")
    if v is None:
        return None
    try:
        f = float(v)
        if 0 <= f <= 100:
            return f
    except (TypeError, ValueError):
        pass
    return None


def _get_entry_trend(t: TradeSummary) -> Optional[str]:
    """Извлекает тренд на входе из сырой записи сделки."""
    if not t.raw:
        return None
    v = t.raw.get("entry_trend") or t.raw.get("trend")
    if v is None or v == "":
        return None
    return str(v).strip().upper() or None


def _trade_summary_to_dict(t: TradeSummary) -> Dict[str, Any]:
    """Сериализация одной сделки для отчёта (все метрики для UI/экспорта)."""
    exit_dt = datetime.fromtimestamp(t.exit_timestamp, tz=timezone.utc) if t.exit_timestamp else None
    return {
        "symbol": t.symbol,
        "exit_timestamp": t.exit_timestamp,
        "exit_time_iso": exit_dt.isoformat() if exit_dt else None,
        "pnl": round(t.pnl, 4),
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "position_size_usdt": t.position_size_usdt,
        "direction": t.direction,
        "bot_id": t.bot_id,
        "close_reason": t.close_reason,
        "decision_source": t.decision_source,
        "entry_rsi": _get_entry_rsi(t),
        "entry_trend": _get_entry_trend(t),
    }


def _rsi_bucket_label(rsi: float) -> str:
    """Возвращает метку диапазона RSI для заданного значения."""
    for low, high, label in RSI_BUCKETS:
        if low <= rsi <= high:
            return label
    return "unknown"


def _deduplicate_trade_summaries(summaries: List[TradeSummary], window_sec: float = 120.0) -> List[TradeSummary]:
    """Убирает дубликаты: одна и та же сделка могла попасть из бота и из импорта с биржи. Группировка по (symbol, exit_timestamp в окне window_sec)."""
    if not summaries:
        return summaries
    seen: Dict[tuple, TradeSummary] = {}
    for t in summaries:
        ts = t.exit_timestamp
        if ts is None or ts <= 0:
            key = (t.symbol, -1.0)
        else:
            bucket = round(ts / window_sec) * window_sec
            key = (t.symbol, bucket)
        if key not in seen:
            seen[key] = t
        else:
            # Оставляем запись с более полными данными (есть close_reason) или с большим |pnl|
            existing = seen[key]
            if (t.close_reason or t.bot_id) and not (existing.close_reason or existing.bot_id):
                seen[key] = t
            elif t.raw and existing.raw and (t.raw.get("entry_rsi") is not None) and (existing.raw.get("entry_rsi") is None):
                seen[key] = t
    return list(seen.values())


def analyze_bot_trades(
    bot_summaries: List[TradeSummary],
    trades_list_max: int = 5000,
) -> Dict[str, Any]:
    """Полная аналитика по сделкам ботов (без биржи). Перед расчётом дубликаты по (symbol, exit_timestamp) отбрасываются."""
    closed = [t for t in bot_summaries if t.raw and (t.raw.get("status") == "CLOSED" or t.pnl != 0 or t.raw.get("exit_timestamp"))]
    if not closed:
        closed = bot_summaries
    closed = _deduplicate_trade_summaries(closed)

    total = len(closed)
    total_pnl = sum(t.pnl for t in closed)
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    win_count = len(wins)
    loss_count = len(losses)
    neutral_count = total - win_count - loss_count
    win_rate = (win_count / total * 100) if total else 0.0
    avg_win = (sum(t.pnl for t in wins) / win_count) if win_count else 0.0
    avg_loss = (sum(t.pnl for t in losses) / loss_count) if loss_count else 0.0
    total_win_pnl = sum(t.pnl for t in wins)
    total_loss_pnl = sum(t.pnl for t in losses)
    profit_factor = (total_win_pnl / abs(total_loss_pnl)) if total_loss_pnl and total_loss_pnl < 0 else (None if not total_win_pnl else float("inf"))

    # Сделки с pnl=0 не считаем ни выигрышем, ни проигрышем (часто при RSI_EXIT без сохранённого размера позиции)
    by_close_reason: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    for t in closed:
        reason = t.close_reason or "UNKNOWN"
        by_close_reason[reason]["count"] += 1
        by_close_reason[reason]["pnl"] += t.pnl
        if t.pnl > 0:
            by_close_reason[reason]["wins"] += 1
        elif t.pnl < 0:
            by_close_reason[reason]["losses"] += 1
        else:
            by_close_reason[reason]["neutral"] += 1

    by_symbol: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    for t in closed:
        by_symbol[t.symbol]["count"] += 1
        by_symbol[t.symbol]["pnl"] += t.pnl
        if t.pnl > 0:
            by_symbol[t.symbol]["wins"] += 1
        elif t.pnl < 0:
            by_symbol[t.symbol]["losses"] += 1
        else:
            by_symbol[t.symbol]["neutral"] += 1

    by_decision_source: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    for t in closed:
        src = t.decision_source or "UNKNOWN"
        by_decision_source[src]["count"] += 1
        by_decision_source[src]["pnl"] += t.pnl
        if t.pnl > 0:
            by_decision_source[src]["wins"] += 1
        elif t.pnl < 0:
            by_decision_source[src]["losses"] += 1
        else:
            by_decision_source[src]["neutral"] += 1

    by_bot: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    for t in closed:
        bid = t.bot_id or "NO_BOT"
        by_bot[bid]["count"] += 1
        by_bot[bid]["pnl"] += t.pnl
        if t.pnl > 0:
            by_bot[bid]["wins"] += 1
        elif t.pnl < 0:
            by_bot[bid]["losses"] += 1
        else:
            by_bot[bid]["neutral"] += 1

    # По символам: разбивка по RSI на входе и по тренду на входе
    by_symbol_rsi: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    )
    by_symbol_trend: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0, "neutral": 0})
    )
    for t in closed:
        rsi = _get_entry_rsi(t)
        if rsi is not None:
            bucket = _rsi_bucket_label(rsi)
            by_symbol_rsi[t.symbol][bucket]["count"] += 1
            by_symbol_rsi[t.symbol][bucket]["pnl"] += t.pnl
            if t.pnl > 0:
                by_symbol_rsi[t.symbol][bucket]["wins"] += 1
            elif t.pnl < 0:
                by_symbol_rsi[t.symbol][bucket]["losses"] += 1
            else:
                by_symbol_rsi[t.symbol][bucket]["neutral"] += 1
        trend = _get_entry_trend(t) or "UNKNOWN"
        by_symbol_trend[t.symbol][trend]["count"] += 1
        by_symbol_trend[t.symbol][trend]["pnl"] += t.pnl
        if t.pnl > 0:
            by_symbol_trend[t.symbol][trend]["wins"] += 1
        elif t.pnl < 0:
            by_symbol_trend[t.symbol][trend]["losses"] += 1
        else:
            by_symbol_trend[t.symbol][trend]["neutral"] += 1

    # Неудачные монеты: достаточно сделок и (отрицательный PnL или низкий Win Rate)
    unsuccessful_coins: List[Dict[str, Any]] = []
    for symbol, data in by_symbol.items():
        count = data["count"]
        if count < MIN_TRADES_FOR_UNSUCCESSFUL_COIN:
            continue
        pnl = data["pnl"]
        wins = data["wins"]
        wr = (wins / count * 100) if count else 0
        reasons = []
        if pnl < 0:
            reasons.append("negative_pnl")
        if wr < UNSUCCESSFUL_WIN_RATE_THRESHOLD_PCT:
            reasons.append("low_win_rate")
        if not reasons:
            continue
        unsuccessful_coins.append({
            "symbol": symbol,
            "trades_count": count,
            "pnl_usdt": round(pnl, 2),
            "win_rate_pct": round(wr, 2),
            "wins": data["wins"],
            "losses": data["losses"],
            "reasons": reasons,
        })
    unsuccessful_coins.sort(key=lambda x: (x["pnl_usdt"], -x["win_rate_pct"]))

    # Неудачные настройки по RSI и тренду для каждой неудачной монеты
    unsuccessful_settings: List[Dict[str, Any]] = []
    for uc in unsuccessful_coins:
        symbol = uc["symbol"]
        bad_rsi: List[Dict[str, Any]] = []
        rsi_data = by_symbol_rsi.get(symbol, {})
        for bucket, b in rsi_data.items():
            if b["count"] < MIN_TRADES_FOR_BAD_RSI_BUCKET:
                continue
            wr = (b["wins"] / b["count"] * 100) if b["count"] else 0
            if wr < BAD_RSI_WIN_RATE_THRESHOLD_PCT or b["pnl"] < 0:
                bad_rsi.append({
                    "rsi_range": bucket,
                    "trades_count": b["count"],
                    "pnl_usdt": round(b["pnl"], 2),
                    "win_rate_pct": round(wr, 2),
                })
        bad_trends: List[Dict[str, Any]] = []
        trend_data = by_symbol_trend.get(symbol, {})
        for trend_name, b in trend_data.items():
            if trend_name == "UNKNOWN" and b["count"] < MIN_TRADES_FOR_BAD_RSI_BUCKET:
                continue
            if b["count"] < MIN_TRADES_FOR_BAD_RSI_BUCKET:
                continue
            wr = (b["wins"] / b["count"] * 100) if b["count"] else 0
            if wr < BAD_RSI_WIN_RATE_THRESHOLD_PCT or b["pnl"] < 0:
                bad_trends.append({
                    "trend": trend_name,
                    "trades_count": b["count"],
                    "pnl_usdt": round(b["pnl"], 2),
                    "win_rate_pct": round(wr, 2),
                })
        unsuccessful_settings.append({
            "symbol": symbol,
            "bad_rsi_ranges": bad_rsi,
            "bad_trends": bad_trends,
            "rsi_summary": {k: dict(v) for k, v in rsi_data.items()} if rsi_data else {},
            "trend_summary": {k: dict(v) for k, v in trend_data.items()} if trend_data else {},
        })

    # Удачные монеты: достаточно сделок, PnL > 0 и Win Rate >= порога
    successful_coins: List[Dict[str, Any]] = []
    for symbol, data in by_symbol.items():
        count = data["count"]
        if count < MIN_TRADES_FOR_UNSUCCESSFUL_COIN:
            continue
        pnl = data["pnl"]
        wins = data["wins"]
        wr = (wins / count * 100) if count else 0
        if pnl <= 0 or wr < SUCCESSFUL_WIN_RATE_THRESHOLD_PCT:
            continue
        successful_coins.append({
            "symbol": symbol,
            "trades_count": count,
            "pnl_usdt": round(pnl, 2),
            "win_rate_pct": round(wr, 2),
            "wins": data["wins"],
            "losses": data["losses"],
        })
    successful_coins.sort(key=lambda x: (-x["pnl_usdt"], -x["win_rate_pct"]))

    # Удачные настройки по RSI и тренду для каждой удачной монеты
    successful_settings: List[Dict[str, Any]] = []
    for sc in successful_coins:
        symbol = sc["symbol"]
        good_rsi: List[Dict[str, Any]] = []
        rsi_data = by_symbol_rsi.get(symbol, {})
        for bucket, b in rsi_data.items():
            if b["count"] < MIN_TRADES_FOR_BAD_RSI_BUCKET:
                continue
            wr = (b["wins"] / b["count"] * 100) if b["count"] else 0
            if wr >= GOOD_RSI_WIN_RATE_THRESHOLD_PCT and b["pnl"] > 0:
                good_rsi.append({
                    "rsi_range": bucket,
                    "trades_count": b["count"],
                    "pnl_usdt": round(b["pnl"], 2),
                    "win_rate_pct": round(wr, 2),
                })
        good_trends: List[Dict[str, Any]] = []
        trend_data = by_symbol_trend.get(symbol, {})
        for trend_name, b in trend_data.items():
            if b["count"] < MIN_TRADES_FOR_BAD_RSI_BUCKET:
                continue
            wr = (b["wins"] / b["count"] * 100) if b["count"] else 0
            if wr >= GOOD_RSI_WIN_RATE_THRESHOLD_PCT and b["pnl"] > 0:
                good_trends.append({
                    "trend": trend_name,
                    "trades_count": b["count"],
                    "pnl_usdt": round(b["pnl"], 2),
                    "win_rate_pct": round(wr, 2),
                })
        successful_settings.append({
            "symbol": symbol,
            "good_rsi_ranges": good_rsi,
            "good_trends": good_trends,
        })

    series = _compute_series(closed)
    drawdown = _compute_drawdown(closed)

    # Потенциальные ошибки: по close_reason или по признакам
    error_keywords = ("error", "ERROR", "fail", "exception", "timeout", "cancel", "reject")
    possible_errors = []
    for t in closed:
        reason = (t.close_reason or "")
        if any(kw in reason for kw in error_keywords):
            possible_errors.append({
                "symbol": t.symbol,
                "exit_timestamp": t.exit_timestamp,
                "pnl": t.pnl,
                "close_reason": t.close_reason,
                "bot_id": t.bot_id,
            })
        if t.raw:
            extra = t.raw.get("extra_data") or t.raw.get("extra_data_json")
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            if extra and isinstance(extra, dict) and any(kw in str(extra).lower() for kw in ("error", "fail", "exception")):
                possible_errors.append({
                    "symbol": t.symbol,
                    "exit_timestamp": t.exit_timestamp,
                    "pnl": t.pnl,
                    "close_reason": t.close_reason,
                    "bot_id": t.bot_id,
                    "extra": extra,
                })

    # Список сделок для детальной таблицы (последние по времени, лимит trades_list_max)
    sorted_closed = sorted(closed, key=lambda x: x.exit_timestamp or 0)
    trades_list = [_trade_summary_to_dict(t) for t in sorted_closed[-trades_list_max:]]

    return {
        "total_trades": total,
        "total_pnl_usdt": round(total_pnl, 2),
        "win_count": win_count,
        "loss_count": loss_count,
        "neutral_count": neutral_count,
        "win_rate_pct": round(win_rate, 2),
        "avg_win_usdt": round(avg_win, 2),
        "avg_loss_usdt": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None and profit_factor != float("inf") else (None if profit_factor is None else 999.99),
        "by_close_reason": {k: dict(v) for k, v in by_close_reason.items()},
        "by_symbol": {k: dict(v) for k, v in by_symbol.items()},
        "by_decision_source": {k: dict(v) for k, v in by_decision_source.items()},
        "by_bot": {k: dict(v) for k, v in by_bot.items()},
        "consecutive_series": series,
        "drawdown": drawdown,
        "possible_errors_count": len(possible_errors),
        "possible_errors": possible_errors[:100],
        "unsuccessful_coins": unsuccessful_coins,
        "unsuccessful_settings": unsuccessful_settings,
        "successful_coins": successful_coins,
        "successful_settings": successful_settings,
        "trades": trades_list,
    }


def analyze_exchange_trades(
    exchange_summaries: List[TradeSummary],
) -> Dict[str, Any]:
    """Аналитика только по сделкам с биржи (агрегированная)."""
    total = len(exchange_summaries)
    total_pnl = sum(t.pnl for t in exchange_summaries)
    wins = [t for t in exchange_summaries if t.pnl > 0]
    losses = [t for t in exchange_summaries if t.pnl < 0]
    win_rate = (len(wins) / total * 100) if total else 0.0
    by_symbol: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in exchange_summaries:
        by_symbol[t.symbol]["count"] += 1
        by_symbol[t.symbol]["pnl"] += t.pnl
    series = _compute_series(exchange_summaries)
    drawdown = _compute_drawdown(exchange_summaries)
    return {
        "total_trades": total,
        "total_pnl_usdt": round(total_pnl, 2),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "by_symbol": {k: dict(v) for k, v in by_symbol.items()},
        "consecutive_series": series,
        "drawdown": drawdown,
    }


def _build_merged_trade_summaries(
    reconciliation: Dict[str, Any],
    ex_summaries: List[TradeSummary],
    bot_summaries: List[TradeSummary],
) -> List[TradeSummary]:
    """
    Строит объединённый список сделок: для совпавших — данные биржи (цены, PnL),
    для остальных — данные бота или биржи. Источник истины по ценам и PnL — биржа.
    """
    merged: List[TradeSummary] = []
    # Совпавшие: берём цены и PnL с биржи, close_reason/decision_source/entry_rsi/entry_trend — из бота
    for m in reconciliation.get("matched", []):
        raw = m.get("bot_raw") or {}
        merged.append(
            TradeSummary(
                symbol=m["symbol"],
                exit_timestamp=m["exit_timestamp"],
                pnl=m["pnl"],
                entry_price=m.get("entry_price") or 0.0,
                exit_price=m.get("exit_price") or 0.0,
                position_size_usdt=m.get("position_size_usdt"),
                direction=m.get("direction") or "LONG",
                source="exchange",
                bot_id=m.get("bot_id"),
                close_reason=m.get("close_reason"),
                decision_source=m.get("decision_source"),
                raw_id=m.get("bot_id"),
                raw=raw,
            )
        )
    # Только на бирже (ручные и т.д.)
    for o in reconciliation.get("only_on_exchange", []):
        raw = o.get("raw") or {}
        merged.append(
            TradeSummary(
                symbol=o["symbol"],
                exit_timestamp=o["exit_timestamp"],
                pnl=o["pnl"],
                entry_price=o.get("entry_price") or 0.0,
                exit_price=o.get("exit_price") or 0.0,
                position_size_usdt=o.get("position_size_usdt"),
                direction=o.get("direction") or "LONG",
                source="exchange",
                raw=raw,
            )
        )
    # Только в БД ботов (нет на бирже)
    for o in reconciliation.get("only_in_bots", []):
        bot = o.get("bot")
        if bot is not None:
            merged.append(bot)
    return merged


def run_full_analytics(
    exchange_trades: Optional[List[Dict[str, Any]]] = None,
    bot_trades: Optional[List[Dict[str, Any]]] = None,
    load_bot_trades_from_db: bool = True,
    load_exchange_from_api: bool = False,
    exchange_instance: Any = None,
    exchange_period: str = "all",
    bots_db_limit: Optional[int] = 50000,
) -> Dict[str, Any]:
    """
    Запускает полную аналитику торговли.

    Источники данных:
    - exchange_trades: если передан, используется как сделки с биржи
    - bot_trades: если передан, используется как сделки ботов
    - load_bot_trades_from_db=True: подгружает из bots_data.db (bot_trades_history)
    - load_exchange_from_api=True: требует exchange_instance, вызывает get_closed_pnl(period=exchange_period)

    Returns:
        Словарь с ключами: exchange_analytics, bot_analytics, reconciliation, summary, generated_at.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    # Загрузка сделок ботов из БД
    if bot_trades is None and load_bot_trades_from_db:
        try:
            from bot_engine.bots_database import get_bots_database
            db = get_bots_database()
            bot_trades = db.get_bot_trades_history(
                status="CLOSED",
                limit=bots_db_limit,
            )
            if not bot_trades:
                bot_trades = []
        except Exception as e:
            logger.warning("Не удалось загрузить сделки ботов из БД: %s", e)
            bot_trades = []

    if bot_trades is None:
        bot_trades = []

    # Загрузка сделок с биржи
    if exchange_trades is None and load_exchange_from_api and exchange_instance is not None:
        try:
            if hasattr(exchange_instance, "get_closed_pnl"):
                exchange_trades = exchange_instance.get_closed_pnl(
                    sort_by="time",
                    period=exchange_period,
                ) or []
            else:
                exchange_trades = []
        except Exception as e:
            logger.warning("Не удалось загрузить сделки с биржи: %s", e)
            exchange_trades = []

    if exchange_trades is None:
        exchange_trades = []

    ex_summaries = exchange_trades_to_summaries(exchange_trades)
    bot_summaries = bot_trades_to_summaries(bot_trades)

    exchange_analytics = analyze_exchange_trades(ex_summaries) if ex_summaries else {}
    reconciliation = {}
    if ex_summaries and bot_summaries:
        reconciliation = reconcile_trades(ex_summaries, bot_summaries)

    # Если была сверка с биржей — используем объединённые данные (цены и PnL с биржи для совпавших)
    summaries_for_analytics = bot_summaries
    if reconciliation:
        merged = _build_merged_trade_summaries(reconciliation, ex_summaries, bot_summaries)
        if merged:
            summaries_for_analytics = merged

    bot_analytics = analyze_bot_trades(summaries_for_analytics) if summaries_for_analytics else {}

    summary = {
        "exchange_trades_count": len(ex_summaries),
        "bot_trades_count": len(bot_summaries),
        "reconciliation_matched": reconciliation.get("matched_count", 0),
        "reconciliation_only_exchange": reconciliation.get("only_on_exchange_count", 0),
        "reconciliation_only_bots": reconciliation.get("only_in_bots_count", 0),
        "reconciliation_pnl_mismatches": reconciliation.get("pnl_mismatch_count", 0),
        "bot_win_rate_pct": bot_analytics.get("win_rate_pct"),
        "bot_total_pnl_usdt": bot_analytics.get("total_pnl_usdt"),
        "exchange_total_pnl_usdt": exchange_analytics.get("total_pnl_usdt"),
    }

    return {
        "generated_at": generated_at,
        "exchange_analytics": exchange_analytics,
        "bot_analytics": bot_analytics,
        "reconciliation": reconciliation,
        "summary": summary,
    }


def get_analytics_for_ai(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает из полного отчёта структурированные данные для AI модуля:
    - проблемы (ошибки, расхождения, серии убытков, неудачные монеты/настройки)
    - метрики по причинам закрытия и символам
    - рекомендации (текстовые тезисы)
    """
    bot = report.get("bot_analytics") or {}
    recon = report.get("reconciliation") or {}
    problems: List[str] = []
    recommendations: List[str] = []

    # Неудачные монеты и настройки по RSI/тренду
    unsuccessful_coins = bot.get("unsuccessful_coins", [])
    unsuccessful_settings = bot.get("unsuccessful_settings", [])
    if unsuccessful_coins:
        problems.append(
            f"Неудачные монеты (PnL < 0 или Win Rate < 45%): {[c['symbol'] for c in unsuccessful_coins]}. "
            "Рекомендуется скорректировать настройки или исключить из списка."
        )
        recommendations.append(
            "По неудачным монетам проверить bad_rsi_ranges и bad_trends в unsuccessful_settings: "
            "избегать входов в указанных диапазонах RSI и при указанных трендах."
        )
    for us in unsuccessful_settings:
        bad_rsi = us.get("bad_rsi_ranges", [])
        bad_trends = us.get("bad_trends", [])
        if bad_rsi or bad_trends:
            rec_parts = [f"Монета {us.get('symbol')}:"]
            if bad_rsi:
                rec_parts.append("неудачные RSI-диапазоны " + ", ".join(r["rsi_range"] for r in bad_rsi))
            if bad_trends:
                rec_parts.append("неудачные тренды " + ", ".join(str(t["trend"]) for t in bad_trends))
            recommendations.append(" ".join(rec_parts))

    if bot.get("possible_errors_count", 0) > 0:
        problems.append(
            f"Обнаружено потенциальных ошибок в сделках: {bot['possible_errors_count']}. "
            "Проверьте close_reason и extra_data."
        )
        recommendations.append("Проанализировать записи possible_errors и при необходимости скорректировать логику закрытия.")

    max_losses = (bot.get("consecutive_series") or {}).get("max_consecutive_losses", 0)
    if max_losses >= 3:
        problems.append(f"Серия убыточных сделок подряд: до {max_losses}. Возможен переторг или неблагоприятный режим рынка.")
        recommendations.append("Рассмотреть фильтр по сериям убытков (пауза или уменьшение размера).")

    dd = (bot.get("drawdown") or {}).get("max_drawdown_usdt")
    if dd is not None and dd > 10:
        problems.append(f"Максимальная просадка по эквити: {dd} USDT.")
        recommendations.append("Проверить настройки Stop Loss и макс. просадки.")

    if recon.get("only_on_exchange_count", 0) > 5:
        problems.append(
            f"На бирже {recon['only_on_exchange_count']} сделок без соответствия в истории ботов. "
            "Часть сделок могла быть закрыта вручную или другим клиентом."
        )
        recommendations.append("Синхронизировать историю: rebuild_bot_history_from_exchange или импорт закрытых PnL.")

    if recon.get("pnl_mismatch_count", 0) > 0:
        problems.append(f"Расхождение PnL между биржей и историей ботов: {recon['pnl_mismatch_count']} сделок.")
        recommendations.append("Проверить округление и комиссии при сохранении сделок.")

    by_reason = bot.get("by_close_reason") or {}
    stop_loss_count = by_reason.get("STOP_LOSS", {}).get("count", 0) + by_reason.get("Stop Loss", {}).get("count", 0)
    total_closed = bot.get("total_trades", 0)
    if total_closed and stop_loss_count / total_closed > 0.5:
        recommendations.append("Большая доля закрытий по Stop Loss — рассмотреть ослабление SL или улучшение фильтров входа.")

    return {
        "problems": problems,
        "recommendations": recommendations,
        "metrics": {
            "win_rate_pct": bot.get("win_rate_pct"),
            "total_pnl_usdt": bot.get("total_pnl_usdt"),
            "max_consecutive_losses": (bot.get("consecutive_series") or {}).get("max_consecutive_losses"),
            "max_drawdown_usdt": (bot.get("drawdown") or {}).get("max_drawdown_usdt"),
            "by_close_reason_counts": {k: v.get("count", 0) for k, v in by_reason.items()},
        },
        "unsuccessful_coins": unsuccessful_coins,
        "unsuccessful_settings": unsuccessful_settings,
        "successful_coins": bot.get("successful_coins", []),
        "successful_settings": bot.get("successful_settings", []),
        "generated_at": report.get("generated_at"),
    }
