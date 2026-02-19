# -*- coding: utf-8 -*-
"""
Единая аналитика для ИИ: сбор событий при каждой сделке, решении, действии бота.

События пишутся в ai_data.db -> ai_analytics_events.
ИИ использует эти данные для обучения на ошибках и принятия решений.

Типы событий (event_type):
- TRADE_OPEN, TRADE_CLOSE — открытие/закрытие позиции
- AI_DECISION_ENTER, AI_DECISION_EXIT, AI_DECISION_HOLD, AI_DECISION_REFUSE — решение ИИ
- BOT_ACTION — действие бота (открыл, закрыл по SL/TP/RSI и т.д.)
- ENTRY_BLOCKED, ENTRY_REFUSED — вход заблокирован/отклонён
- PARAMS_CHANGE, ROUND_SUCCESS, VIRTUAL_OPEN, VIRTUAL_CLOSE — события FullAI
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AI.Analytics")

_EVENT_TYPES = frozenset({
    "TRADE_OPEN", "TRADE_CLOSE",
    "AI_DECISION_ENTER", "AI_DECISION_EXIT", "AI_DECISION_HOLD", "AI_DECISION_REFUSE",
    "BOT_ACTION", "ENTRY_BLOCKED", "ENTRY_REFUSED",
    "PARAMS_CHANGE", "ROUND_SUCCESS", "VIRTUAL_OPEN", "VIRTUAL_CLOSE",
    "REAL_OPEN", "REAL_CLOSE", "EXIT_HOLD",
})


def _get_ai_db():
    try:
        from bot_engine.ai.ai_database import get_ai_database
        return get_ai_database()
    except Exception as e:
        logger.debug("ai_analytics: нет доступа к AI БД: %s", e)
        return None


def log_event(
    event_type: str,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    source: Optional[str] = None,
    reason: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирует событие аналитики для ИИ. Вызывается при каждой сделке, решении, действии.
    """
    ts = time.time()
    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    data_json = json.dumps(data, ensure_ascii=False) if data else None
    db = _get_ai_db()
    if not db:
        return
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO ai_analytics_events (ts, ts_iso, event_type, symbol, direction, source, reason, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, ts_iso, event_type, (symbol or "").upper() if symbol else None,
                 direction, source, reason, data_json, ts_iso),
            )
            conn.commit()
    except Exception as e:
        logger.debug("ai_analytics log_event: %s", e)


def log_trade_open(
    symbol: str,
    direction: str,
    entry_price: float,
    position_size_usdt: Optional[float] = None,
    entry_rsi: Optional[float] = None,
    entry_trend: Optional[str] = None,
    source: str = "BOT",
    **extra,
) -> None:
    """Логирует открытие позиции."""
    log_event(
        "TRADE_OPEN",
        symbol=symbol,
        direction=direction,
        source=source,
        data={
            "entry_price": entry_price,
            "position_size_usdt": position_size_usdt,
            "entry_rsi": entry_rsi,
            "entry_trend": entry_trend,
            **extra,
        },
    )


def log_trade_close(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    reason: Optional[str] = None,
    entry_rsi: Optional[float] = None,
    exit_rsi: Optional[float] = None,
    source: str = "BOT",
    **extra,
) -> None:
    """Логирует закрытие позиции. Сбрасывает кеш аналитики для актуальности следующего решения."""
    invalidate_analytics_cache()
    log_event(
        "TRADE_CLOSE",
        symbol=symbol,
        direction=direction,
        source=source,
        reason=reason,
        data={
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "entry_rsi": entry_rsi,
            "exit_rsi": exit_rsi,
            **extra,
        },
    )


def log_ai_decision(
    event_type: str,
    symbol: str,
    direction: Optional[str],
    decision: str,
    reason: Optional[str] = None,
    confidence: Optional[float] = None,
    **extra,
) -> None:
    """Логирует решение ИИ (ENTER/EXIT/HOLD/REFUSE)."""
    log_event(
        event_type,
        symbol=symbol,
        direction=direction,
        source="AI",
        reason=reason,
        data={"decision": decision, "confidence": confidence, **extra},
    )


def log_bot_action(
    symbol: str,
    action: str,
    direction: Optional[str] = None,
    reason: Optional[str] = None,
    **extra,
) -> None:
    """Логирует действие бота (открыл, закрыл по SL/TP/RSI и т.д.)."""
    log_event(
        "BOT_ACTION",
        symbol=symbol,
        direction=direction,
        source="BOT",
        reason=reason,
        data={"action": action, **extra},
    )


def get_events(
    symbol: Optional[str] = None,
    event_type: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Возвращает события аналитики для ИИ."""
    db = _get_ai_db()
    if not db:
        return []
    conditions = []
    params: List[Any] = []
    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol.upper())
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if from_ts is not None:
        conditions.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        conditions.append("ts <= ?")
        params.append(to_ts)
    where = (" AND " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    try:
        with db._get_connection() as conn:
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            cursor = conn.cursor()
            rows = cursor.execute(
                f"SELECT * FROM ai_analytics_events{where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
            out = []
            for r in rows:
                rec = dict(r)
                if rec.get("data_json"):
                    try:
                        rec["data"] = json.loads(rec["data_json"])
                    except Exception:
                        pass
                if "data_json" in rec:
                    del rec["data_json"]
                out.append(rec)
            return out
    except Exception as e:
        logger.debug("ai_analytics get_events: %s", e)
        return []


_analytics_cache: Dict[str, Any] = {}
_analytics_cache_ts: float = 0.0
_ANALYTICS_CACHE_TTL_SEC = 300  # 5 минут


def _get_cached_analytics_for_entry(symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Кеширует контекст аналитики с TTL. Используется ИИ перед решением о входе.
    Сначала пробует память, потом БД (ai_experience_snapshot), затем полный расчёт.
    Опыт сохраняется в БД — ИИ не переобучается с нуля.
    """
    global _analytics_cache, _analytics_cache_ts
    now = time.time()
    if now - _analytics_cache_ts < _ANALYTICS_CACHE_TTL_SEC and _analytics_cache:
        return _analytics_cache
    try:
        db = _get_ai_db()
        if db and hasattr(db, "get_ai_experience_snapshot"):
            snap = db.get_ai_experience_snapshot()
            if snap and snap.get("generated_at"):
                try:
                    gen_str = str(snap["generated_at"]).replace("Z", "+00:00").replace(" ", "T")
                    gen = datetime.fromisoformat(gen_str)
                    age_sec = now - gen.timestamp()
                    if age_sec < 86400:
                        ctx = {
                            "problems": snap.get("problems", []),
                            "recommendations": snap.get("recommendations", []),
                            "metrics": snap.get("metrics", {}),
                            "unsuccessful_coins": snap.get("unsuccessful_coins", []),
                            "unsuccessful_settings": snap.get("unsuccessful_settings", []),
                        }
                        _analytics_cache = ctx
                        _analytics_cache_ts = now
                        return ctx
                except Exception:
                    pass
        ctx = get_ai_analytics_context(symbol=symbol, hours_back=168, include_report=True)
        _analytics_cache = ctx
        _analytics_cache_ts = now
        return ctx
    except Exception as e:
        logger.debug("_get_cached_analytics_for_entry: %s", e)
        if _analytics_cache:
            return _analytics_cache
        try:
            db = _get_ai_db()
            if db and hasattr(db, "get_ai_experience_snapshot"):
                snap = db.get_ai_experience_snapshot()
                if snap:
                    return {
                        "problems": snap.get("problems", []),
                        "recommendations": snap.get("recommendations", []),
                        "metrics": snap.get("metrics", {}),
                        "unsuccessful_coins": snap.get("unsuccessful_coins", []),
                        "unsuccessful_settings": snap.get("unsuccessful_settings", []),
                    }
        except Exception:
            pass
        return None


def invalidate_analytics_cache() -> None:
    """Сбрасывает кеш (вызывать после закрытия сделки)."""
    global _analytics_cache_ts
    _analytics_cache_ts = 0.0


def apply_analytics_to_entry_decision(
    symbol: str,
    direction: str,
    rsi: Optional[float],
    trend: Optional[str],
    base_allowed: bool,
    base_confidence: float,
    base_reason: str,
) -> tuple:
    """
    Применяет аналитику к решению о входе: блокирует/снижает уверенность на основе прошлых ошибок.
    Возвращает (allowed, confidence, reason).
    """
    ctx = _get_cached_analytics_for_entry(symbol)
    if not ctx:
        return (base_allowed, base_confidence, base_reason)
    allowed = base_allowed
    confidence = base_confidence
    reason = base_reason
    problems = ctx.get("problems") or []
    metrics = ctx.get("metrics") or {}
    # unsuccessful_coins и unsuccessful_settings приходят из report, а не напрямую
    # get_ai_analytics_context возвращает problems, recommendations, metrics
    # Нужно передать unsuccessful_coins - они в ai_block. Проверим структуру get_ai_analytics_context
    # - она вызывает get_analytics_for_ai который возвращает unsuccessful_coins, unsuccessful_settings
    # Но get_ai_analytics_context не добавляет их в result! Добавлю.
    unsuccessful_coins = ctx.get("unsuccessful_coins") or []
    unsuccessful_settings = ctx.get("unsuccessful_settings") or []
    sym_upper = (symbol or "").upper()
    for uc in unsuccessful_coins:
        if (uc.get("symbol") or "").upper() == sym_upper:
            wr = uc.get("win_rate_pct") or 0
            pnl = uc.get("pnl_usdt") or 0
            if wr < 35 or pnl < -50:
                allowed = False
                reason = f"Аналитика: монета неудачная (Win Rate {wr}%, PnL {pnl} USDT). Блокировка входа."
                return (allowed, 0.0, reason)
            if allowed:
                confidence = max(0, confidence - 0.15)
                reason = base_reason + f" | Снижена уверенность: монета с низким WR ({wr}%)"
    for us in unsuccessful_settings:
        if (us.get("symbol") or "").upper() != sym_upper:
            continue
        bad_rsi = us.get("bad_rsi_ranges") or []
        bad_trends = us.get("bad_trends") or []
        if rsi is not None:
            for br in bad_rsi:
                rng = br.get("rsi_range", "")
                if "-" in rng:
                    try:
                        part = rng.split("(")[0].strip()
                        parts = part.split("-")
                        if len(parts) >= 2:
                            lo, hi = int(parts[0]), int(parts[1])
                            if lo <= rsi <= hi:
                                allowed = False
                                reason = f"Аналитика: RSI {rsi:.1f} в неудачном диапазоне {rng}"
                                return (allowed, 0.0, reason)
                    except (ValueError, IndexError, TypeError):
                        pass
        if trend and bad_trends:
            trend_upper = str(trend).upper()
            for bt in bad_trends:
                t = str(bt.get("trend", "")).upper()
                if t and trend_upper == t:
                    confidence = max(0, confidence - 0.2)
                    reason = base_reason + f" | Тренд {trend} — неудачный по аналитике"
                    break
    if problems and "серия убытков" in " ".join(problems).lower() and confidence > 0.5:
        confidence = min(confidence, 0.6)
        reason = base_reason + " | Учтена серия убыточных сделок"
    return (allowed, confidence, reason)


def get_ai_analytics_context(
    symbol: Optional[str] = None,
    hours_back: float = 24.0,
    include_report: bool = True,
) -> Dict[str, Any]:
    """
    Возвращает контекст аналитики для ИИ: сводка, последние события, проблемы и рекомендации.
    Используется ИИ при принятии решений и для обучения на ошибках.
    include_report=False — только события из ai_analytics_events (быстро, без вызова биржи).
    """
    import time as _time
    to_ts = _time.time()
    from_ts = to_ts - hours_back * 3600
    events = get_events(symbol=symbol, from_ts=from_ts, to_ts=to_ts, limit=200)
    result: Dict[str, Any] = {
        "events_count": len(events),
        "events_sample": events[:50],
        "from_ts": from_ts,
        "to_ts": to_ts,
    }
    if include_report:
        try:
            from bot_engine.trading_analytics import run_full_analytics, get_analytics_for_ai
            exchange_instance = None
            try:
                from app.config import EXCHANGES, ACTIVE_EXCHANGE
                from exchanges.exchange_factory import ExchangeFactory
                cfg = EXCHANGES.get(ACTIVE_EXCHANGE, {})
                if cfg and cfg.get("enabled", True) and cfg.get("api_key") and cfg.get("api_secret"):
                    exchange_instance = ExchangeFactory.create_exchange(
                        ACTIVE_EXCHANGE,
                        cfg.get("api_key"),
                        cfg.get("api_secret"),
                        cfg.get("passphrase"),
                    )
            except Exception:
                pass
            report = run_full_analytics(
                load_bot_trades_from_db=True,
                load_exchange_from_api=exchange_instance is not None,
                exchange_instance=exchange_instance,
                exchange_period="all",
            )
            ai_block = get_analytics_for_ai(report)
            result["problems"] = ai_block.get("problems", [])
            result["recommendations"] = ai_block.get("recommendations", [])
            result["metrics"] = ai_block.get("metrics", {})
            result["unsuccessful_coins"] = ai_block.get("unsuccessful_coins", [])
            result["unsuccessful_settings"] = ai_block.get("unsuccessful_settings", [])
            try:
                db = _get_ai_db()
                if db and hasattr(db, "save_ai_experience_snapshot"):
                    db.save_ai_experience_snapshot(
                        unsuccessful_coins=result["unsuccessful_coins"],
                        unsuccessful_settings=result["unsuccessful_settings"],
                        metrics=result["metrics"],
                        problems=result["problems"],
                        recommendations=result["recommendations"],
                    )
            except Exception as _s:
                logger.debug("save_ai_experience_snapshot: %s", _s)
        except Exception as e:
            logger.debug("get_ai_analytics_context: report %s", e)
            result["problems"] = []
            result["recommendations"] = []
            result["metrics"] = {}
    return result
