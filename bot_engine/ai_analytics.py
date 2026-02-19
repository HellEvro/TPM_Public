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
    """Логирует закрытие позиции."""
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
        except Exception as e:
            logger.debug("get_ai_analytics_context: report %s", e)
            result["problems"] = []
            result["recommendations"] = []
            result["metrics"] = {}
    return result
