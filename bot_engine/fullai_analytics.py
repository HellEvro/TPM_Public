# -*- coding: utf-8 -*-
"""
Аналитика FullAI: отдельная файловая БД для событий ПРИИ (решения, блокировки, виртуальные/реальные сделки).
Не увеличивает размер основной БД. Путь: data/fullai_analytics.db
"""
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('BOTS')

# Типы событий для event_type
EVENT_REAL_OPEN = 'real_open'
EVENT_VIRTUAL_OPEN = 'virtual_open'
EVENT_REAL_CLOSE = 'real_close'
EVENT_VIRTUAL_CLOSE = 'virtual_close'
EVENT_BLOCKED = 'blocked'           # вход заблокирован (напр. loss_reentry)
EVENT_REFUSED = 'refused'           # ИИ отказал во входе
EVENT_PARAMS_CHANGE = 'params_change'
EVENT_ROUND_SUCCESS = 'round_success'  # N виртуальных успешны → разрешена реальная
EVENT_EXIT_HOLD = 'exit_hold'       # ИИ решил не закрывать (держать позицию)

_db_path: Optional[Path] = None
_lock = threading.Lock()
_logged_path = False


def _get_project_root() -> Path:
    """Корень проекта (как в bots_database), чтобы БД была в той же data/, что и bots_data.db."""
    current = Path(__file__).resolve()
    for parent in [current.parent.parent] + list(current.parents):
        if parent and (parent / 'bots.py').exists() and (parent / 'bot_engine').exists():
            return parent
    try:
        return current.parents[1]
    except IndexError:
        return current.parent


def _get_db_path() -> Path:
    global _db_path
    if _db_path is not None:
        return _db_path
    root = _get_project_root()
    data_dir = root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    _db_path = data_dir / 'fullai_analytics.db'
    return _db_path


def _ensure_db_exists() -> Path:
    """Создать файл БД и схему, если их ещё нет (чтобы файл был виден в data/ даже при 0 событий)."""
    path = _get_db_path()
    if path.exists():
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                _init_schema(conn)
                conn.commit()
            finally:
                conn.close()
        logger.info("FullAI analytics: создана БД %s", path)
    except Exception as e:
        logger.warning("FullAI analytics _ensure_db_exists: %s", e)
    return path


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fullai_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ts_iso TEXT,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            direction TEXT,
            is_virtual INTEGER,
            confidence REAL,
            reason TEXT,
            pnl_percent REAL,
            extra_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fullai_ts ON fullai_events(ts);
        CREATE INDEX IF NOT EXISTS idx_fullai_symbol ON fullai_events(symbol);
        CREATE INDEX IF NOT EXISTS idx_fullai_type ON fullai_events(event_type);
    """)


def append_event(
    symbol: str,
    event_type: str,
    direction: Optional[str] = None,
    is_virtual: Optional[bool] = None,
    confidence: Optional[float] = None,
    reason: Optional[str] = None,
    pnl_percent: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Добавить одно событие FullAI в аналитику."""
    import time
    ts = time.time()
    ts_iso = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(ts))
    symbol = (symbol or '').upper()
    if not symbol and event_type not in (EVENT_PARAMS_CHANGE, EVENT_ROUND_SUCCESS):
        return
    extra_json = json.dumps(extra, ensure_ascii=False) if extra else None
    global _logged_path
    path = _get_db_path()
    try:
        with _lock:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                _init_schema(conn)
                conn.execute(
                    """INSERT INTO fullai_events (ts, ts_iso, symbol, event_type, direction, is_virtual, confidence, reason, pnl_percent, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ts, ts_iso, symbol or '', event_type, direction, 1 if is_virtual else 0 if is_virtual is False else None,
                     confidence, reason, pnl_percent, extra_json),
                )
                conn.commit()
                if not _logged_path:
                    _logged_path = True
                    logger.info("FullAI analytics: запись в БД %s (событие %s, %s)", path, event_type, symbol or "-")
            finally:
                conn.close()
    except Exception as e:
        logger.warning("FullAI analytics append_event (путь %s): %s", path, e)


def get_events(
    symbol: Optional[str] = None,
    event_type: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Список событий с фильтрами."""
    path = _ensure_db_exists()
    if not path.exists():
        return []
    conditions = []
    args: List[Any] = []
    if symbol:
        conditions.append("symbol = ?")
        args.append(symbol.upper())
    if event_type:
        conditions.append("event_type = ?")
        args.append(event_type)
    if from_ts is not None:
        conditions.append("ts >= ?")
        args.append(from_ts)
    if to_ts is not None:
        conditions.append("ts <= ?")
        args.append(to_ts)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    args.append(limit)
    try:
        with _lock:
            conn = sqlite3.connect(str(path), timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    f"SELECT id, ts, ts_iso, symbol, event_type, direction, is_virtual, confidence, reason, pnl_percent, extra_json FROM fullai_events{where} ORDER BY ts DESC LIMIT ?",
                    args,
                ).fetchall()
                out = []
                for r in rows:
                    rec = dict(r)
                    rec['is_virtual'] = bool(rec['is_virtual']) if rec['is_virtual'] is not None else None
                    if rec.get('extra_json'):
                        try:
                            rec['extra'] = json.loads(rec['extra_json'])
                        except Exception:
                            pass
                    if 'extra_json' in rec:
                        del rec['extra_json']
                    out.append(rec)
                return out
            finally:
                conn.close()
    except Exception as e:
        logger.warning("FullAI analytics get_events: %s", e)
        return []


def get_summary(
    symbol: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Сводка по событиям FullAI за период."""
    path = _ensure_db_exists()
    if not path.exists():
        return _empty_summary()
    conditions = []
    args: List[Any] = []
    if symbol:
        conditions.append("symbol = ?")
        args.append(symbol.upper())
    if from_ts is not None:
        conditions.append("ts >= ?")
        args.append(from_ts)
    if to_ts is not None:
        conditions.append("ts <= ?")
        args.append(to_ts)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    and_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
    try:
        with _lock:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                base = f"SELECT event_type, COUNT(*) as cnt FROM fullai_events {where} GROUP BY event_type"
                rows = conn.execute(base, args).fetchall()
                counts = {r[0]: r[1] for r in rows}
                # Реальные закрытия с PnL
                sql_pnl = "SELECT pnl_percent FROM fullai_events WHERE event_type = ? AND pnl_percent IS NOT NULL" + and_extra
                args_pnl = [EVENT_REAL_CLOSE] + args if conditions else [EVENT_REAL_CLOSE]
                pnl_rows = conn.execute(sql_pnl, args_pnl).fetchall()
                real_pnls = [r[0] for r in pnl_rows if r[0] is not None]
                real_wins = sum(1 for p in real_pnls if p >= 0)
                real_losses = sum(1 for p in real_pnls if p < 0)
                sql_vc = "SELECT extra_json FROM fullai_events WHERE event_type = ?" + and_extra
                args_vc = [EVENT_VIRTUAL_CLOSE] + args if conditions else [EVENT_VIRTUAL_CLOSE]
                virtual_close_rows = conn.execute(sql_vc, args_vc).fetchall()
                virtual_ok = 0
                virtual_fail = 0
                for r in virtual_close_rows:
                    if r[0]:
                        try:
                            ex = json.loads(r[0])
                            if ex.get('success'):
                                virtual_ok += 1
                            else:
                                virtual_fail += 1
                        except Exception:
                            pass
                return {
                    'counts': counts,
                    'real_open': counts.get(EVENT_REAL_OPEN, 0),
                    'virtual_open': counts.get(EVENT_VIRTUAL_OPEN, 0),
                    'real_close': counts.get(EVENT_REAL_CLOSE, 0),
                    'virtual_close': counts.get(EVENT_VIRTUAL_CLOSE, 0),
                    'blocked': counts.get(EVENT_BLOCKED, 0),
                    'refused': counts.get(EVENT_REFUSED, 0),
                    'params_change': counts.get(EVENT_PARAMS_CHANGE, 0),
                    'round_success': counts.get(EVENT_ROUND_SUCCESS, 0),
                    'exit_hold': counts.get(EVENT_EXIT_HOLD, 0),
                    'real_wins': real_wins,
                    'real_losses': real_losses,
                    'real_total': len(real_pnls),
                    'virtual_ok': virtual_ok,
                    'virtual_fail': virtual_fail,
                    'virtual_total': virtual_ok + virtual_fail,
                }
            finally:
                conn.close()
    except Exception as e:
        logger.warning("FullAI analytics get_summary: %s", e)
        return _empty_summary()


def _empty_summary() -> Dict[str, Any]:
    return {
        'counts': {},
        'real_open': 0, 'virtual_open': 0, 'real_close': 0, 'virtual_close': 0,
        'blocked': 0, 'refused': 0, 'params_change': 0, 'round_success': 0, 'exit_hold': 0,
        'real_wins': 0, 'real_losses': 0, 'real_total': 0,
        'virtual_ok': 0, 'virtual_fail': 0, 'virtual_total': 0,
    }


def get_db_info() -> Dict[str, Any]:
    """Путь к БД и общее число событий (без фильтра по периоду) — для диагностики."""
    path = _ensure_db_exists()
    total = 0
    if path.exists():
        try:
            with _lock:
                conn = sqlite3.connect(str(path), timeout=5)
                try:
                    row = conn.execute("SELECT COUNT(*) FROM fullai_events").fetchone()
                    total = row[0] if row else 0
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("FullAI analytics get_db_info: %s", e)
    return {'db_path': str(path), 'total_events': total}
