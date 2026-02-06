"""
ExitScam Learner — подбор параметров ExitScam по истории свечей для каждой монеты.

Идея: по распределению изменений цены (одна свеча и суммарно N свечей) определяем пороги,
выше которых движение считаем «скамом» (памп/дамп). Параметры сохраняются в индивидуальные
настройки монеты.

- Одна свеча: body_pct = |close - open| / open * 100. Порог = перцентиль (например 95%).
- N свечей: total_pct = |last_close - first_open| / first_open * 100 по скользящим окнам.
  N и порог подбираются по истории.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("AI.ExitScamLearner")

# Ограничения параметров (как в фильтре и UI)
# На 1m ТФ движения могут быть 0.01% и меньше — минимумы позволяют сырым перцентилям пройти
MIN_SINGLE_PERCENT = 0.01
MAX_SINGLE_PERCENT = 60.0
MIN_MULTI_COUNT = 2
MAX_MULTI_COUNT = 12
MIN_MULTI_PERCENT = 0.01
MAX_MULTI_PERCENT = 150.0
MIN_CANDLES_LOOKBACK = 8
MAX_CANDLES_LOOKBACK = 30
DEFAULT_AGGRESSIVENESS = "normal"  # normal | conservative | aggressive


def _normalize_candle(candle: Dict[str, Any]) -> Dict[str, float]:
    return {
        "open": float(candle.get("open", 0) or 0),
        "close": float(candle.get("close", 0) or 0),
        "high": float(candle.get("high", candle.get("close", 0)) or 0),
        "low": float(candle.get("low", candle.get("close", 0)) or 0),
    }


def _body_percent(candle: Dict[str, float]) -> Optional[float]:
    o, c = candle.get("open"), candle.get("close")
    if o is None or c is None or o <= 0:
        return None
    return abs((c - o) / o) * 100.0


def _rolling_total_percent(candles: List[Dict[str, float]], n: int) -> List[float]:
    """Суммарное изменение в % за последние n свечей: |last_close - first_open|/first_open*100."""
    out: List[float] = []
    for i in range(len(candles) - n + 1):
        window = candles[i : i + n]
        first_open = window[0].get("open")
        last_close = window[-1].get("close")
        if first_open is None or last_close is None or first_open <= 0:
            continue
        pct = abs((last_close - first_open) / first_open) * 100.0
        out.append(pct)
    return out


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    if f >= len(sorted_values) - 1:
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[f + 1] - sorted_values[f])


def compute_exit_scam_params(
    candles: List[Dict[str, Any]],
    aggressiveness: str = DEFAULT_AGGRESSIVENESS,
    min_candles: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    По истории свечей вычисляет параметры ExitScam для этой монеты.

    Args:
        candles: список свечей (open, close, high, low).
        aggressiveness: 'conservative' (99th — реже блокируем), 'normal' (95th), 'aggressive' (90th — чаще блокируем).
        min_candles: минимум свечей для анализа.

    Returns:
        (params_dict, stats_dict).
        params_dict: exit_scam_candles, exit_scam_single_candle_percent,
                     exit_scam_multi_candle_count, exit_scam_multi_candle_percent.
        stats_dict: диагностика (перцентили, выбранный N и т.д.).
    """
    normalized = [_normalize_candle(c) for c in candles]
    if len(normalized) < min_candles:
        logger.warning(
            "ExitScam learner: недостаточно свечей %s (нужно минимум %s)",
            len(normalized),
            min_candles,
        )
        return _default_params(), {"error": "insufficient_candles", "count": len(normalized)}

    # Перцентиль в зависимости от агрессивности
    if aggressiveness == "conservative":
        pct_level = 99.0
    elif aggressiveness == "aggressive":
        pct_level = 90.0
    else:
        pct_level = 95.0

    # 1) Одна свеча: распределение body %
    body_pcts = []
    for c in normalized:
        bp = _body_percent(c)
        if bp is not None:
            body_pcts.append(bp)
    if not body_pcts:
        return _default_params(), {"error": "no_body_pct"}
    body_pcts_sorted = sorted(body_pcts)
    single_raw = _percentile(body_pcts_sorted, pct_level)
    single_candle_percent = round(
        max(MIN_SINGLE_PERCENT, min(MAX_SINGLE_PERCENT, single_raw)), 1
    )

    # 2) Суммарное изменение за N свечей: перебираем N
    best_n = 4
    best_multi_pct = MIN_MULTI_PERCENT
    multi_stats: Dict[int, float] = {}
    for n in range(MIN_MULTI_COUNT, min(MAX_MULTI_COUNT + 1, len(normalized) // 2)):
        rolling = _rolling_total_percent(normalized, n)
        if len(rolling) < 10:
            continue
        rolling_sorted = sorted(rolling)
        multi_pct_raw = _percentile(rolling_sorted, pct_level)
        multi_pct = round(
            max(MIN_MULTI_PERCENT, min(MAX_MULTI_PERCENT, multi_pct_raw)), 1
        )
        multi_stats[n] = multi_pct
        # Выбираем N, при котором 95th максимальный — самые сильные движения видны за N свечей
        if multi_pct >= best_multi_pct:
            best_multi_pct = multi_pct
            best_n = n

    multi_candle_count = max(MIN_MULTI_COUNT, min(MAX_MULTI_COUNT, best_n))
    multi_candle_percent = best_multi_pct

    # 3) Сколько свечей смотреть назад: достаточно для single + multi проверок
    exit_scam_candles = max(
        MIN_CANDLES_LOOKBACK,
        min(MAX_CANDLES_LOOKBACK, multi_candle_count * 3, len(normalized)),
    )

    params = {
        "exit_scam_candles": exit_scam_candles,
        "exit_scam_single_candle_percent": single_candle_percent,
        "exit_scam_multi_candle_count": multi_candle_count,
        "exit_scam_multi_candle_percent": multi_candle_percent,
    }
    stats = {
        "aggressiveness": aggressiveness,
        "percentile_used": pct_level,
        "single_raw_pct": round(single_raw, 2),
        "multi_per_n": multi_stats,
        "chosen_n": multi_candle_count,
        "candles_analyzed": len(normalized),
    }
    logger.info(
        "ExitScam learner: single=%s%%, multi N=%s %s%%, lookback=%s"
        % (single_candle_percent, multi_candle_count, multi_candle_percent, exit_scam_candles),
    )
    return params, stats


def _default_params() -> Dict[str, Any]:
    return {
        "exit_scam_candles": 12,
        "exit_scam_single_candle_percent": 15.0,
        "exit_scam_multi_candle_count": 4,
        "exit_scam_multi_candle_percent": 50.0,
    }
