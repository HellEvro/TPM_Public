# -*- coding: utf-8 -*-
"""
AI-анализатор выводов по сделкам.

Генерирует конкретные, разнообразные выводы на основе:
- PnL, ROI, причины закрытия
- RSI на входе/выходе, тренд
- Направления (LONG/SHORT)

Интерфейс готов для подключения PyTorch-модели (дообучаемой на истории).
"""

import logging
import random
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger('AI.TradeConclusionAnalyzer')

# Флаг: использовать ли PyTorch-модель при наличии
USE_PYTORCH_IF_AVAILABLE = True
_MODEL_PATH = Path('data/ai/trade_conclusion_model.pt')


_analyzer_instance = None


def analyze_trade_conclusion(trade: Dict[str, Any]) -> str:
    """
    Генерирует детальный вывод по сделке для колонки «Вывод».

    Args:
        trade: dict с pnl, roi, close_reason, direction, entry_rsi, exit_rsi, entry_trend, exit_trend, symbol

    Returns:
        Строка вывода (1-2 предложения, конкретный анализ)
    """
    global _analyzer_instance
    try:
        if _analyzer_instance is None:
            _analyzer_instance = _get_analyzer()
        return _analyzer_instance.analyze(trade)
    except Exception as e:
        logger.debug("TradeConclusionAnalyzer: %s", e)
        return _fallback_conclusion(trade)


def _get_analyzer() -> 'TradeConclusionAnalyzer':
    if USE_PYTORCH_IF_AVAILABLE and _MODEL_PATH.exists():
        try:
            from .trade_conclusion_model import TradeConclusionPyTorchModel
            model = TradeConclusionPyTorchModel()
            if model.load():
                return TradeConclusionAnalyzer(use_pytorch_model=model)
        except Exception as e:
            logger.debug("PyTorch trade conclusion model not used: %s", e)
    return TradeConclusionAnalyzer(use_pytorch_model=None)


def _fallback_conclusion(trade: Dict[str, Any]) -> str:
    """Минимальный вывод при ошибке."""
    pnl = float(trade.get('pnl') or 0)
    reason = (trade.get('close_reason') or '').strip()
    if pnl >= 0:
        return f'Прибыль. {reason or "Закрыто по условию"}'
    return f'Убыток. {reason or "Закрыто по условию"}'


class TradeConclusionAnalyzer:
    """
    Анализатор выводов по сделкам.
    Правила + опционально PyTorch-модель.
    """

    def __init__(self, use_pytorch_model: Optional[Any] = None):
        self.pytorch_model = use_pytorch_model

    def analyze(self, trade: Dict[str, Any]) -> str:
        if self.pytorch_model:
            try:
                out = self.pytorch_model.predict(trade)
                if out:
                    return out
            except Exception as e:
                logger.debug("PyTorch predict failed: %s", e)
        return self._rule_based_analyze(trade)

    def _rule_based_analyze(self, trade: Dict[str, Any]) -> str:
        pnl = float(trade.get('pnl') or 0)
        roi_raw = trade.get('roi')
        roi_pct = None
        if roi_raw is not None:
            r = float(roi_raw)
            roi_pct = r * 100 if 0 < abs(r) < 1.5 else r
        if roi_pct is None and trade.get('entry_price') and trade.get('exit_price'):
            ep, xp = float(trade['entry_price']), float(trade['exit_price'])
            if ep > 0:
                if (trade.get('direction') or '').upper() == 'LONG':
                    roi_pct = ((xp - ep) / ep) * 100
                else:
                    roi_pct = ((ep - xp) / ep) * 100
        reason = (trade.get('close_reason') or '').upper()
        direction = (trade.get('direction') or 'LONG').upper()
        entry_rsi = trade.get('entry_rsi')
        exit_rsi = trade.get('exit_rsi')
        if entry_rsi is None and trade.get('extra_data'):
            ed = trade['extra_data'] if isinstance(trade['extra_data'], dict) else {}
            entry_rsi = ed.get('entry_rsi') or ed.get('rsi')
        symbol = trade.get('symbol') or ''

        is_profit = pnl >= 0
        roi_abs = abs(roi_pct) if roi_pct is not None else 0

        if is_profit:
            return self._conclusion_profit(
                roi_pct=roi_pct, roi_abs=roi_abs, reason=reason,
                direction=direction, entry_rsi=entry_rsi, exit_rsi=exit_rsi, symbol=symbol
            )
        return self._conclusion_loss(
            roi_pct=roi_pct, roi_abs=roi_abs, reason=reason,
            direction=direction, entry_rsi=entry_rsi, exit_rsi=exit_rsi, symbol=symbol
        )

    def _conclusion_profit(
        self,
        roi_pct: Optional[float],
        roi_abs: float,
        reason: str,
        direction: str,
        entry_rsi: Optional[float],
        exit_rsi: Optional[float],
        symbol: str,
    ) -> str:
        # TP
        if any(x in reason for x in ('TP', 'TAKE_PROFIT', 'ТЕЙК')):
            if roi_pct is not None:
                return random.choice([
                    'Прибыль. Выход по TP — цель достигнута.',
                    'Прибыль. TP сработал — фиксация в целевой зоне.',
                    f'Прибыль +{roi_abs:.1f}%. TP — стратегия отработала по плану.',
                ])
            return random.choice([
                'Прибыль. Выход по TP — цель достигнута.',
                'Прибыль. TP сработал — фиксация в целевой зоне.',
            ])
        # RSI
        if any(x in reason for x in ('RSI', 'РСИ')):
            if roi_abs >= 30:
                return random.choice([
                    f'Прибыль +{roi_abs:.1f}%. RSI выход — сильный импульс, выход на локальном экстремуме.',
                    f'Крупная прибыль +{roi_abs:.1f}%. RSI сработал — своевременный выход в зоне перекупленности/перепроданности.',
                ])
            if roi_abs >= 15:
                return random.choice([
                    f'Прибыль +{roi_abs:.1f}%. RSI сигнал — выход в плюсе, хороший результат.',
                    f'Прибыль +{roi_abs:.1f}%. Выход по RSI — зафиксирована часть движения.',
                ])
            if roi_abs >= 5:
                return random.choice([
                    f'Прибыль +{roi_abs:.1f}%. RSI — умеренный выход, сигнал сработал.',
                    f'Небольшая прибыль +{roi_abs:.1f}%. RSI выход в плюсе.',
                ])
            return random.choice([
                'Прибыль. RSI — выход в плюсе, сигнал сработал.',
                'Прибыль. Выход по RSI — фиксация минимальной прибыли.',
            ])
        # MANUAL
        if 'MANUAL' in reason:
            return f'Прибыль. Ручное закрытие — +{roi_abs:.1f}%' if roi_pct is not None else 'Прибыль. Ручное закрытие.'
        return f'Прибыль. {reason or "Закрыто по условию"}'

    def _conclusion_loss(
        self,
        roi_pct: Optional[float],
        roi_abs: float,
        reason: str,
        direction: str,
        entry_rsi: Optional[float],
        exit_rsi: Optional[float],
        symbol: str,
    ) -> str:
        # SL
        if any(x in reason for x in ('SL', 'STOP', 'СЛОСС')):
            if roi_abs >= 15:
                return random.choice([
                    f'Убыток -{roi_abs:.1f}%. SL сработал — сильное движение против, стоп защитил от больших потерь.',
                    f'Крупный убыток -{roi_abs:.1f}%. Стоп-лосс — тренд пошёл против входа.',
                ])
            return random.choice([
                'Убыток. SL — стоп сработал, фиксация убытка.',
                f'Убыток -{roi_abs:.1f}%. Стоп-лосс отработал.',
            ])
        # RSI — основная причина однообразия
        if any(x in reason for x in ('RSI', 'РСИ')):
            if roi_abs >= 30:
                return random.choice([
                    f'Убыток -{roi_abs:.1f}%. RSI выход — сильный разворот после закрытия, возможно преждевременный выход или неудачный вход.',
                    f'Крупный убыток -{roi_abs:.1f}%. Выход по RSI в минусе — тренд продолжился против позиции.',
                ])
            if roi_abs >= 15:
                return random.choice([
                    f'Убыток -{roi_abs:.1f}%. RSI — выход до разворота, цена могла пойти в целевую зону после закрытия.',
                    f'Убыток -{roi_abs:.1f}%. Выход по RSI в минусе — слабый импульс при входе или ранний стоп.',
                ])
            if roi_abs >= 8:
                return random.choice([
                    f'Убыток -{roi_abs:.1f}%. RSI выход в минусе — возможно ранний выход; проверить пороги RSI.',
                    f'Убыток -{roi_abs:.1f}%. Выход по RSI до разворота — рассмотреть более широкий диапазон удержания.',
                ])
            # Малый убыток
            return random.choice([
                f'Убыток -{roi_abs:.1f}%. RSI — возможно ранний выход; малый убыток.',
                f'Небольшой убыток -{roi_abs:.1f}%. RSI выход в минусе — сигнал сработал с опозданием.',
            ])
        # MANUAL
        if 'MANUAL' in reason:
            return f'Убыток. Ручное закрытие — -{roi_abs:.1f}%' if roi_pct is not None else 'Убыток. Ручное закрытие.'
        return f'Убыток. {reason or "Закрыто по условию"}'
