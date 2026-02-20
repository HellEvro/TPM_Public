# -*- coding: utf-8 -*-
"""
PyTorch-модель для генерации выводов по сделкам.

Модель дообучается на истории: (признаки сделки) -> (id вывода).
Интерфейс: predict(trade_dict) -> str.

Обучение: python scripts/ai/train_trade_conclusion_model.py
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger('AI.TradeConclusionModel')

_MODEL_PATH = Path('data/ai/trade_conclusion_model.pt')


class TradeConclusionPyTorchModel:
    """
    Обёртка над PyTorch-моделью.
    При отсутствии обученной модели load() возвращает False.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or _MODEL_PATH
        self.model = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import torch
            self.model = torch.load(self.model_path, map_location='cpu')
            return self.model is not None
        except Exception as e:
            logger.warning("Не удалось загрузить модель выводов: %s", e)
            return False

    def predict(self, trade: Dict[str, Any]) -> Optional[str]:
        if not self.model:
            return None
        try:
            features = self._trade_to_features(trade)
            # TODO: вызов модели, маппинг id -> текст вывода
            # Пока заглушка — модель может вернуть None, тогда используются правила
            return None
        except Exception as e:
            logger.debug("predict failed: %s", e)
            return None

    def _trade_to_features(self, trade: Dict[str, Any]) -> list:
        """Преобразует сделку в вектор признаков для модели."""
        pnl = float(trade.get('pnl') or 0)
        roi = trade.get('roi')
        if roi is not None:
            roi_f = float(roi)
            roi_pct = roi_f * 100 if 0 < abs(roi_f) < 1.5 else roi_f
        else:
            roi_pct = 0.0
        reason = (trade.get('close_reason') or '').upper()
        direction = 1 if (trade.get('direction') or '').upper() == 'LONG' else 0
        entry_rsi = float(trade.get('entry_rsi') or 50)
        exit_rsi = float(trade.get('exit_rsi') or 50)
        # Признаки: roi_pct, pnl, direction, reason_encoded, entry_rsi, exit_rsi
        reason_idx = 0
        if 'RSI' in reason:
            reason_idx = 1
        elif any(x in reason for x in ('SL', 'STOP')):
            reason_idx = 2
        elif any(x in reason for x in ('TP', 'TAKE')):
            reason_idx = 3
        elif 'MANUAL' in reason:
            reason_idx = 4
        return [roi_pct / 100.0, pnl / 100.0, direction, reason_idx / 4.0, entry_rsi / 100.0, exit_rsi / 100.0]
