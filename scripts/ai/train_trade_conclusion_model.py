# -*- coding: utf-8 -*-
"""
Обучение PyTorch-модели для генерации выводов по сделкам.

Использование:
  python scripts/ai/train_trade_conclusion_model.py

Модель обучается предсказывать id шаблона вывода по признакам сделки.
Данные: bot_trades_history из bots_data.db.
После обучения — analyze_trade_conclusion будет использовать модель (если data/ai/trade_conclusion_model.pt существует).
"""

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("PyTorch не установлен. Установите: pip install torch")
        return 1

    from bot_engine.bots_database import get_bots_database
    from bot_engine.ai.trade_conclusion_analyzer import TradeConclusionAnalyzer

    db = get_bots_database()
    trades = db.get_bot_trades_history(status='CLOSED', limit=5000)
    if not trades:
        logger.warning("Нет закрытых сделок для обучения")
        return 0

    analyzer = TradeConclusionAnalyzer(use_pytorch_model=None)
    # Собираем пары (признаки, вывод) — правило генерирует "правильный" вывод как метку
    pairs = []
    for t in trades:
        conclusion = analyzer._rule_based_analyze(t)
        # Признаки — см. trade_conclusion_model._trade_to_features
        pnl = float(t.get('pnl') or 0)
        roi = t.get('roi')
        roi_pct = (float(roi) * 100 if roi is not None and 0 < abs(float(roi)) < 1.5 else (float(roi) if roi is not None else 0))
        if roi_pct == 0 and t.get('entry_price') and t.get('exit_price'):
            ep, xp = float(t['entry_price']), float(t['exit_price'])
            if ep > 0:
                if (t.get('direction') or '').upper() == 'LONG':
                    roi_pct = ((xp - ep) / ep) * 100
                else:
                    roi_pct = ((ep - xp) / ep) * 100
        reason = (t.get('close_reason') or '').upper()
        direction = 1 if (t.get('direction') or '').upper() == 'LONG' else 0
        entry_rsi = float(t.get('entry_rsi') or 50)
        exit_rsi = float(t.get('exit_rsi') or 50)
        reason_idx = 0
        if 'RSI' in reason:
            reason_idx = 1
        elif any(x in reason for x in ('SL', 'STOP')):
            reason_idx = 2
        elif any(x in reason for x in ('TP', 'TAKE')):
            reason_idx = 3
        elif 'MANUAL' in reason:
            reason_idx = 4
        features = [roi_pct / 100.0, pnl / 100.0, direction, reason_idx / 4.0, entry_rsi / 100.0, exit_rsi / 100.0]
        pairs.append((features, conclusion))

    # TODO: маппинг conclusion -> class_id, обучение nn.Module
    # Пока — сохраняем пары для отладки; полная реализация потребует словаря шаблонов
    logger.info("Собрано %d пар (признаки, вывод) для обучения. Полная реализация — в разработке.", len(pairs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
