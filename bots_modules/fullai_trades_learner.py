# -*- coding: utf-8 -*-
"""
FullAI: изучение совершённых сделок и доработка параметров (блок 7).
ИИ сам учится понимать, что значат параметры и как они влияют на сделки:
- при наличии обученной модели (ParameterQualityPredictor / AIContinuousLearning) — рекомендации ИИ;
- иначе — эвристика по win_rate как стартовый шаг, пока ИИ набирает данные для обучения.
Обновляет **только** конфиг FullAI и таблицу full_ai_coin_params.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger('BOTS')


def _is_fullai_enabled() -> bool:
    """FullAI включён только если full_ai_control в пользовательском конфиге."""
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            return (bots_data.get('auto_bot_config') or {}).get('full_ai_control', False)
    except Exception:
        return False


def _get_ai_param_recommendation(
    symbol: str,
    current: Dict[str, Any],
    evals: List[Dict],
    fullai_global: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Спрашивает ИИ (обученные модели), как изменить параметры на основе исходов сделок.
    ИИ сам понимает связь параметров с результатами и предлагает изменения.
    Returns: (new_params_dict, reason_for_log) или None (тогда использовать эвристику).
    """
    # 1) ParameterQualityPredictor: модель предсказания качества параметров (обучена на исходах)
    try:
        from bot_engine.ai.parameter_quality_predictor import ParameterQualityPredictor
        predictor = ParameterQualityPredictor()
        if predictor.is_trained and predictor.model:
            base_rsi = {
                'oversold': current.get('rsi_long_threshold') or fullai_global.get('rsi_long_threshold') or 29,
                'overbought': current.get('rsi_short_threshold') or fullai_global.get('rsi_short_threshold') or 71,
                'exit_long_with_trend': current.get('rsi_exit_long_with_trend') or fullai_global.get('rsi_exit_long_with_trend') or 65,
                'exit_long_against_trend': current.get('rsi_exit_long_against_trend') or fullai_global.get('rsi_exit_long_against_trend') or 60,
                'exit_short_with_trend': current.get('rsi_exit_short_with_trend') or fullai_global.get('rsi_exit_short_with_trend') or 35,
                'exit_short_against_trend': current.get('rsi_exit_short_against_trend') or fullai_global.get('rsi_exit_short_against_trend') or 40,
            }
            risk = {
                'stop_loss': float(current.get('max_loss_percent') or fullai_global.get('max_loss_percent') or 10),
                'take_profit': float(current.get('take_profit_percent') or fullai_global.get('take_profit_percent') or 15),
                'trailing_stop_activation': float(current.get('trailing_stop_activation') or fullai_global.get('trailing_stop_activation') or 20),
                'trailing_stop_distance': float(current.get('trailing_stop_distance') or fullai_global.get('trailing_stop_distance') or 5),
            }
            suggestions = predictor.suggest_optimal_params(base_rsi, risk_params=risk, num_suggestions=3)
            if suggestions:
                best_rsi, quality = suggestions[0]
                new_params = {
                    **current,
                    'take_profit_percent': round(risk.get('take_profit', 15), 1),
                    'max_loss_percent': round(risk.get('stop_loss', 10), 1),
                    'rsi_long_threshold': best_rsi.get('oversold', base_rsi['oversold']),
                    'rsi_short_threshold': best_rsi.get('overbought', base_rsi['overbought']),
                    'rsi_exit_long_with_trend': best_rsi.get('exit_long_with_trend', base_rsi['exit_long_with_trend']),
                    'rsi_exit_long_against_trend': best_rsi.get('exit_long_against_trend', base_rsi['exit_long_against_trend']),
                    'rsi_exit_short_with_trend': best_rsi.get('exit_short_with_trend', base_rsi['exit_short_with_trend']),
                    'rsi_exit_short_against_trend': best_rsi.get('exit_short_against_trend', base_rsi['exit_short_against_trend']),
                }
                reason = (
                    "ИИ сам обучился на истории сделок: модель предсказания качества параметров рекомендует эти значения "
                    "(понимает, как параметры влияют на исход сделок). Предсказанное качество: %.2f." % quality
                )
                return (new_params, reason)
    except Exception as e:
        logger.debug("[FullAI learner] ParameterQualityPredictor не применим: %s", e)

    # 2) AIContinuousLearning: база знаний по сделкам, оптимальные параметры по символу
    try:
        from bot_engine.ai.ai_continuous_learning import AIContinuousLearning
        cl = AIContinuousLearning()
        raw_trades = [{'symbol': e.get('symbol'), 'pnl': e.get('roi', 0) * 0.01, 'success': e.get('success')} for e in evals]
        if raw_trades:
            cl.learn_from_real_trades(raw_trades)
        optimal = cl.get_optimal_parameters_for_symbol(symbol)
        if optimal and isinstance(optimal, dict):
            new_params = {**current}
            for key in ('take_profit_percent', 'max_loss_percent', 'rsi_long_threshold', 'rsi_short_threshold'):
                if key in optimal and optimal[key] is not None:
                    new_params[key] = optimal[key]
            if new_params != current:
                reason = (
                    "ИИ сам научился торговать: база знаний (AIContinuousLearning) обновлена исходами сделок, "
                    "оптимальные параметры для символа взяты из накопленного опыта."
                )
                return (new_params, reason)
    except Exception as e:
        logger.debug("[FullAI learner] AIContinuousLearning не применим: %s", e)

    return None


def _evaluate_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Оценка одной сделки: успех/неудача по roi, is_successful, close_reason.
    Возвращает dict с ключами: success (bool), roi (float), reason (str).
    """
    roi = trade.get('roi')
    if roi is None and trade.get('pnl') is not None and trade.get('position_size_usdt'):
        try:
            roi = float(trade['pnl']) / float(trade['position_size_usdt']) * 100.0
        except (TypeError, ZeroDivisionError):
            roi = 0.0
    roi = float(roi) if roi is not None else 0.0
    is_ok = trade.get('is_successful', False)
    if isinstance(is_ok, (int, float)):
        is_ok = bool(is_ok)
    if not is_ok and roi == 0.0:
        is_ok = roi > 0
    reason = trade.get('close_reason') or ''
    return {'success': is_ok, 'roi': roi, 'reason': reason, 'symbol': trade.get('symbol', '')}


def run_fullai_trades_analysis(
    days_back: int = 7,
    min_trades_per_symbol: int = 2,
    adjust_params: bool = True,
) -> Dict[str, Any]:
    """
    Анализ закрытых сделок и обновление параметров FullAI по монетам.
    Вызывать при включённом FullAI: по расписанию и/или после закрытия сделки.
    Пишет только в full_ai_config (БД) и full_ai_coin_params. Не трогает
    пользовательский конфиг и individual_coin_settings.
    """
    if not _is_fullai_enabled():
        logger.debug("[FullAI learner] Пропуск: full_ai_control выключен")
        return {'success': True, 'skipped': True, 'reason': 'FullAI disabled'}
    try:
        from bot_engine.bots_database import get_bots_database
        from bots_modules.imports_and_globals import (
            get_effective_auto_bot_config,
            load_full_ai_config_from_db,
            save_full_ai_config_to_db,
        )
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            status='CLOSED',
            days_back=days_back,
            limit=500,
        )
        if not trades:
            logger.info("[FullAI логика] Изучение сделок: закрытых сделок за период нет, пропуск.")
            return {'success': True, 'analyzed': 0, 'updated_symbols': []}
        logger.info(
            "[FullAI логика] Изучение сделок: запуск анализа за последние %s дн., сделок=%s, min_сделок_на_монету=%s. Обучение на ошибках/успехах по исходам.",
            days_back, len(trades), min_trades_per_symbol,
        )
        by_symbol: Dict[str, List[Dict]] = {}
        for t in trades:
            sym = (t.get('symbol') or '').upper()
            if not sym:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(_evaluate_trade(t))
        updated = []
        changes_list: List[Dict[str, Any]] = []  # [{symbol, param, old, new, reason}, ...]
        for symbol, evals in by_symbol.items():
            if len(evals) < min_trades_per_symbol:
                continue
            wins = sum(1 for e in evals if e.get('success'))
            total = len(evals)
            win_rate = wins / total if total else 0
            avg_roi = sum(e.get('roi', 0) for e in evals) / total if total else 0
            current = db.load_full_ai_coin_params(symbol) or {}
            if not adjust_params:
                continue
            fullai_global = get_effective_auto_bot_config()
            changed = False
            new_params = {**current}
            reason_text = ""

            # Сначала пробуем ИИ: сам понимает параметры и учится на сделках
            ai_result = _get_ai_param_recommendation(symbol, current, evals, fullai_global)
            if ai_result is not None:
                new_params, reason_text = ai_result
                changed = True
            else:
                # Эвристика по win_rate, пока ИИ не обучен или не применим
                tp = current.get('take_profit_percent') or fullai_global.get('take_profit_percent') or 15
                sl = current.get('max_loss_percent') or fullai_global.get('max_loss_percent') or 10
                tp_f = float(tp)
                sl_f = float(sl)
                if win_rate < 0.4 and total >= 3:
                    tp_f = max(5, tp_f - 2)
                    sl_f = min(20, sl_f + 2)
                    changed = True
                    new_params = {**current, 'take_profit_percent': round(tp_f, 1), 'max_loss_percent': round(sl_f, 1)}
                    reason_text = (
                        f"Эвристика (ИИ-модель пока не обучена/не применима): низкий win_rate={win_rate:.2f} (< 0.4), n={total} сделок. "
                        "Временная логика: снижаем TP, увеличиваем SL. После накопления данных ИИ сам научится подбирать параметры."
                    )
                elif win_rate >= 0.6 and total >= 3:
                    tp_f = min(50, tp_f + 2)
                    sl_f = max(5, sl_f - 1)
                    changed = True
                    new_params = {**current, 'take_profit_percent': round(tp_f, 1), 'max_loss_percent': round(sl_f, 1)}
                    reason_text = (
                        f"Эвристика (ИИ-модель пока не обучена/не применима): высокий win_rate={win_rate:.2f} (>= 0.6), n={total} сделок. "
                        "Временная логика: повышаем TP, снижаем SL. ИИ будет сам обучаться на сделках и поймёт, как параметры влияют на результат."
                    )
            if changed:
                if db.save_full_ai_coin_params(symbol, new_params):
                    updated.append(symbol)
                    tp_old = current.get('take_profit_percent') or fullai_global.get('take_profit_percent') or 15
                    sl_old = current.get('max_loss_percent') or fullai_global.get('max_loss_percent') or 10
                    tp_new = new_params.get('take_profit_percent', tp_old)
                    sl_new = new_params.get('max_loss_percent', sl_old)
                    if tp_old != tp_new:
                        changes_list.append({
                            'symbol': symbol,
                            'param': 'take_profit_percent',
                            'old': tp_old,
                            'new': tp_new,
                            'reason': reason_text,
                        })
                    if sl_old != sl_new:
                        changes_list.append({
                            'symbol': symbol,
                            'param': 'max_loss_percent',
                            'old': sl_old,
                            'new': sl_new,
                            'reason': reason_text,
                        })
                    logger.info(
                        "[FullAI изменения] Источник: изучение сделок (learner). Монета: %s. "
                        "Было: take_profit_percent=%s%%, max_loss_percent=%s%%. "
                        "Стало: take_profit_percent=%s%%, max_loss_percent=%s%%. "
                        "Причина: %s",
                        symbol, tp_old, sl_old, tp_new, sl_new, reason_text,
                    )
                    logger.info(
                        "[FullAI логика] Оценка сделок по монете: всего=%s, успешных=%s, win_rate=%s%%, avg_roi=%s%%. %s",
                        total, wins, round(win_rate * 100, 1), round(avg_roi, 2), reason_text,
                    )
        if updated:
            logger.info(
                "[FullAI логика] Изучение сделок завершено: проанализировано %s сделок, монет с достаточными данными=%s, "
                "параметры обновлены по монетам: %s (обучение на win_rate: <0.4 → ужесточаем SL/снижаем TP, >=0.6 → повышаем TP/снижаем SL).",
                len(trades), len(by_symbol), updated,
            )
        else:
            logger.info(
                "[FullAI логика] Изучение сделок завершено: проанализировано %s сделок, монет=%s, изменений нет (win_rate в пределах 0.4–0.6 или мало сделок).",
                len(trades), len(by_symbol),
            )
        return {
            'success': True,
            'analyzed': len(trades),
            'symbols_evaluated': len(by_symbol),
            'updated_symbols': updated,
            'changes': changes_list,  # [{symbol, param, old, new, reason}, ...] для UI "старое -> новое"
        }
    except Exception as e:
        logger.exception(f"[FullAI learner] Ошибка: {e}")
        return {'success': False, 'error': str(e)}


def run_fullai_trades_analysis_after_close(symbol: Optional[str] = None):
    """
    Короткий запуск анализа после закрытия сделки (например по одной монете или все за 1 день).
    Вызывать из bot_class после успешного закрытия позиции в режиме FullAI.
    """
    run_fullai_trades_analysis(days_back=1, min_trades_per_symbol=1, adjust_params=True)
