# -*- coding: utf-8 -*-
"""
FullAI: изучение совершённых сделок и доработка параметров (блок 7).
ИИ сам учится понимать, что значат параметры и как они влияют на сделки:
- при наличии обученной модели (ParameterQualityPredictor / AIContinuousLearning) — рекомендации ИИ;
- иначе — эвристика по win_rate как стартовый шаг, пока ИИ набирает данные для обучения.
Обновляет **только** конфиг FullAI и таблицу full_ai_coin_params.
"""
import logging
import re
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
        # Вызываем обучение только при достаточном числе сделок (≥10), иначе learn_from_real_trades только спамит логи
        if raw_trades and len(raw_trades) >= 10:
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


def _parse_rsi_range(label: str) -> Optional[Tuple[float, float]]:
    """Парсит rsi_range типа '26-30', '31-35' -> (low, high)."""
    if not label or not isinstance(label, str):
        return None
    m = re.match(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', label)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def _suggest_rsi_from_bad_ranges(bad_rsi_ranges: List[Dict], for_long: bool) -> Optional[float]:
    """
    На основе неудачных RSI-диапазонов предлагает порог.
    for_long=True: rsi_long_threshold (вход при RSI <= X). Плохие низкие диапазоны (26-30) → порог 25.
    for_long=False: rsi_short_threshold (вход при RSI >= X). Плохие высокие диапазоны (66-70) → порог 71.
    """
    if not bad_rsi_ranges:
        return None
    parsed = []
    for r in bad_rsi_ranges:
        lab = r.get('rsi_range') if isinstance(r, dict) else str(r)
        p = _parse_rsi_range(lab)
        if p:
            parsed.append(p)
    if not parsed:
        return None
    if for_long:
        # LONG: плохие низкие RSI (26-30, 31-35) → threshold ниже, вход только при RSI < min(low)
        low_ranges = [(pl, ph) for pl, ph in parsed if ph <= 50]
        if not low_ranges:
            return None
        min_low = min(pl for pl, _ in low_ranges)
        return max(1, min_low - 1)
    else:
        # SHORT: плохие высокие RSI (66-70, 71-100) → threshold выше, вход при RSI > max(high)
        high_ranges = [(pl, ph) for pl, ph in parsed if pl >= 50]
        if not high_ranges:
            return None
        max_high = max(ph for _, ph in high_ranges)
        return min(99, max_high + 1)


def run_fullai_trades_analysis(
    days_back: int = 7,
    min_trades_per_symbol: int = 2,
    adjust_params: bool = True,
    symbol_filter: Optional[str] = None,
    limit: int = 2000,
) -> Dict[str, Any]:
    """
    Анализ закрытых сделок и обновление параметров FullAI по монетам.
    Использует и свои данные (bot_trades_history), и торговую аналитику (run_full_analytics):
    - unsuccessful_coins, successful_coins, bad_rsi_ranges, bad_trends.
    Пишет только в full_ai_config (БД) и full_ai_coin_params.
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

        # Загружаем торговую аналитику (unsuccessful/successful coins, bad RSI, bad trends)
        trading_ai: Dict[str, Any] = {}
        try:
            from bot_engine.trading_analytics import run_full_analytics, get_analytics_for_ai
            report = run_full_analytics(
                load_bot_trades_from_db=True,
                load_exchange_from_api=False,
                bots_db_limit=5000,
            )
            trading_ai = get_analytics_for_ai(report)
            uc = trading_ai.get('unsuccessful_coins') or []
            sc = trading_ai.get('successful_coins') or []
            us_map = {s['symbol']: s for s in (trading_ai.get('unsuccessful_settings') or [])}
            trading_ai['_unsuccessful_symbols'] = {c['symbol'] for c in uc}
            trading_ai['_successful_symbols'] = {c['symbol'] for c in sc}
            trading_ai['_successful_settings_by_symbol'] = {
                s['symbol']: s for s in (trading_ai.get('successful_settings') or [])
            }
            trading_ai['_unsuccessful_settings_by_symbol'] = us_map
            logger.info("[FullAI логика] Торговая аналитика загружена: неудачных монет=%s, удачных=%s",
                        len(trading_ai['_unsuccessful_symbols']), len(trading_ai['_successful_symbols']))
        except Exception as e:
            logger.warning("[FullAI learner] Не удалось загрузить торговую аналитику: %s", e)

        def _build_insights(ta: Dict, trades_count: int) -> Dict[str, List[str]]:
            out = {'mistakes': [], 'successes': [], 'recommendations': []}
            problems = ta.get('problems') or []
            recs = ta.get('recommendations') or []
            uc = ta.get('unsuccessful_coins') or []
            sc = ta.get('successful_coins') or []
            metrics = ta.get('metrics') or {}
            wr = metrics.get('win_rate_pct')
            total_pnl = metrics.get('total_pnl_usdt')
            for p in problems:
                out['mistakes'].append(p)
            for r in recs:
                out['recommendations'].append(r)
            if uc:
                out['mistakes'].append(
                    f"Неудачные монеты ({len(uc)}): {', '.join(c['symbol'] for c in uc[:15])}{'...' if len(uc) > 15 else ''}. "
                    "Win Rate < 45% или PnL < 0 — нужно избегать текущих настроек входа/выхода."
                )
            if sc:
                out['successes'].append(
                    f"Удачные монеты ({len(sc)}): {', '.join(c['symbol'] for c in sc[:15])}{'...' if len(sc) > 15 else ''}. "
                    "Win Rate ≥ 55% и PnL > 0 — повторить текущие настройки (RSI-пороги, TP/SL)."
                )
            if wr is not None and total_pnl is not None:
                if wr < 45 or (total_pnl is not None and total_pnl < 0):
                    out['mistakes'].append(
                        f"Общий Win Rate {wr:.1f}%, PnL {total_pnl:.2f} USDT. "
                        "Система допускает слишком много убыточных сделок — ужесточить TP/SL или RSI-пороги."
                    )
                elif wr >= 55 and total_pnl and total_pnl > 0:
                    out['successes'].append(
                        f"Общий Win Rate {wr:.1f}%, PnL {total_pnl:+.2f} USDT. "
                        "Текущие параметры работают — можно слегка повысить TP для увеличения прибыли."
                    )
            return out

        trades = db.get_bot_trades_history(
            status='CLOSED',
            symbol=symbol_filter,
            days_back=days_back,
            limit=limit,
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
        insights = _build_insights(trading_ai, len(trades))
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

            # Контекст из торговой аналитики
            in_unsuccessful = symbol in (trading_ai.get('_unsuccessful_symbols') or set())
            in_successful = symbol in (trading_ai.get('_successful_symbols') or set())
            us_settings = (trading_ai.get('_unsuccessful_settings_by_symbol') or {}).get(symbol, {})
            bad_rsi = us_settings.get('bad_rsi_ranges') or []

            # Сначала пробуем ИИ: сам понимает параметры и учится на сделках
            ai_result = _get_ai_param_recommendation(symbol, current, evals, fullai_global)
            if ai_result is not None:
                new_params, reason_text = ai_result
                changed = True
                if in_unsuccessful:
                    reason_text += " Торговая аналитика: монета в неудачных."
                elif in_successful:
                    reason_text += " Торговая аналитика: монета в удачных."
            else:
                # Эвристика по win_rate + учёт торговой аналитики
                tp = current.get('take_profit_percent') or fullai_global.get('take_profit_percent') or 15
                sl = current.get('max_loss_percent') or fullai_global.get('max_loss_percent') or 10
                tp_f = float(tp)
                sl_f = float(sl)
                # Пороги: при неудачной монете — чувствительнее к плохому win_rate
                wr_low = 0.4
                wr_high = 0.6
                if in_unsuccessful:
                    wr_low = 0.45  # при неудачной монете — чуть раньше ужесточаем
                if in_successful:
                    wr_low, wr_high = 0.35, 0.65  # при удачной — реже меняем (избегаем переоптимизации)
                if win_rate < wr_low and total >= 3:
                    tp_f = max(5, tp_f - 2)
                    sl_f = min(20, sl_f + 2)
                    changed = True
                    new_params = {**current, 'take_profit_percent': round(tp_f, 1), 'max_loss_percent': round(sl_f, 1)}
                    reason_text = (
                        f"Эвристика: низкий win_rate={win_rate:.2f} (< {wr_low}), n={total} сделок. "
                        "Снижаем TP, увеличиваем SL."
                    )
                    if in_unsuccessful:
                        reason_text += " Торговая аналитика: монета в неудачных — подтверждает ужесточение."
                elif win_rate >= wr_high and total >= 3:
                    tp_f = min(50, tp_f + 2)
                    sl_f = max(5, sl_f - 1)
                    changed = True
                    new_params = {**current, 'take_profit_percent': round(tp_f, 1), 'max_loss_percent': round(sl_f, 1)}
                    reason_text = (
                        f"Эвристика: высокий win_rate={win_rate:.2f} (>= {wr_high}), n={total} сделок. "
                        "Повышаем TP, снижаем SL."
                    )
                    if in_successful:
                        reason_text += " Торговая аналитика: монета в удачных — подтверждает ослабление."
            # Дополнительно: RSI-пороги из неудачных диапазонов (торговая аналитика)
            if us_settings and bad_rsi and not changed:
                rsi_long_new = _suggest_rsi_from_bad_ranges(bad_rsi, for_long=True)
                rsi_short_new = _suggest_rsi_from_bad_ranges(bad_rsi, for_long=False)
                rsi_long_cur = current.get('rsi_long_threshold') or fullai_global.get('rsi_long_threshold') or 29
                rsi_short_cur = current.get('rsi_short_threshold') or fullai_global.get('rsi_short_threshold') or 71
                if rsi_long_new is not None and abs(rsi_long_new - rsi_long_cur) >= 2:
                    new_params = {**new_params, 'rsi_long_threshold': int(round(rsi_long_new))}
                    changed = True
                    reason_text = (reason_text or '') + (
                        f" Торговая аналитика: bad_rsi_ranges -> rsi_long_threshold {rsi_long_cur} -> {int(rsi_long_new)}."
                    )
                if rsi_short_new is not None and abs(rsi_short_new - rsi_short_cur) >= 2:
                    new_params = {**new_params, 'rsi_short_threshold': int(round(rsi_short_new))}
                    changed = True
                    reason_text = (reason_text or '') + (
                        f" Торговая аналитика: bad_rsi_ranges -> rsi_short_threshold {rsi_short_cur} -> {int(rsi_short_new)}."
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
                    rsi_long_old = current.get('rsi_long_threshold') or fullai_global.get('rsi_long_threshold') or 29
                    rsi_long_nw = new_params.get('rsi_long_threshold', rsi_long_old)
                    if rsi_long_old != rsi_long_nw:
                        changes_list.append({
                            'symbol': symbol,
                            'param': 'rsi_long_threshold',
                            'old': rsi_long_old,
                            'new': rsi_long_nw,
                            'reason': reason_text,
                        })
                    rsi_short_old = current.get('rsi_short_threshold') or fullai_global.get('rsi_short_threshold') or 71
                    rsi_short_nw = new_params.get('rsi_short_threshold', rsi_short_old)
                    if rsi_short_old != rsi_short_nw:
                        changes_list.append({
                            'symbol': symbol,
                            'param': 'rsi_short_threshold',
                            'old': rsi_short_old,
                            'new': rsi_short_nw,
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
            'insights': insights,  # ошибки, успехи, рекомендации для отображения пользователю
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
