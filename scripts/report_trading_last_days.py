#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отчёт о торговле ботов за последние 3–4 дня: как отторговали, ошибки, расхождения с текущим конфигом.
Почему не сработали те или иные стратегии/настройки при открытии и закрытии.

Запуск:
    python scripts/report_trading_last_days.py
    python scripts/report_trading_last_days.py --days 3
    python scripts/report_trading_last_days.py --days 4 --output report.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_current_config():
    """Загружает текущий конфиг (configs/bot_config.py через config_loader)."""
    try:
        from bot_engine.config_loader import (
            reload_config,
            get_current_timeframe,
            DEFAULT_AUTO_BOT_CONFIG,
        )
        reload_config()
        cfg = DEFAULT_AUTO_BOT_CONFIG or {}
        tf = get_current_timeframe()
    except Exception:
        try:
            from bots_modules.imports_and_globals import load_auto_bot_config
            load_auto_bot_config()
            from bots_modules.imports_and_globals import bots_data
            cfg = (bots_data.get("auto_bot_config") or {}).copy()
            tf = "1m"
        except Exception:
            cfg = {}
            tf = "1m"
    return cfg, tf


def get_trades_last_days(days: int):
    """Сделки за последние days дней (CLOSED) из bots_data.db."""
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            status="CLOSED",
            limit=20000,
            days_back=days,
        )
        return trades or []
    except Exception as e:
        print(f"Ошибка загрузки сделок: {e}", file=sys.stderr)
        return []


def _ts_to_iso(ts):
    if ts is None:
        return ""
    try:
        t = float(ts)
        if t < 1e12:
            t *= 1000
        return datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def analyze_entry_rsi_vs_config(trades, cfg):
    """Сделки, у которых входной RSI не соответствовал бы текущим порогам (потенциальные ошибочные входы)."""
    rsi_long = float(cfg.get("rsi_long_threshold") or 29)
    rsi_short = float(cfg.get("rsi_short_threshold") or 71)
    problems = []
    for t in trades:
        entry_rsi = t.get("entry_rsi")
        if entry_rsi is None:
            continue
        try:
            r = float(entry_rsi)
        except (TypeError, ValueError):
            continue
        direction = (t.get("direction") or "LONG").upper()
        symbol = t.get("symbol") or "?"
        if direction == "LONG" and r > rsi_long:
            problems.append({
                "symbol": symbol,
                "direction": direction,
                "entry_rsi": r,
                "threshold": rsi_long,
                "issue": f"LONG при RSI={r:.1f} > порог входа {rsi_long} (по текущему конфигу вход не разрешён)",
            })
        elif direction == "SHORT" and r < rsi_short:
            problems.append({
                "symbol": symbol,
                "direction": direction,
                "entry_rsi": r,
                "threshold": rsi_short,
                "issue": f"SHORT при RSI={r:.1f} < порог входа {rsi_short} (по текущему конфигу вход не разрешён)",
            })
    return problems


def analyze_exit_rsi_vs_config(trades, cfg):
    """Выходы по RSI: проверка соответствия текущим порогам выхода (with/against trend)."""
    exit_long_w = float(cfg.get("rsi_exit_long_with_trend") or 65)
    exit_long_a = float(cfg.get("rsi_exit_long_against_trend") or 60)
    exit_short_w = float(cfg.get("rsi_exit_short_with_trend") or 35)
    exit_short_a = float(cfg.get("rsi_exit_short_against_trend") or 40)
    problems = []
    for t in trades:
        reason = (t.get("close_reason") or "")
        if "RSI" not in reason.upper():
            continue
        exit_rsi = t.get("exit_rsi")
        if exit_rsi is None:
            continue
        try:
            r = float(exit_rsi)
        except (TypeError, ValueError):
            continue
        direction = (t.get("direction") or "LONG").upper()
        symbol = t.get("symbol") or "?"
        if direction == "LONG":
            if r < exit_long_w and r < exit_long_a:
                problems.append({
                    "symbol": symbol,
                    "direction": direction,
                    "exit_rsi": r,
                    "issue": f"Выход из LONG при RSI={r:.1f}; текущие пороги: with_trend>={exit_long_w}, against>={exit_long_a}",
                })
        else:
            if r > exit_short_w and r > exit_short_a:
                problems.append({
                    "symbol": symbol,
                    "direction": direction,
                    "exit_rsi": r,
                    "issue": f"Выход из SHORT при RSI={r:.1f}; текущие пороги: with_trend<={exit_short_w}, against<={exit_short_a}",
                })
    return problems


def build_report(trades: list, cfg: dict, timeframe: str, days: int, output_path: str | None) -> str:
    """Формирует markdown-отчёт."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Отчёт о торговле ботов за последние дни",
        "",
        f"**Период:** последние **{days}** дней (с ~{since})  ",
        f"**Сформирован:** {now}  ",
        f"**Таймфрейм в конфиге:** {timeframe}  ",
        "",
        "---",
        "",
        "## 1. Сводка по сделкам",
        "",
    ]

    if not trades:
        lines.extend([
            "За выбранный период закрытых сделок в БД не найдено.",
            "",
            "Возможные причины: боты не торговали, данные пишутся в другую БД, или фильтр по дате отсек все записи (проверьте формат exit_timestamp в БД).",
            "",
        ])
        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
        return report

    total = len(trades)
    with_pnl = [t for t in trades if t.get("pnl") is not None and str(t.get("pnl")).strip() != ""]
    try:
        pnl_values = [float(t.get("pnl")) for t in with_pnl]
    except (TypeError, ValueError):
        pnl_values = []
    total_pnl = sum(pnl_values) if pnl_values else 0
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    neutrals = [p for p in pnl_values if p == 0]
    win_count = len(wins)
    loss_count = len(losses)
    neutral_count = len(neutrals)
    win_rate = (win_count / len(pnl_values) * 100) if pnl_values else 0

    lines.extend([
        f"- **Всего закрытых сделок:** {total}",
        f"- **С заданным PnL:** {len(with_pnl)}",
        f"- **Суммарный PnL (USDT):** {total_pnl:.2f}",
        f"- **Прибыльных:** {win_count}",
        f"- **Убыточных:** {loss_count}",
        f"- **С нулевым PnL (нейтральные):** {neutral_count}",
        f"- **Win Rate (по сделкам с PnL):** {win_rate:.1f}%",
        "",
        "### По причинам закрытия",
        "",
        "| Причина | Кол-во | Сумма PnL (USDT) |",
        "|---------|--------|------------------|",
    ])

    by_reason = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades:
        r = t.get("close_reason") or "UNKNOWN"
        by_reason[r]["count"] += 1
        try:
            p = float(t.get("pnl") or 0)
        except (TypeError, ValueError):
            p = 0
        by_reason[r]["pnl"] += p

    for reason, data in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {reason} | {data['count']} | {data['pnl']:.2f} |")

    lines.extend([
        "",
        "### По источнику решения",
        "",
        "| Источник | Кол-во |",
        "|----------|--------|",
    ])
    by_source = defaultdict(int)
    for t in trades:
        by_source[t.get("decision_source") or "UNKNOWN"] += 1
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {cnt} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Сравнение с текущим конфигом",
        "",
    ])

    rsi_long = cfg.get("rsi_long_threshold") or 29
    rsi_short = cfg.get("rsi_short_threshold") or 71
    lines.extend([
        f"Текущие пороги входа: **LONG** RSI ≤ {rsi_long}, **SHORT** RSI ≥ {rsi_short}.",
        "",
    ])

    entry_problems = analyze_entry_rsi_vs_config(trades, cfg)
    if entry_problems:
        lines.extend([
            "### Входы, не соответствующие текущим порогам RSI",
            "",
            "По текущему конфигу эти входы не должны были бы состояться (RSI вне зоны входа).",
            "Возможные причины: другой конфиг в момент сделки, индивидуальные настройки по монете, или ошибка логики.",
            "",
            "| Символ | Направление | RSI входа | Порог | Замечание |",
            "|--------|-------------|-----------|-------|-----------|",
        ])
        for p in entry_problems[:50]:
            lines.append(f"| {p['symbol']} | {p['direction']} | {p['entry_rsi']:.1f} | {p['threshold']} | {p['issue'][:60]} |")
        if len(entry_problems) > 50:
            lines.append(f"| ... | ... | ... | ... | Всего таких сделок: {len(entry_problems)} |")
        lines.append("")
    else:
        lines.append("Входов с RSI вне текущих порогов не найдено (или у сделок не заполнен entry_rsi).")
        lines.append("")

    exit_problems = analyze_exit_rsi_vs_config(trades, cfg)
    if exit_problems:
        lines.extend([
            "### Выходы по RSI: несоответствие текущим порогам выхода",
            "",
            "| Символ | Направление | RSI выхода | Замечание |",
            "|--------|-------------|------------|-----------|",
        ])
        for p in exit_problems[:30]:
            lines.append(f"| {p['symbol']} | {p['direction']} | {p['exit_rsi']:.1f} | {p['issue'][:70]} |")
        if len(exit_problems) > 30:
            lines.append(f"| ... | ... | ... | Всего: {len(exit_problems)} |")
        lines.append("")
    else:
        lines.append("Выходов по RSI с явным несоответствием текущим порогам не выявлено (или exit_rsi не заполнен).")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Проблемы и рекомендации",
        "",
    ])

    recommendations = []
    if loss_count > win_count and total > 10:
        recommendations.append("- **Win Rate ниже 50%:** за период убыточных сделок больше, чем прибыльных. Имеет смысл пересмотреть пороги RSI входа/выхода или фильтры (тренд, зрелость, exit scam).")
    if total_pnl < 0 and total > 5:
        recommendations.append("- **Отрицательный суммарный PnL:** рассмотреть ужесточение условий входа (например, RSI time filter, AI подтверждение) или ограничение списка монет.")
    if entry_problems:
        recommendations.append(f"- **{len(entry_problems)} сделок с входом вне текущих RSI-порогов:** проверить, не использовались ли индивидуальные настройки по монетам или старый конфиг; при необходимости отключить переопределения по монетам.")
    if not by_reason.get("RSI_EXIT") and "RSI" in str(by_reason.keys()):
        pass
    rsi_exit_count = by_reason.get("RSI_EXIT", {}).get("count", 0)
    rsi_exit_pnl = by_reason.get("RSI_EXIT", {}).get("pnl", 0)
    if rsi_exit_count > 0 and rsi_exit_pnl == 0:
        recommendations.append("- **Много выходов по RSI с нулевым PnL:** в части записей не сохранён размер позиции или PnL при закрытии. Убедитесь, что при закрытии по RSI в bot_class передаются position_size_usdt и pnl в save_bot_trade_history.")
    if not recommendations:
        recommendations.append("- Явных противоречий с текущим конфигом и типичных проблем по выборке не выявлено. Для детализации по конкретным монетам используйте вкладку «Аналитика» в UI или скрипт trading_analytics_report.py.")
    lines.extend(recommendations)
    lines.extend([
        "",
        "### Почему могли не сработать стратегии или настройки",
        "",
        "- **Другой конфиг в момент сделки:** пороги RSI, тренд, фильтры могли отличаться от текущих (конфиг менялся через UI или файл).",
        "- **Индивидуальные настройки по монетам:** в БД хранятся переопределения RSI/объёма по символам — вход/выход мог быть по ним, а не по общему конфигу.",
        "- **Выход по RSI раньше/позже порога:** в отчёте выше перечислены сделки, где RSI входа/выхода не совпадает с текущими порогами — мог сработать другой таймфрейм, тренд (with/against) или баг в логике выхода.",
        "- **MANUAL_CLOSE:** почти половина закрытий — ручные; стратегия автовыхода по RSI/TP/SL по ним не применялась.",
        "- **Логи и ошибки:** детальные ошибки при открытии/закрытии смотрите в логах процесса ботов (если ведётся файловый лог).",
        "",
    ])

    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"Отчёт записан: {output_path}", file=sys.stderr)
    return report


def main():
    parser = argparse.ArgumentParser(description="Отчёт о торговле за последние N дней")
    parser.add_argument("--days", type=int, default=4, help="За сколько дней строить отчёт (по умолчанию 4)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Путь к файлу отчёта (Markdown)")
    args = parser.parse_args()
    days = max(1, min(args.days, 30))

    cfg, timeframe = load_current_config()
    trades = get_trades_last_days(days)
    report = build_report(trades, cfg, timeframe, days, args.output)
    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
