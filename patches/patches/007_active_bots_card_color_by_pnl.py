"""
Патч 007: цвет карточек на странице «Активные боты» по PnL (зелёный — прибыль, красный — убыток),
а не по направлению Long/Short.
"""
from pathlib import Path

OLD = """                    const exchangeUrl = this.getExchangeLink(bot.symbol, 'bybit');
                    const isLong = (bot.position_side || '').toUpperCase() === 'LONG';
                    const cardBg = isVirtual ? 'rgba(156, 39, 176, 0.12)' : (isLong ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)');"""

NEW = """                    const exchangeUrl = this.getExchangeLink(bot.symbol, 'bybit');
                    // Цвет карточки по PnL: зелёный — прибыль, красный — убыток (направление Long/Short уже показано подписью)
                    const pnlValue = isVirtual ? (bot.unrealized_pnl ?? 0) : (bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0);
                    const isProfit = Number(pnlValue) >= 0;
                    const cardBg = isVirtual ? 'rgba(156, 39, 176, 0.12)' : (isProfit ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)');"""


def apply(project_root: Path) -> bool:
    path = project_root / "static" / "js" / "managers" / "bots_manager.js"
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8")
    if "isProfit" in text and "cardBg = isVirtual" in text:
        return True
    if OLD not in text:
        return True
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return True

