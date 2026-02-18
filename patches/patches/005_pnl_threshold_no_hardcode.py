"""
Патч 005: дефолт порога PnL = 10, один источник правды, без хардкода в HTML.
- config.js: DEFAULTS.PNL_THRESHOLD = 10
- app.py: DEFAULTS.PNL_THRESHOLD = 10, использование в get_positions и background_update
- positions.html: убрать value="100", плейсхолдер; средняя колонка "PnL < — USDT"
- positions_manager.js: дефолт из DEFAULTS
- positions.js: вызов updateBlockHeader('PROFITABLE', 0) при инициализации фильтра
"""
from pathlib import Path


def apply(project_root: Path) -> bool:
    applied = False

    # 1) config.js
    config_js = project_root / "static" / "js" / "config.js"
    if config_js.exists():
        text = config_js.read_text(encoding="utf-8")
        if "PNL_THRESHOLD: 100" in text:
            text = text.replace("PNL_THRESHOLD: 100", "PNL_THRESHOLD: 10")
            config_js.write_text(text, encoding="utf-8")
            applied = True

    # 2) app.py
    app_py = project_root / "app.py"
    if app_py.exists():
        text = app_py.read_text(encoding="utf-8")
        if "PNL_THRESHOLD = 100" in text:
            text = text.replace("PNL_THRESHOLD = 100", "PNL_THRESHOLD = 10")
            applied = True
        if "pnl_threshold', 100)" in text:
            text = text.replace("pnl_threshold', 100)", "pnl_threshold', DEFAULTS.PNL_THRESHOLD)")
            applied = True
        if "pnl >= 100:" in text and "DEFAULTS.PNL_THRESHOLD" not in text:
            text = text.replace("pnl >= 100:", "pnl >= DEFAULTS.PNL_THRESHOLD:")
            applied = True
        if applied or "DEFAULTS.PNL_THRESHOLD" in text:
            if applied:
                app_py.write_text(text, encoding="utf-8")
            applied = True

    # 3) positions.html
    positions_html = project_root / "templates" / "pages" / "positions.html"
    if positions_html.exists():
        text = positions_html.read_text(encoding="utf-8")
        if 'value="100"' in text and "pnl-filter-input" in text:
            text = text.replace(
                '<input type="number" id="pnl-filter-input" value="100" min="5" step="5" class="header-input">',
                '<input type="number" id="pnl-filter-input" min="5" step="5" class="header-input" placeholder="—">',
            )
            applied = True
        if "PnL < 100 USDT" in text:
            text = text.replace("PnL < 100 USDT", "PnL < — USDT")
            applied = True
        if applied:
            positions_html.write_text(text, encoding="utf-8")

    # 4) positions_manager.js
    pm_js = project_root / "static" / "js" / "positions_manager.js"
    if pm_js.exists():
        text = pm_js.read_text(encoding="utf-8")
        if "|| 100;" in text or "|| 100)" in text:
            text = text.replace(
                "parseFloat(localStorage.getItem('pnl_threshold')) || 100",
                "parseFloat(localStorage.getItem('pnl_threshold')) || (typeof DEFAULTS !== 'undefined' ? DEFAULTS.PNL_THRESHOLD : 10)",
            )
            text = text.replace(
                "parseFloat(value) || 100",
                "parseFloat(value) || (typeof DEFAULTS !== 'undefined' ? DEFAULTS.PNL_THRESHOLD : 10)",
            )
            pm_js.write_text(text, encoding="utf-8")
            applied = True

    # 5) positions.js — вызов updateBlockHeader при инициализации фильтра
    pos_js = project_root / "static" / "js" / "positions.js"
    if pos_js.exists():
        text = pos_js.read_text(encoding="utf-8")
        if "filterInput.value = this.pnlThreshold;" in text and "this.updateBlockHeader('PROFITABLE', 0)" not in text:
            text = text.replace(
                "filterInput.value = this.pnlThreshold;\n            filterInput.addEventListener",
                "filterInput.value = this.pnlThreshold;\n            this.updateBlockHeader('PROFITABLE', 0);\n            filterInput.addEventListener",
            )
            pos_js.write_text(text, encoding="utf-8")
            applied = True

    return True

