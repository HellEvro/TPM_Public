"""
Патч 010: вход/выход по лимиту RSI — RSI_LIMIT_ENTRY_ENABLED, RSI_LIMIT_EXIT_ENABLED,
RSI_LIMIT_OFFSET_PERCENT, RSI_LIMIT_EXIT_OFFSET_PERCENT в DefaultAutoBotConfig и AutoBotConfig;
расчёт цены лимитного ордера по RSI в trading_bot; подписи в API.
"""
from pathlib import Path

RSI_LIMIT_BLOCK = '''    RSI_LIMIT_ENTRY_ENABLED = True          # Вход лимитным ордером по расчётной цене «RSI = порог»
    RSI_LIMIT_EXIT_ENABLED = True           # Выход лимитным по расчётной цене «RSI = порог выхода»
    RSI_LIMIT_OFFSET_PERCENT = 0.2          # Смещение лимита входа по RSI (%)
    RSI_LIMIT_EXIT_OFFSET_PERCENT = 0.2     # Смещение лимита выхода по RSI (%)

'''


def _add_rsi_limit_to_config(text: str) -> str | None:
    """Добавляет RSI_LIMIT_* в DefaultAutoBotConfig и AutoBotConfig. Возвращает новый текст или None если уже есть."""
    if "RSI_LIMIT_EXIT_OFFSET_PERCENT" in text:
        return None  # Уже всё есть
    # Добавляем RSI_LIMIT_EXIT_OFFSET_PERCENT если есть старый блок (3 строки без 4-й)
    if "RSI_LIMIT_ENTRY_ENABLED" in text and "RSI_LIMIT_OFFSET_PERCENT" in text:
        old3 = "    RSI_LIMIT_OFFSET_PERCENT = 0.2          # Смещение лимита в % (LONG: ниже расчётной, SHORT: выше)"
        new4 = "    RSI_LIMIT_OFFSET_PERCENT = 0.2          # Смещение лимита входа по RSI (%)\n    RSI_LIMIT_EXIT_OFFSET_PERCENT = 0.2     # Смещение лимита выхода по RSI (%)"
        if old3 in text and "RSI_LIMIT_EXIT_OFFSET_PERCENT" not in text:
            text = text.replace(old3, new4, 1)
            return text
        old3b = "    RSI_LIMIT_OFFSET_PERCENT = 0.2          # Смещение лимита входа по RSI (%)"
        if old3b in text and "RSI_LIMIT_EXIT_OFFSET_PERCENT" not in text:
            text = text.replace(
                old3b,
                "    RSI_LIMIT_OFFSET_PERCENT = 0.2          # Смещение лимита входа по RSI (%)\n    RSI_LIMIT_EXIT_OFFSET_PERCENT = 0.2     # Смещение лимита выхода по RSI (%)",
                1,
            )
            return text
    marker = "    # --- Размер позиции и плечо ---"
    if marker not in text:
        return None
    new_text = text.replace(marker, RSI_LIMIT_BLOCK + marker, 2)
    return new_text if new_text != text else None


def _add_rsi_limit_to_autobotconfig(text: str) -> str | None:
    """Добавляет RSI_LIMIT_* в AutoBotConfig если у класса другая структура (нет # --- Размер позиции ---)."""
    if "RSI_LIMIT_EXIT_OFFSET_PERCENT" in text:
        return None
    if "class AutoBotConfig" not in text:
        return None
    # Ищем SYSTEM_TIMEFRAME в блоке AutoBotConfig
    idx = text.find("class AutoBotConfig")
    if idx == -1:
        return None
    rest = text[idx:]
    if "RSI_LIMIT_ENTRY_ENABLED" in rest and "RSI_LIMIT_EXIT_OFFSET_PERCENT" in rest:
        return None
    # Вставляем после SYSTEM_TIMEFRAME = '1m' или RSI_EXIT_SHORT_AGAINST_TREND
    for needle in [
        "    SYSTEM_TIMEFRAME = '1m'                 # Таймфрейм системы",
        "    SYSTEM_TIMEFRAME = ",
        "    RSI_EXIT_SHORT_AGAINST_TREND = 45       # Выход из SHORT против тренда",
    ]:
        pos = rest.find(needle)
        if pos != -1:
            end = rest.find("\n", pos) + 1
            insert_at = end
            if "RSI_LIMIT_ENTRY_ENABLED" not in rest[:insert_at]:
                new_rest = rest[:insert_at] + RSI_LIMIT_BLOCK + rest[insert_at:]
                return text[:idx] + new_rest
    return None


def _patch_trading_bot(text: str) -> str | None:
    """Добавляет import и блок RSI limit entry в trading_bot. Возвращает новый текст или None."""
    if "estimate_price_for_rsi" in text and "rsi_limit_entry_enabled" in text:
        return None
    # Import
    if "from .utils.rsi_utils import estimate_price_for_rsi" not in text:
        old_import = "from .scaling_calculator import calculate_scaling_for_bot\n"
        new_import = "from .scaling_calculator import calculate_scaling_for_bot\nfrom .utils.rsi_utils import estimate_price_for_rsi\n"
        text = text.replace(old_import, new_import, 1)
    # Блок RSI limit entry (после "if force_market_entry:")
    old_block = '            # ✅ АВТОВХОД: при force_market_entry всегда по рынку, лимитные ордера не используем\n            if force_market_entry:\n                self.logger.info(f" {self.symbol}: 🚀 Автовход — вход строго по рынку (лимитные ордера не используются)")'
    new_block = '''            # ✅ АВТОВХОД: при force_market_entry — по рынку или по лимиту RSI (если включено)
            if force_market_entry:
                rsi_limit_entry = self.config.get('rsi_limit_entry_enabled', False)
                if rsi_limit_entry:
                    # Расчёт цены лимитного входа по RSI и размещение лимитника
                    tf_use = self.config.get('entry_timeframe') or get_current_timeframe()
                    try:
                        chart_response = self.exchange.get_chart_data(self.symbol, tf_use, '14d')
                        candles = chart_response.get('data', {}).get('candles', []) if chart_response and chart_response.get('success') else []
                        if candles and len(candles) >= 15:
                            closes = [float(c.get('close', 0)) for c in candles]
                            threshold = (self.config.get('rsi_long_threshold') if side == 'LONG' else self.config.get('rsi_short_threshold'))
                            if threshold is None:
                                threshold = 29 if side == 'LONG' else 71
                            limit_price = estimate_price_for_rsi(closes, threshold, 14, side)
                            if limit_price and limit_price > 0:
                                offset_pct = float(self.config.get('rsi_limit_offset_percent', 0.2) or 0.2) / 100.0
                                if side == 'LONG':
                                    limit_price = limit_price * (1.0 - offset_pct)
                                else:
                                    limit_price = limit_price * (1.0 + offset_pct)
                                # Не выставляем второй лимитник, если уже есть открытый ордер по этой стороне
                                if hasattr(self.exchange, 'get_open_orders'):
                                    try:
                                        open_orders = self.exchange.get_open_orders(self.symbol)
                                        limit_side = 'Buy' if side == 'LONG' else 'Sell'
                                        if any(o.get('order_type', '').lower() == 'limit' and o.get('side') == limit_side for o in open_orders):
                                            self.logger.info(f" {self.symbol}: Лимитный ордер входа по RSI уже есть, ждём исполнения")
                                            return {'success': True, 'message': 'limit_order_pending'}
                                    except Exception:
                                        pass
                                quantity = self._calculate_position_size()
                                if quantity:
                                    leverage = self.config.get('leverage')
                                    order_result = self.exchange.place_order(
                                        symbol=self.symbol,
                                        side=side,
                                        quantity=quantity,
                                        order_type='limit',
                                        price=limit_price,
                                        leverage=leverage
                                    )
                                    if order_result.get('success'):
                                        self.logger.info(f" {self.symbol}: Лимитный вход по RSI размещён @ {limit_price} (порог RSI={threshold})")
                                        return {'success': True, 'message': 'limit_order_placed', 'order_id': order_result.get('order_id'), 'price': limit_price}
                                    self.logger.warning(f" {self.symbol}: Не удалось разместить лимит по RSI: {order_result.get('message', '')}")
                        else:
                            self.logger.debug(f" {self.symbol}: Недостаточно свечей для расчёта цены по RSI, вход по рынку")
                    except Exception as e:
                        self.logger.warning(f" {self.symbol}: Ошибка расчёта лимита по RSI: {e}, вход по рынку")
                if not rsi_limit_entry:
                    self.logger.info(f" {self.symbol}: 🚀 Автовход — вход по рынку (лимит по RSI выключен)")'''
    if old_block in text and new_block not in text:
        text = text.replace(old_block, new_block, 1)
    elif "rsi_limit_entry_enabled" not in text:
        return None
    return text


def _patch_api_endpoints(text: str) -> str | None:
    """Добавляет подписи rsi_limit_* в API. Возвращает новый текст или None."""
    if "'rsi_limit_exit_offset_percent':" in text:
        return None
    if "'rsi_limit_entry_enabled':" in text:
        old = "    'rsi_limit_offset_percent': 'Смещение лимита по RSI (%)',\n    'rsi_time_filter_enabled':"
        new = "    'rsi_limit_offset_percent': 'Смещение лимита входа по RSI (%)',\n    'rsi_limit_exit_offset_percent': 'Смещение лимита выхода по RSI (%)',\n    'rsi_time_filter_enabled':"
        if old in text:
            return text.replace(old, new, 1)
        return None
    old = "    'rsi_exit_min_move_percent': 'Мин. % движения для выхода по RSI (блокирует до достижения)',\n    'rsi_time_filter_enabled':"
    new = "    'rsi_exit_min_move_percent': 'Мин. % движения для выхода по RSI (блокирует до достижения)',\n    'rsi_limit_entry_enabled': 'Вход лимитом по цене RSI (расчёт цены по порогу)',\n    'rsi_limit_exit_enabled': 'Выход лимитом по цене RSI',\n    'rsi_limit_offset_percent': 'Смещение лимита входа по RSI (%)',\n    'rsi_limit_exit_offset_percent': 'Смещение лимита выхода по RSI (%)',\n    'rsi_time_filter_enabled':"
    if old not in text:
        return None
    return text.replace(old, new, 1)


def apply(project_root: Path) -> bool:
    applied = False
    example_path = project_root / "configs" / "bot_config.example.py"
    if example_path.exists():
        text = example_path.read_text(encoding="utf-8")
        new_text = _add_rsi_limit_to_config(text)
        if new_text is not None:
            example_path.write_text(new_text, encoding="utf-8")
            applied = True
        else:
            new_text = _add_rsi_limit_to_autobotconfig(text)
            if new_text is not None:
                example_path.write_text(new_text, encoding="utf-8")
                applied = True

    config_path = project_root / "configs" / "bot_config.py"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        new_text = _add_rsi_limit_to_config(text)
        if new_text is not None:
            config_path.write_text(new_text, encoding="utf-8")
            applied = True
        else:
            new_text = _add_rsi_limit_to_autobotconfig(text)
            if new_text is not None:
                config_path.write_text(new_text, encoding="utf-8")
                applied = True

    # 3. bot_engine/trading_bot.py
    tb_path = project_root / "bot_engine" / "trading_bot.py"
    if tb_path.exists():
        text = tb_path.read_text(encoding="utf-8")
        new_text = _patch_trading_bot(text)
        if new_text is not None:
            tb_path.write_text(new_text, encoding="utf-8")
            applied = True

    # 4. bots_modules/api_endpoints.py
    api_path = project_root / "bots_modules" / "api_endpoints.py"
    if api_path.exists():
        text = api_path.read_text(encoding="utf-8")
        new_text = _patch_api_endpoints(text)
        if new_text is not None:
            api_path.write_text(new_text, encoding="utf-8")
            applied = True

    return True
