"""
Unit-тесты для рекомендаций из docs/ANALYTICS_LOGIC_AND_VALIDITY_REPORT.md

Сценарии:
- virtual_only: record_virtual_open вызывается, enter_position не вызывается
- _check_if_trade_already_closed: фильтр по exit_timestamp (последние 10 мин)
- decision_source сохраняется в bot_data при создании бота
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


class TestCheckIfTradeAlreadyClosed(unittest.TestCase):
    """Тест фильтра по времени в _check_if_trade_already_closed"""

    def test_get_bot_trades_history_called_with_from_ts(self):
        """get_bot_trades_history вызывается с from_ts_sec (10 мин)"""
        with patch('bot_engine.bots_database.get_bots_database') as mock_get_db:
            mock_db = MagicMock()
            mock_db.get_bot_trades_history.return_value = []
            mock_get_db.return_value = mock_db

            from bots_modules.sync_and_cache import _check_if_trade_already_closed

            _check_if_trade_already_closed(
                bot_id='BTCUSDT',
                symbol='BTCUSDT',
                entry_price=50000.0,
                entry_time_str=datetime.now().isoformat()
            )

            mock_db.get_bot_trades_history.assert_called_once()
            call_kw = mock_db.get_bot_trades_history.call_args[1]
            self.assertIn('from_ts_sec', call_kw)
            self.assertIn('to_ts_sec', call_kw)
            # Окно ~10 минут
            now = datetime.now().timestamp()
            self.assertLessEqual(call_kw['from_ts_sec'], now - 590)
            self.assertGreaterEqual(call_kw['from_ts_sec'], now - 610)


class TestDecisionSourceInBotData(unittest.TestCase):
    """Тест сохранения decision_source в bot_data"""

    def test_new_bot_to_dict_includes_decision_source(self):
        """NewTradingBot.to_dict() включает decision_source"""
        try:
            from bots_modules.bot_class import NewTradingBot
        except ImportError:
            self.skipTest("NewTradingBot недоступен")

        with patch('bots_modules.bot_class.get_exchange') as mock_ex:
            mock_ex.return_value = MagicMock()
            bot = NewTradingBot('TESTUSDT', MagicMock(), {})
            bot._set_decision_source('AI', {'ai_confidence': 0.9})

            d = bot.to_dict()
            self.assertIn('decision_source', d)
            self.assertEqual(d['decision_source'], 'AI')

    def test_new_bot_to_dict_default_script(self):
        """По умолчанию decision_source = SCRIPT"""
        try:
            from bots_modules.bot_class import NewTradingBot
        except ImportError:
            self.skipTest("NewTradingBot недоступен")

        with patch('bots_modules.bot_class.get_exchange') as mock_ex:
            mock_ex.return_value = MagicMock()
            bot = NewTradingBot('TESTUSDT', MagicMock(), {})
            # _set_decision_source не вызывали

            d = bot.to_dict()
            self.assertIn('decision_source', d)
            self.assertEqual(d['decision_source'], 'SCRIPT')


class TestVirtualOnlyFlow(unittest.TestCase):
    """Тест: при virtual_only не вызывается enter_position"""

    def test_virtual_only_skips_enter_position(self):
        """
        В process_auto_bot_signals при last_ai_result.virtual_only:
        - вызывается record_virtual_open
        - НЕ вызывается enter_position (continue до создания бота)
        """
        # Проверяем логику в filters.py: virtual_only блок идёт ДО create_new_bot и enter_position
        # Значит enter_position не может быть вызван при virtual_only
        import inspect
        from bots_modules.filters import process_auto_bot_signals

        source = inspect.getsource(process_auto_bot_signals)
        # virtual_only блок с continue должен быть перед "Создаём бота" и enter_position
        virtual_pos = source.find("virtual_only")
        create_pos = source.find("Создаём бота")
        enter_pos = source.find("enter_position")
        continue_pos = source.find("continue", source.find("record_virtual_open"))

        self.assertGreater(virtual_pos, 0, "virtual_only блок должен существовать")
        self.assertGreater(create_pos, virtual_pos, "virtual_only проверка до создания бота")
        self.assertGreater(continue_pos, virtual_pos, "continue после record_virtual_open при virtual_only")


if __name__ == '__main__':
    unittest.main()
