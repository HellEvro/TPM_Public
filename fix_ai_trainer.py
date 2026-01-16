#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправление ошибок линтера в ai_trainer.py
"""

import re

def fix_ai_trainer():
    with open('bot_engine/ai/ai_trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Исправим использование undefined переменных в _register_win_rate_success
    # Найдем функцию и заменим её содержимое

    # Заменим default_target на self.win_rate_targets_default
    content = content.replace('max(default_target, 80.0)', 'max(self.win_rate_targets_default, 80.0)')

    # Теперь заменим всю функцию _register_win_rate_success
    func_pattern = r'(def _register_win_rate_success\(self, symbol: str, achieved_win_rate: float\):.*?)(\n\s*def|\nclass|\Z)'
    func_match = re.search(func_pattern, content, re.DOTALL)

    if func_match:
        func_start = func_match.start(1)
        func_end = func_match.end(1)

        # Получаем текущую функцию
        current_func = content[func_start:func_end]

        # Новая функция
        new_func_content = '''    def _register_win_rate_success(self, symbol: str, achieved_win_rate: float):
        """
        Зафиксировать успешное достижение цели Win Rate и повысить порог на 1%.
        """
        if not self.ai_db:
            return

        try:
            symbol_key = (symbol or '').upper()
            current_target = self._get_win_rate_target(symbol_key)

            # Получаем или создаем запись для символа
            win_rate_data = self.ai_db.get_win_rate_target(symbol_key) or {}
            entry = {
                'target': current_target,
                'symbol': symbol_key,
                'created_at': win_rate_data.get('created_at', datetime.now().isoformat()),
                'last_updated': datetime.now().isoformat()
            }

            # Обновляем существующие поля
            for key, value in win_rate_data.items():
                if key not in entry:
                    entry[key] = value

            if current_target >= 100.0:
                reset_target = max(self.win_rate_targets_default, 80.0)
                if current_target != reset_target:
                    entry['target'] = reset_target
                    entry['last_target_reset_at'] = datetime.now().isoformat()
                    entry['last_target_reset_reason'] = 'reached_100_then_reset'
                    logger.info(
                        f"   🔁 {symbol}: цель Win Rate достигла 100%, сбрасываем до {reset_target:.1f}% "
                        f"для повторного цикла обучения"
                    )
            else:
                if achieved_win_rate >= current_target:
                    new_target = min(current_target + 1.0, 100.0)
                    if new_target > current_target:
                        entry['target'] = new_target
                        entry['last_target_increment_at'] = datetime.now().isoformat()
                        entry['last_target_increment_win_rate'] = achieved_win_rate
                        entry['increments'] = entry.get('increments', 0) + 1
                        logger.info(
                            f"   🚀 {symbol}: цель Win Rate повышена с {current_target:.1f}% до {new_target:.1f}% "
                            f"(достигнуто {achieved_win_rate:.1f}%)"
                        )
                else:
                    entry['target'] = current_target

            # Сохраняем в БД
            self.ai_db.save_win_rate_target(symbol_key, entry)
            self.win_rate_targets_dirty = True
        except Exception as e:
            logger.debug(f"⚠️ Не удалось обновить цель Win Rate для {symbol}: {e}")
'''

        # Заменяем функцию
        content = content.replace(current_func, new_func_content)

    # Сохраняем исправленный файл
    with open('bot_engine/ai/ai_trainer.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("ai_trainer.py исправлен")

if __name__ == '__main__':
    fix_ai_trainer()