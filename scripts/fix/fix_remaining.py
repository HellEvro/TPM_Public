#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Исправление оставшихся ошибок"""

import re

with open('bots.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Исправляем конкретные паттерны
replacements = [
    # Строка 1924
    ('            logger.warning(f"[NEW_AUTO_FILTER] {symbol}: ❌ БЛОКИРОВКА: Обнаружены резкие движения цены (ExitScam)")\n                return False',
     '            logger.warning(f"[NEW_AUTO_FILTER] {symbol}: ❌ БЛОКИРОВКА: Обнаружены резкие движения цены (ExitScam)")\n            return False'),
    
    # Строка 2120
    ('        return new_bot\n        \n            except Exception as e:\n        logger.error(f"[CREATE_BOT] ❌ Ошибка создания бота для {symbol}: {e}")',
     '        return new_bot\n        \n    except Exception as e:\n        logger.error(f"[CREATE_BOT] ❌ Ошибка создания бота для {symbol}: {e}")'),
    
    # Строка 2188
    ('            logger.info(f"[TEST_EXIT_SCAM] {symbol}: ✅ РЕЗУЛЬТАТ: ПРОЙДЕН")\n                    else:\n            logger.warning',
     '            logger.info(f"[TEST_EXIT_SCAM] {symbol}: ✅ РЕЗУЛЬТАТ: ПРОЙДЕН")\n        else:\n            logger.warning'),
    
    # Строка 2231
    ('        # Получаем свечи\n                if not ensure_exchange_initialized():',
     '        # Получаем свечи\n        if not ensure_exchange_initialized():'),
    
    # Строка 2235-2240
    ('            return\n                \n                chart_response = exchange.get_chart_data(symbol, \'6h\', \'30d\')\n                if not chart_response',
     '            return\n                \n        chart_response = exchange.get_chart_data(symbol, \'6h\', \'30d\')\n        if not chart_response'),
    
    ('            return\n        \n                candles = chart_response.get(\'data\', {}).get(\'candles\', [])',
     '            return\n        \n        candles = chart_response.get(\'data\', {}).get(\'candles\', [])'),
    
    # Строка 2256
    ('        # Определяем ОРИГИНАЛЬНЫЙ сигнал на основе только RSI (игнорируя другие фильтры)\n    with bots_data_lock:',
     '        # Определяем ОРИГИНАЛЬНЫЙ сигнал на основе только RSI (игнорируя другие фильтры)\n        with bots_data_lock:'),
    
    # Строка 2713
    ('                logger.debug(f"[NEW_BOT_{self.symbol}] 📈 Обновлена максимальная прибыль: {profit_percent:.2f}%")\n            \n                except Exception as e:\n            logger.error',
     '                logger.debug(f"[NEW_BOT_{self.symbol}] 📈 Обновлена максимальная прибыль: {profit_percent:.2f}%")\n            \n        except Exception as e:\n            logger.error'),
    
    # Строка 2725
    ('                positions_list = exchange_positions[0] if exchange_positions else []\n                else:\n                positions_list',
     '                positions_list = exchange_positions[0] if exchange_positions else []\n            else:\n                positions_list'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open('bots.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all remaining errors!")

