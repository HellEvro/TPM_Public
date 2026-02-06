#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления ботами через AI

Управляет ботами на основе предсказаний AI
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any

logger = logging.getLogger('AI.BotManager')

class AIBotManager:
    """
    Класс для управления ботами через AI
    """

    def __init__(self, bots_service_url: str = 'http://127.0.0.1:5001'):
        """
        Инициализация менеджера ботов

        Args:
            bots_service_url: URL сервиса bots.py
        """
        self.bots_service_url = bots_service_url
        # УДАЛЕНО: self.config_dir - конфиги теперь сохраняются в БД (bot_configs)

        logger.info("✅ AIBotManager инициализирован")

    def _call_bots_api(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Optional[Dict]:
        """Вызов API bots.py"""
        try:
            url = f"{self.bots_service_url}{endpoint}"

            if method == 'GET':
                response = requests.get(url, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=10)
            else:
                return None

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ API {endpoint} вернул статус {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Сервис bots.py недоступен по адресу {self.bots_service_url}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка вызова API {endpoint}: {e}")
            return None

    def get_bots_list(self) -> List[Dict]:
        """Получить список всех ботов"""
        try:
            response = self._call_bots_api('/api/bots/list')
            if response and response.get('success'):
                return response.get('bots', [])
            return []
        except Exception as e:
            logger.error(f"❌ Ошибка получения списка ботов: {e}")
            return []

    def get_bot_status(self, symbol: str) -> Optional[Dict]:
        """Получить статус бота"""
        try:
            response = self._call_bots_api(f'/api/bots/status/{symbol}')
            if response and response.get('success'):
                return response.get('bot', {})
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса бота {symbol}: {e}")
            return None

    def start_bot(self, symbol: str, config: Dict = None) -> bool:
        """
        Запустить бота

        Args:
            symbol: Символ монеты
            config: Конфигурация бота

        Returns:
            True если успешно
        """
        try:
            data = {
                'symbol': symbol,
                'config': config or {}
            }

            response = self._call_bots_api('/api/bots/start', method='POST', data=data)

            if response and response.get('success'):
                logger.info(f"✅ Бот {symbol} запущен")
                return True
            else:
                logger.warning(f"⚠️ Не удалось запустить бота {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка запуска бота {symbol}: {e}")
            return False

    def stop_bot(self, symbol: str) -> bool:
        """
        Остановить бота

        Args:
            symbol: Символ монеты

        Returns:
            True если успешно
        """
        try:
            data = {
                'symbol': symbol,
                'action': 'stop'
            }

            response = self._call_bots_api('/api/bots/control', method='POST', data=data)

            if response and response.get('success'):
                logger.warning(f"✅ Бот {symbol} остановлен")
                return True
            else:
                logger.warning(f"⚠️ Не удалось остановить бота {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка остановки бота {symbol}: {e}")
            return False

    def update_bot_config(self, symbol: str, config: Dict) -> bool:
        """
        Обновить конфигурацию бота

        Args:
            symbol: Символ монеты
            config: Новая конфигурация

        Returns:
            True если успешно
        """
        try:
            data = {
                'symbol': symbol,
                'config': config
            }

            response = self._call_bots_api(
                f'/api/bots/individual-settings/{symbol}',
                method='POST',
                data=data
            )

            if response and response.get('success'):
                logger.info(f"✅ Конфигурация бота {symbol} обновлена")

                # Сохраняем конфигурацию в БД вместо файла
                try:
                    from bot_engine.ai.ai_database import get_ai_database
                    ai_db = get_ai_database()
                    if ai_db:
                        ai_db.save_bot_config(symbol, config)

                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить конфиг в БД: {e}")

                return True
            else:
                logger.warning(f"⚠️ Не удалось обновить конфигурацию бота {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка обновления конфигурации бота {symbol}: {e}")
            return False

    def manage_bots_with_ai(self, predictions: Dict[str, Dict]):
        """
        Управление ботами на основе предсказаний AI

        Args:
            predictions: Словарь предсказаний {symbol: prediction_dict}
        """
        try:
            bots = self.get_bots_list()
            bot_symbols = {bot.get('symbol') for bot in bots}

            def _conf_01(v):
                if v is None:
                    return 0.0
                v = float(v)
                return (v / 100.0) if v > 1 else v

            min_conf_raw = 0.7
            try:
                from bot_engine.config_loader import AIConfig
                min_conf_raw = getattr(AIConfig, 'AI_CONFIDENCE_THRESHOLD', 0.7)
            except Exception:
                pass
            min_conf_01 = _conf_01(min_conf_raw)

            for symbol, prediction in predictions.items():
                signal = prediction.get('signal')
                confidence = prediction.get('confidence', 0)
                conf_01 = _conf_01(confidence)

                if conf_01 < min_conf_01:
                    continue

                bot_status = self.get_bot_status(symbol)

                if signal == 'LONG' or signal == 'SHORT':
                    # Нужно открыть позицию
                    if not bot_status or bot_status.get('status') == 'IDLE':
                        # Запускаем бота если его нет или он в IDLE
                        self.start_bot(symbol)
                        logger.info(f"🤖 AI запустил бота {symbol} (сигнал: {signal}, уверенность: {conf_01:.2%})")

                elif signal == 'WAIT':
                    # Нужно закрыть позицию если открыта
                    if bot_status and bot_status.get('status') != 'IDLE':
                        # Останавливаем бота
                        self.stop_bot(symbol)
                        logger.warning(f"🤖 AI остановил бота {symbol} (сигнал: WAIT)")

        except Exception as e:
            logger.error(f"❌ Ошибка управления ботами через AI: {e}")
